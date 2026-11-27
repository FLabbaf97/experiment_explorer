"""Any cached file can be listed, and text files can be read (issue #7)."""

import pytest

from helpers import RUN, write_experiment

REPORT_EXPERIMENT = {
    "README.md": "# Top\n",
    "analysis/REPORT.md": "# Findings\nModel A beats chance.\n",
    "analysis/deep/notes.txt": "plain notes\n",
    "analysis/summary.json": '{"roc_auc": 0.9}\n',
    f"{RUN}/results/test_results.csv": "metric,value\nroc_auc,0.9\n",
}


def test_list_cached_files_lists_every_file_with_its_size(call_tool, cache_dir):
    root = write_experiment(cache_dir, "exp", REPORT_EXPERIMENT)
    (root / f"{RUN}/artifacts").mkdir(parents=True)
    (root / f"{RUN}/artifacts/model.pkl").write_bytes(b"\x80\x04binary\x00")
    (root / ".s3_manifest.json").write_text("{}")

    result = call_tool("list_cached_files", experiment_name="exp")

    sizes = {entry["relative_path"]: entry["size_bytes"] for entry in result["files"]}
    assert sizes == {
        **{path: len(body.encode()) for path, body in REPORT_EXPERIMENT.items()},
        f"{RUN}/artifacts/model.pkl": 9,
    }
    assert result["count"] == 6
    assert result["truncated"] is False


@pytest.mark.parametrize(
    "relative_path", ["analysis/REPORT.md", "analysis/deep/notes.txt", "analysis/summary.json"]
)
def test_read_cached_file_returns_text_files_at_any_depth(call_tool, cache_dir, relative_path):
    write_experiment(cache_dir, "exp", REPORT_EXPERIMENT)

    result = call_tool("read_cached_file", experiment_name="exp", relative_path=relative_path)

    assert result["content"] == REPORT_EXPERIMENT[relative_path]
    assert result["size_bytes"] == len(REPORT_EXPERIMENT[relative_path].encode())
    assert result["truncated"] is False


@pytest.mark.parametrize(
    ("max_bytes", "content", "truncated"),
    [(5, "abcde", True), (10, "abcdefghij", False), (50, "abcdefghij", False)],
)
def test_read_cached_file_flags_content_cut_short(
    call_tool, cache_dir, max_bytes, content, truncated
):
    write_experiment(cache_dir, "exp", {"notes.txt": "abcdefghij"})

    result = call_tool(
        "read_cached_file", experiment_name="exp", relative_path="notes.txt", max_bytes=max_bytes
    )

    assert result["content"] == content
    assert result["bytes_returned"] == len(content)
    assert result["size_bytes"] == 10
    assert result["truncated"] is truncated


def test_read_cached_file_never_splits_a_character(call_tool, cache_dir):
    write_experiment(cache_dir, "exp", {"notes.md": "abécd"})  # é is two bytes

    result = call_tool(
        "read_cached_file", experiment_name="exp", relative_path="notes.md", max_bytes=3
    )

    assert result["content"] == "ab"
    assert result["truncated"] is True


@pytest.mark.parametrize("max_bytes", [0, -1])
def test_read_cached_file_rejects_limits_below_one(call_tool, cache_dir, max_bytes):
    write_experiment(cache_dir, "exp", {"notes.txt": "abc"})

    result = call_tool(
        "read_cached_file", experiment_name="exp", relative_path="notes.txt", max_bytes=max_bytes
    )

    assert "max_bytes" in result["error"]


@pytest.mark.parametrize(
    ("relative_path", "body"),
    [
        ("artifacts/model.pkl", b"\x80\x04\x95binary"),
        ("plots/roc.png", b"\x89PNG\r\n\x1a\n"),
        ("artifacts/model.json.gz", b"\x1f\x8b\x08"),
        ("disguised.txt", b"text\x00with a null byte"),
    ],
)
def test_binary_files_are_listed_but_never_opened(call_tool, cache_dir, relative_path, body):
    root = write_experiment(cache_dir, "exp", {"README.md": "# r\n"})
    (root / relative_path).parent.mkdir(parents=True, exist_ok=True)
    (root / relative_path).write_bytes(body)

    listed = call_tool("list_cached_files", experiment_name="exp")
    result = call_tool("read_cached_file", experiment_name="exp", relative_path=relative_path)

    assert relative_path in [entry["relative_path"] for entry in listed["files"]]
    assert "never opened" in result["error"]
    assert "content" not in result


def test_read_cached_file_points_csvs_to_the_metrics_tool(call_tool, cache_dir):
    write_experiment(cache_dir, "exp", REPORT_EXPERIMENT)

    result = call_tool(
        "read_cached_file",
        experiment_name="exp",
        relative_path=f"{RUN}/results/test_results.csv",
    )

    assert "read_experiment_metrics" in result["error"]


def test_read_cached_file_reports_missing_files(call_tool, cache_dir):
    write_experiment(cache_dir, "exp", REPORT_EXPERIMENT)

    result = call_tool("read_cached_file", experiment_name="exp", relative_path="nope.md")

    assert "not found" in result["error"]


@pytest.mark.parametrize(("max_files", "truncated"), [(2, True), (3, False), (5, False)])
def test_list_cached_files_is_truncated_only_when_more_files_exist(
    call_tool, cache_dir, max_files, truncated
):
    write_experiment(cache_dir, "exp", {"a.md": "1", "b/c.txt": "2", "d/e/f.json": "3"})

    result = call_tool("list_cached_files", experiment_name="exp", max_files=max_files)

    assert result["count"] == min(max_files, 3)
    assert result["total_count"] == 3
    assert result["truncated"] is truncated


def test_list_cached_files_skips_links_that_leave_the_experiment(call_tool, cache_dir, tmp_path):
    root = write_experiment(cache_dir, "exp", {"README.md": "# r\n"})
    outside = write_experiment(tmp_path, "outside", {"secret.md": "s", "dir/secret.txt": "s"})
    (root / "link.md").symlink_to(outside / "secret.md")
    (root / "linked_dir").symlink_to(outside / "dir")

    result = call_tool("list_cached_files", experiment_name="exp")

    assert [entry["relative_path"] for entry in result["files"]] == ["README.md"]


def test_summarize_finds_reports_and_readmes_at_any_depth(call_tool, cache_dir):
    write_experiment(cache_dir, "exp", REPORT_EXPERIMENT)

    summary = call_tool("summarize_experiment", experiment_name="exp")

    assert summary["docs"] == ["README.md", "analysis/REPORT.md"]

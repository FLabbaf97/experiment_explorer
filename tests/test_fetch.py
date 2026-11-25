import json

import pytest

from helpers import BUCKET, STANDARD_EXPERIMENT, put_experiment


def test_list_experiments_shows_top_level_folders(call_tool, s3):
    put_experiment(s3, "exp_a")
    put_experiment(s3, "exp_b")

    result = call_tool("list_experiments")

    assert result["experiments"] == ["exp_a", "exp_b"]
    assert result["bucket"] == BUCKET


def test_list_experiment_files_returns_relative_paths(call_tool, s3):
    put_experiment(s3, "exp")
    put_experiment(s3, "exp_other")

    result = call_tool("list_experiment_files", experiment_name="exp")

    assert sorted(f["relative_path"] for f in result["files"]) == sorted(STANDARD_EXPERIMENT)


def test_fetch_downloads_experiment_and_summarizes_it(call_tool, s3, cache_dir):
    put_experiment(s3, "exp")

    result = call_tool("fetch_experiment", experiment_name="exp")

    assert result["downloaded_count"] == len(STANDARD_EXPERIMENT)
    for relative_path, body in STANDARD_EXPERIMENT.items():
        assert (cache_dir / "exp" / relative_path).read_text() == body
    assert result["experiment_summary"]["run_count"] == 1


def test_fetch_again_skips_unchanged_files(call_tool, s3):
    put_experiment(s3, "exp")
    call_tool("fetch_experiment", experiment_name="exp")

    result = call_tool("fetch_experiment", experiment_name="exp")

    assert result["downloaded_count"] == 0
    assert result["skipped_count"] == len(STANDARD_EXPERIMENT)


def test_fetch_does_not_touch_sibling_experiments(call_tool, s3, cache_dir):
    put_experiment(s3, "exp")
    put_experiment(s3, "exp_other")

    call_tool("fetch_experiment", experiment_name="exp")

    assert [p.name for p in cache_dir.iterdir()] == ["exp"]


def test_fetch_blocks_keys_that_escape_the_experiment_folder(call_tool, s3, cache_dir, tmp_path):
    unsafe_keys = ["exp/../../escaped.txt", "exp//absolute.txt", "exp/a/.."]
    for key in unsafe_keys:
        s3.put_object(Bucket=BUCKET, Key=key, Body=b"x")
    s3.put_object(Bucket=BUCKET, Key="exp/safe.csv", Body=b"ok")

    result = call_tool("fetch_experiment", experiment_name="exp")

    assert sorted(result["blocked_keys"]) == sorted(unsafe_keys)
    assert result["downloaded_files"] == ["safe.csv"]
    assert not (tmp_path / "escaped.txt").exists()
    assert not (cache_dir / "escaped.txt").exists()


def test_fetch_unknown_experiment_returns_error_and_leaves_no_folder(call_tool, s3, cache_dir):
    put_experiment(s3, "exp")

    result = call_tool("fetch_experiment", experiment_name="typo_exp")

    assert "not found" in result["error"]
    assert not (cache_dir / "typo_exp").exists()
    assert call_tool("list_cached_experiments")["experiments"] == []


def test_fetch_redownloads_a_changed_file_of_the_same_size(call_tool, s3, cache_dir):
    put_experiment(s3, "exp", {"metrics.csv": "a\n1\n"})
    call_tool("fetch_experiment", experiment_name="exp")
    put_experiment(s3, "exp", {"metrics.csv": "a\n2\n"})

    result = call_tool("fetch_experiment", experiment_name="exp")

    assert result["downloaded_files"] == ["metrics.csv"]
    assert (cache_dir / "exp" / "metrics.csv").read_text() == "a\n2\n"


def test_forced_refresh_removes_files_deleted_from_s3_but_keeps_local_files(
    call_tool, s3, cache_dir
):
    put_experiment(s3, "exp", {"keep.csv": "a\n1\n", "old/gone.csv": "a\n1\n"})
    call_tool("fetch_experiment", experiment_name="exp")
    (cache_dir / "exp" / "my_notes.md").write_text("local only")
    s3.delete_object(Bucket=BUCKET, Key="exp/old/gone.csv")

    result = call_tool("fetch_experiment", experiment_name="exp", force_refresh=True)

    assert result["removed_files"] == ["old/gone.csv"]
    assert not (cache_dir / "exp" / "old").exists()
    assert (cache_dir / "exp" / "keep.csv").exists()
    assert (cache_dir / "exp" / "my_notes.md").read_text() == "local only"


def test_normal_fetch_reports_files_deleted_from_s3_without_removing_them(call_tool, s3, cache_dir):
    put_experiment(s3, "exp", {"keep.csv": "a\n1\n", "gone.csv": "a\n1\n"})
    call_tool("fetch_experiment", experiment_name="exp")
    s3.delete_object(Bucket=BUCKET, Key="exp/gone.csv")

    result = call_tool("fetch_experiment", experiment_name="exp")

    assert result["missing_in_s3"] == ["gone.csv"]
    assert result["removed_files"] == []
    assert (cache_dir / "exp" / "gone.csv").exists()


@pytest.mark.parametrize("manifest", ["[1, 2]", '{"metrics.csv": "oops"}', "not json"])
def test_fetch_recovers_from_a_damaged_manifest(call_tool, s3, cache_dir, manifest):
    put_experiment(s3, "exp", {"metrics.csv": "a\n1\n"})
    call_tool("fetch_experiment", experiment_name="exp")
    (cache_dir / "exp" / ".s3_manifest.json").write_text(manifest)

    result = call_tool("fetch_experiment", experiment_name="exp")

    assert result["downloaded_files"] == ["metrics.csv"]


def test_forced_refresh_only_reports_files_it_actually_removed(call_tool, s3, cache_dir):
    put_experiment(s3, "exp", {"keep.csv": "a\n1\n"})
    call_tool("fetch_experiment", experiment_name="exp")
    (cache_dir / "exp" / "edited_locally.csv").write_text("my own longer content")
    (cache_dir / "exp" / "folder").mkdir()
    manifest_path = cache_dir / "exp" / ".s3_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    stale = {"etag": '"x"', "size": 3}
    for key in ("edited_locally.csv", "folder", "../../outside.txt"):
        manifest[key] = stale
    manifest_path.write_text(json.dumps(manifest))

    result = call_tool("fetch_experiment", experiment_name="exp", force_refresh=True)

    assert result["removed_files"] == []
    assert (cache_dir / "exp" / "edited_locally.csv").exists()
    assert (cache_dir / "exp" / "folder").is_dir()


def test_list_files_of_unknown_experiment_returns_not_found(call_tool, s3):
    put_experiment(s3, "exp")

    assert "not found" in call_tool("list_experiment_files", experiment_name="typo")["error"]


def test_forced_refresh_of_an_experiment_deleted_from_s3_clears_its_files(call_tool, s3, cache_dir):
    put_experiment(s3, "exp", {"a.csv": "a\n1\n"})
    put_experiment(s3, "other", {"b.csv": "b\n1\n"})
    call_tool("fetch_experiment", experiment_name="exp")
    (cache_dir / "exp" / "my_notes.md").write_text("local only")
    s3.delete_object(Bucket=BUCKET, Key="exp/a.csv")

    result = call_tool("fetch_experiment", experiment_name="exp", force_refresh=True)

    assert "not found" in result["error"]
    assert "removed 1" in result["error"]
    assert not (cache_dir / "exp" / "a.csv").exists()
    assert (cache_dir / "exp" / "my_notes.md").exists()

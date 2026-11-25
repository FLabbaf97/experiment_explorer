import pytest

from experiment_explorer.paths import validate_experiment_name

UNSAFE_NAMES = [
    "",
    " ",
    "/",
    "//",
    ".",
    "..",
    "../x",
    "a/../b",
    "a/./b",
    "a//b",
    "/tmp/x",
    "C:/x",
    "a\\b",
    " a",
    "a ",
    "a\0b",
]


@pytest.mark.parametrize("name", UNSAFE_NAMES)
def test_unsafe_names_are_rejected(name):
    with pytest.raises(ValueError):
        validate_experiment_name(name)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("exp", "exp"),
        ("exp/", "exp"),
        ("parent/exp", "parent/exp"),
        ("2026-01-01_run", "2026-01-01_run"),
    ],
)
def test_safe_names_are_normalized(name, expected):
    assert validate_experiment_name(name) == expected


@pytest.mark.parametrize("name", ["", "/", " ", "..", "/tmp/x"])
@pytest.mark.parametrize(
    ("tool", "extra"),
    [
        ("fetch_experiment", {}),
        ("list_experiment_files", {}),
        ("summarize_experiment", {}),
        ("read_experiment_metrics", {"relative_path": "m.csv"}),
        ("read_experiment_config", {"relative_path": "c.yaml"}),
    ],
)
def test_every_tool_rejects_unsafe_names(call_tool, s3, cache_dir, tool, extra, name):
    result = call_tool(tool, experiment_name=name, **extra)

    assert "error" in result
    assert list(cache_dir.iterdir()) == []

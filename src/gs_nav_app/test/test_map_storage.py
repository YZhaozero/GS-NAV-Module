from datetime import datetime
from pathlib import Path

from gs_nav_app.map_storage import default_map_directory, unique_map_file


def test_workspace_gs_nav_map_is_the_default(monkeypatch):
    monkeypatch.delenv("GS_NAV_MAP_DIR", raising=False)
    expected = Path(__file__).resolve().parents[2] / "gs_nav_map"
    assert default_map_directory() == expected.resolve()


def test_configured_map_directory_has_priority(tmp_path, monkeypatch):
    environment_path = tmp_path / "environment"
    configured_path = tmp_path / "configured"
    monkeypatch.setenv("GS_NAV_MAP_DIR", str(environment_path))
    assert default_map_directory(str(configured_path)) == configured_path


def test_pcd_and_grid_names_are_unique_in_the_same_directory(tmp_path):
    now = datetime(2026, 9, 18, 12, 34, 56, 789000)
    pcd = unique_map_file(tmp_path, "factory map", ".pcd", now)
    yaml_path = unique_map_file(tmp_path, "factory map", ".yaml", now)
    assert pcd.name == "factory_map_20260918_123456_789.pcd"
    assert yaml_path.name == "factory_map_20260918_123456_789.yaml"

    yaml_path.with_suffix(".pgm").touch()
    repeated = unique_map_file(tmp_path, "factory map", ".yaml", now)
    assert repeated.name == "factory_map_20260918_123456_789_01.yaml"

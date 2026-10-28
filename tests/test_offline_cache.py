from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import relagent.relbench_loader as loader


def test_explicit_snapshot_is_loaded_without_name_resolution(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "rel-arxiv"
    dataset_dir.mkdir()
    (dataset_dir / "manifest.yaml").write_text("name: rel-arxiv\n")
    dataset = SimpleNamespace(get_db=Mock(return_value=object()))
    load = Mock(return_value=dataset)
    monkeypatch.setenv("RELBENCH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(loader.relbench, "load_dataset", load)
    loaded = loader.RelBenchLoader("rel-arxiv")
    assert loaded.dataset is dataset
    load.assert_called_once_with(str(dataset_dir))


def test_missing_local_dataset_fails_without_network_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("RELBENCH_DATA_DIR", str(tmp_path))
    load = Mock()
    monkeypatch.setattr(loader.relbench, "load_dataset", load)
    with pytest.raises(ValueError, match="Dataset missing from local cache"):
        loader.RelBenchLoader("rel-arxiv")
    load.assert_not_called()


def test_task_protected_database_is_loaded_only_once(monkeypatch):
    protected = object()
    task_dataset = SimpleNamespace(get_db=Mock(return_value=protected))
    dataset = SimpleNamespace(get_db=Mock(), load_task=Mock(return_value=SimpleNamespace(dataset=task_dataset)))
    monkeypatch.delenv('RELBENCH_DATA_DIR', raising=False)
    monkeypatch.setattr(loader.relbench, 'load_dataset', Mock(return_value=dataset))
    loaded = loader.RelBenchLoader('rel-ratebeer', 'user-churn')
    dataset.get_db.assert_not_called()
    task_dataset.get_db.assert_not_called()
    assert loaded.get_database() is protected
    assert loaded.get_database() is protected
    task_dataset.get_db.assert_called_once()

"""Tests for edge model manager module."""

import pytest

from src.edge.model_manager import EdgeModelManager, ModelVersion


class TestEdgeModelManager:
    """Test suite for EdgeModelManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = EdgeModelManager(
            model_dir="models/",
            max_versions=3,
            validation_threshold=0.80,
        )

    def test_register_model(self):
        """Test model registration."""
        mv = self.manager.register_model(
            model_name="anomaly_detector",
            version="1.0.0",
            metrics={"accuracy": 0.95},
        )
        assert isinstance(mv, ModelVersion)
        assert mv.model_name == "anomaly_detector"
        assert mv.version == "1.0.0"
        assert mv.is_active is True

    def test_get_active_version(self):
        """Test retrieving active model version."""
        self.manager.register_model("test_model", "1.0.0", {"accuracy": 0.9})
        self.manager.register_model("test_model", "2.0.0", {"accuracy": 0.95})

        active = self.manager.get_active_version("test_model")
        assert active is not None
        assert active.version == "2.0.0"

    def test_rollback(self):
        """Test model rollback."""
        self.manager.register_model("test_model", "1.0.0", {"accuracy": 0.9})
        self.manager.register_model("test_model", "2.0.0", {"accuracy": 0.95})

        success = self.manager.rollback("test_model")
        assert success is True

        active = self.manager.get_active_version("test_model")
        assert active.version == "1.0.0"

    def test_rollback_no_history(self):
        """Test rollback with insufficient history fails gracefully."""
        self.manager.register_model("test_model", "1.0.0", {"accuracy": 0.9})
        success = self.manager.rollback("test_model")
        assert success is False

    def test_validate_model_update(self):
        """Test model validation against threshold."""
        assert self.manager.validate_model_update({"accuracy": 0.90}) is True
        assert self.manager.validate_model_update({"accuracy": 0.70}) is False

    def test_max_versions_pruning(self):
        """Test that old versions are pruned beyond max_versions."""
        for i in range(5):
            self.manager.register_model("test_model", f"{i}.0.0", {"accuracy": 0.9})

        history = self.manager.get_version_history("test_model")
        assert len(history) == 3  # max_versions = 3

    def test_model_health_check(self):
        """Test model health status reporting."""
        self.manager.register_model("test_model", "1.0.0", {"accuracy": 0.9})
        self.manager.record_inference("test_model", 5.0, success=True)
        self.manager.record_inference("test_model", 3.0, success=True)

        health = self.manager.check_model_health("test_model")
        assert health["status"] == "healthy"
        assert health["inference_count"] == 2

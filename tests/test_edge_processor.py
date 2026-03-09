"""Tests for edge processor module."""

import pytest

from src.edge.edge_processor import EdgeProcessor, ProcessingResult
from src.sensors.data_ingestion import MachineState


class TestEdgeProcessor:
    """Test suite for EdgeProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.edge = EdgeProcessor(
            device_id="test-edge-001",
            batch_size=50,
            window_size=20,
        )

    def test_register_machine(self):
        """Test machine registration."""
        self.edge.register_machine("M-001", MachineState.NORMAL, random_seed=42)
        status = self.edge.get_machine_status("M-001")
        assert status["machine_id"] == "M-001"
        assert status["state"] == "normal"

    def test_unregistered_machine_returns_none(self):
        """Test processing unregistered machine returns None."""
        result = self.edge.process_cycle("nonexistent")
        assert result is None

    def test_process_cycle_before_training(self):
        """Test processing before model training still returns result."""
        self.edge.register_machine("M-001", random_seed=42)
        result = self.edge.process_cycle("M-001")
        assert result is not None
        assert result.machine_id == "M-001"
        assert len(result.sensor_readings) > 0
        # No anomaly/RUL predictions before training
        assert result.anomaly_result is None

    def test_full_pipeline_with_training(self):
        """Test complete pipeline: register, train, process."""
        self.edge.register_machine("M-001", MachineState.NORMAL, random_seed=42)
        self.edge.register_machine("M-002", MachineState.DEGRADING, random_seed=43)

        # Train models
        metrics = self.edge.train_models(n_training_machines=10, max_life_cycles=100)
        assert "anomaly_detector" in metrics
        assert "rul_predictor" in metrics

        # Run enough cycles to build window
        for _ in range(25):
            results = self.edge.process_all_machines()

        assert len(results) == 2
        for result in results:
            assert isinstance(result, ProcessingResult)
            assert result.sensor_readings
            # After sufficient cycles with training, anomaly detection should work
            assert result.anomaly_result is not None
            assert result.rul_prediction is not None

    def test_result_serialization(self):
        """Test ProcessingResult serialization."""
        self.edge.register_machine("M-001", random_seed=42)
        self.edge.train_models(n_training_machines=5, max_life_cycles=50)

        for _ in range(10):
            self.edge.process_cycle("M-001")

        result = self.edge.process_cycle("M-001")
        result_dict = result.to_dict()

        assert "machine_id" in result_dict
        assert "timestamp" in result_dict
        assert "processing_time_ms" in result_dict

    def test_stats(self):
        """Test edge processor statistics."""
        self.edge.register_machine("M-001", random_seed=42)
        self.edge.process_cycle("M-001")

        stats = self.edge.get_stats()
        assert stats["device_id"] == "test-edge-001"
        assert stats["registered_machines"] == 1
        assert stats["total_cycles"] == 1

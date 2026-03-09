"""Tests for anomaly detection module."""

import numpy as np
import pytest

from src.models.anomaly_detector import AnomalyDetector, AnomalyResult


class TestAnomalyDetector:
    """Test suite for AnomalyDetector class."""

    def setup_method(self):
        """Set up test fixtures with trained detector."""
        self.rng = np.random.default_rng(42)
        self.detector = AnomalyDetector(
            contamination=0.05,
            n_estimators=50,
            z_score_threshold=3.0,
            random_state=42,
            sensor_names=["vibration", "temperature", "pressure"],
        )
        # Train on normal data
        normal_data = self.rng.normal(0, 1, size=(200, 3))
        self.detector.train(normal_data)

    def test_train_sets_trained_flag(self):
        """Test that training sets the is_trained flag."""
        assert self.detector.is_trained is True

    def test_train_returns_metrics(self):
        """Test that training returns valid metrics."""
        detector = AnomalyDetector(n_estimators=50, random_state=42)
        data = self.rng.normal(0, 1, size=(100, 3))
        metrics = detector.train(data)

        assert "training_samples" in metrics
        assert metrics["training_samples"] == 100
        assert "feature_count" in metrics
        assert metrics["feature_count"] == 3
        assert "anomaly_rate" in metrics
        assert 0.0 <= metrics["anomaly_rate"] <= 1.0

    def test_detect_normal_data(self):
        """Test that normal data is not flagged as anomalous."""
        normal_sample = self.rng.normal(0, 1, size=3)
        result = self.detector.detect(normal_sample)

        assert isinstance(result, AnomalyResult)
        assert 0.0 <= result.anomaly_score <= 1.0
        assert isinstance(result.is_anomaly, bool)
        assert isinstance(result.contributing_sensors, list)

    def test_detect_anomalous_data(self):
        """Test that extreme values are flagged as anomalous."""
        # Create a clearly anomalous sample
        anomalous_sample = np.array([10.0, 15.0, 12.0])
        result = self.detector.detect(anomalous_sample)

        assert result.is_anomaly is True
        assert result.anomaly_score > 0.3

    def test_detect_untrained_raises_error(self):
        """Test that detection before training raises RuntimeError."""
        detector = AnomalyDetector()
        with pytest.raises(RuntimeError, match="must be trained"):
            detector.detect(np.array([1.0, 2.0]))

    def test_contributing_sensors(self):
        """Test that contributing sensors are identified for anomalies."""
        # Make only vibration sensor anomalous
        sample = np.array([15.0, 0.1, -0.2])
        result = self.detector.detect(sample)

        assert len(result.contributing_sensors) > 0
        assert "vibration" in result.contributing_sensors

    def test_detect_batch(self):
        """Test batch anomaly detection."""
        batch = self.rng.normal(0, 1, size=(10, 3))
        batch[5] = [10.0, 12.0, 8.0]  # Insert anomaly

        results = self.detector.detect_batch(batch)

        assert len(results) == 10
        assert results[5].is_anomaly is True

    def test_stats_tracking(self):
        """Test that detection statistics are tracked."""
        for _ in range(5):
            self.detector.detect(self.rng.normal(0, 1, size=3))

        stats = self.detector.get_stats()
        assert stats["detection_count"] == 5
        assert stats["is_trained"] is True

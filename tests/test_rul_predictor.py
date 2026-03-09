"""Tests for RUL prediction module."""

import numpy as np
import pytest

from src.models.rul_predictor import RemainingUsefulLifePredictor, RULPrediction


class TestRULPredictor:
    """Test suite for RemainingUsefulLifePredictor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.predictor = RemainingUsefulLifePredictor(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
        # Train on synthetic data
        features, rul_labels = RemainingUsefulLifePredictor.generate_degradation_dataset(
            n_machines=20,
            max_life_cycles=100,
            n_sensors=5,
            random_state=42,
        )
        self.predictor.train(features, rul_labels, cross_validate=True)

    def test_generate_degradation_dataset(self):
        """Test synthetic dataset generation."""
        features, labels = RemainingUsefulLifePredictor.generate_degradation_dataset(
            n_machines=5, max_life_cycles=50, n_sensors=3, random_state=42
        )
        assert features.ndim == 2
        assert labels.ndim == 1
        assert features.shape[0] == labels.shape[0]
        assert features.shape[0] > 0
        assert np.all(labels >= 0)

    def test_train_returns_metrics(self):
        """Test that training returns valid metrics."""
        predictor = RemainingUsefulLifePredictor(n_estimators=30, random_state=42)
        features, labels = RemainingUsefulLifePredictor.generate_degradation_dataset(
            n_machines=10, max_life_cycles=50, random_state=42
        )
        metrics = predictor.train(features, labels)

        assert "train_r2_score" in metrics
        assert "training_samples" in metrics
        assert metrics["training_samples"] > 0

    def test_predict_returns_prediction(self):
        """Test prediction returns RULPrediction object."""
        features, _ = RemainingUsefulLifePredictor.generate_degradation_dataset(
            n_machines=1, max_life_cycles=50, random_state=99
        )
        pred = self.predictor.predict(features[0])

        assert isinstance(pred, RULPrediction)
        assert pred.rul_cycles >= 0
        assert pred.rul_hours >= 0
        assert pred.confidence_lower <= pred.rul_cycles
        assert pred.confidence_upper >= pred.rul_cycles
        assert 0.0 <= pred.health_index <= 1.0

    def test_predict_untrained_raises_error(self):
        """Test prediction before training raises RuntimeError."""
        predictor = RemainingUsefulLifePredictor()
        with pytest.raises(RuntimeError, match="must be trained"):
            predictor.predict(np.array([1.0, 2.0, 3.0]))

    def test_prediction_serialization(self):
        """Test RULPrediction serializes to dictionary."""
        features, _ = RemainingUsefulLifePredictor.generate_degradation_dataset(
            n_machines=1, max_life_cycles=50, random_state=99
        )
        pred = self.predictor.predict(features[0])
        pred_dict = pred.to_dict()

        assert "rul_cycles" in pred_dict
        assert "rul_hours" in pred_dict
        assert "confidence_lower" in pred_dict
        assert "confidence_upper" in pred_dict
        assert "health_index" in pred_dict

    def test_feature_importances(self):
        """Test feature importance retrieval."""
        importances = self.predictor.get_feature_importances()
        assert len(importances) > 0
        assert all(0.0 <= v <= 1.0 for v in importances.values())

    def test_predict_batch(self):
        """Test batch prediction."""
        features, _ = RemainingUsefulLifePredictor.generate_degradation_dataset(
            n_machines=1, max_life_cycles=50, random_state=99
        )
        batch = features[:5]
        predictions = self.predictor.predict_batch(batch)

        assert len(predictions) == 5
        assert all(isinstance(p, RULPrediction) for p in predictions)

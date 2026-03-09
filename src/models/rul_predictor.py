"""
Remaining Useful Life (RUL) prediction module.

Uses Gradient Boosting regression with engineered degradation features
to predict the remaining operational cycles or hours before equipment
failure. Designed for edge deployment with lightweight models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger("models.rul_predictor")


@dataclass
class RULPrediction:
    """Result of a Remaining Useful Life prediction."""
    rul_cycles: float
    rul_hours: float
    confidence_lower: float
    confidence_upper: float
    health_index: float
    degradation_rate: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize prediction to dictionary."""
        return {
            "rul_cycles": round(self.rul_cycles, 1),
            "rul_hours": round(self.rul_hours, 1),
            "confidence_lower": round(self.confidence_lower, 1),
            "confidence_upper": round(self.confidence_upper, 1),
            "health_index": round(self.health_index, 4),
            "degradation_rate": round(self.degradation_rate, 6),
            "timestamp": self.timestamp.isoformat(),
        }


class RemainingUsefulLifePredictor:
    """
    Predicts remaining useful life of industrial equipment using
    Gradient Boosting regression with engineered degradation features.

    The predictor learns degradation curves from historical data and
    estimates remaining operational time before maintenance is required.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        cycles_per_hour: float = 60.0,
        confidence_level: float = 0.95,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the RUL predictor.

        Args:
            n_estimators: Number of boosting stages.
            max_depth: Maximum depth of individual trees.
            learning_rate: Shrinkage rate for boosting.
            cycles_per_hour: Conversion factor from cycles to hours.
            confidence_level: Confidence interval level (0.5 to 0.99).
            random_state: Random seed for reproducibility.
        """
        self.cycles_per_hour = cycles_per_hour
        self.confidence_level = confidence_level
        self.random_state = random_state

        self._model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            loss="squared_error",
        )
        self._scaler = StandardScaler()

        self._is_trained = False
        self._feature_count = 0
        self._training_samples = 0
        self._prediction_count = 0
        self._prediction_residual_std = 0.0
        self._cv_score: Optional[float] = None

        logger.info(
            "RULPredictor initialized: n_estimators=%d, max_depth=%d, lr=%.3f",
            n_estimators,
            max_depth,
            learning_rate,
        )

    @property
    def is_trained(self) -> bool:
        """Whether the predictor has been trained."""
        return self._is_trained

    @staticmethod
    def generate_degradation_dataset(
        n_machines: int = 50,
        max_life_cycles: int = 200,
        n_sensors: int = 5,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic degradation data for training.

        Simulates multiple machines degrading over time with sensor
        measurements that reflect the degradation process.

        Args:
            n_machines: Number of simulated machines.
            max_life_cycles: Maximum lifecycle in cycles.
            n_sensors: Number of sensor features per reading.
            random_state: Random seed.

        Returns:
            Tuple of (features, rul_labels) where features is a 2D array
            and rul_labels is the remaining useful life for each sample.
        """
        rng = np.random.default_rng(random_state)
        all_features = []
        all_rul = []

        for machine in range(n_machines):
            # Each machine has a random total life
            total_life = rng.integers(
                int(max_life_cycles * 0.4), max_life_cycles + 1
            )
            # Degradation onset varies
            onset = rng.integers(int(total_life * 0.3), int(total_life * 0.6))

            for cycle in range(total_life):
                rul = total_life - cycle - 1

                # Base sensor values
                base_features = rng.normal(0.0, 0.5, size=n_sensors)

                # Add degradation signal after onset
                if cycle > onset:
                    degradation_progress = (cycle - onset) / (total_life - onset)
                    # Non-linear degradation curve
                    degradation = degradation_progress ** 1.5
                    for s in range(n_sensors):
                        weight = 0.5 + rng.random() * 1.5
                        base_features[s] += weight * degradation
                        # Increase noise as degradation progresses
                        base_features[s] += rng.normal(0, 0.1 * degradation)

                # Engineer additional features
                engineered = RemainingUsefulLifePredictor._engineer_features_static(
                    base_features, cycle, total_life
                )
                all_features.append(engineered)
                all_rul.append(rul)

        return np.array(all_features), np.array(all_rul, dtype=np.float64)

    @staticmethod
    def _engineer_features_static(
        raw_features: np.ndarray, cycle: int, max_cycles: int
    ) -> np.ndarray:
        """Engineer additional features from raw sensor data."""
        mean_val = np.mean(raw_features)
        std_val = np.std(raw_features)
        max_val = np.max(raw_features)
        min_val = np.min(raw_features)
        rms = np.sqrt(np.mean(raw_features ** 2))
        range_val = max_val - min_val
        normalized_cycle = cycle / max(max_cycles, 1)

        return np.concatenate([
            raw_features,
            [mean_val, std_val, max_val, min_val, rms, range_val, normalized_cycle],
        ])

    def engineer_features(self, raw_features: np.ndarray, cycle: int = 0) -> np.ndarray:
        """
        Engineer additional features from raw sensor readings.

        Args:
            raw_features: Raw sensor feature vector.
            cycle: Current operational cycle number.

        Returns:
            Enhanced feature vector with engineered features.
        """
        mean_val = np.mean(raw_features)
        std_val = np.std(raw_features)
        max_val = np.max(raw_features)
        min_val = np.min(raw_features)
        rms = np.sqrt(np.mean(raw_features ** 2))
        range_val = max_val - min_val
        normalized_cycle = cycle / 200.0  # Normalize to expected max

        return np.concatenate([
            raw_features,
            [mean_val, std_val, max_val, min_val, rms, range_val, normalized_cycle],
        ])

    def train(
        self,
        features: np.ndarray,
        rul_labels: np.ndarray,
        cross_validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the RUL predictor.

        Args:
            features: 2D array of shape (n_samples, n_features).
            rul_labels: 1D array of RUL values (target).
            cross_validate: Whether to perform cross-validation.

        Returns:
            Training metrics dictionary.
        """
        start_time = time.time()

        if features.ndim == 1:
            features = features.reshape(-1, 1)

        self._feature_count = features.shape[1]
        self._training_samples = features.shape[0]

        # Scale features
        scaled = self._scaler.fit_transform(features)

        # Train model
        self._model.fit(scaled, rul_labels)
        self._is_trained = True

        # Compute training residuals for confidence intervals
        predictions = self._model.predict(scaled)
        residuals = rul_labels - predictions
        self._prediction_residual_std = float(np.std(residuals))

        # Training score
        train_score = self._model.score(scaled, rul_labels)

        # Cross-validation
        cv_scores = None
        if cross_validate and len(rul_labels) >= 10:
            n_splits = min(5, len(rul_labels) // 2)
            cv_scores = cross_val_score(
                self._model, scaled, rul_labels, cv=n_splits, scoring="r2"
            )
            self._cv_score = float(np.mean(cv_scores))

        training_time = time.time() - start_time

        # Feature importances
        importances = self._model.feature_importances_
        top_features = np.argsort(importances)[::-1][:5]

        metrics = {
            "training_samples": self._training_samples,
            "feature_count": self._feature_count,
            "training_time_s": round(training_time, 4),
            "train_r2_score": round(train_score, 4),
            "residual_std": round(self._prediction_residual_std, 4),
            "top_feature_indices": top_features.tolist(),
            "top_feature_importances": [
                round(float(importances[i]), 4) for i in top_features
            ],
        }

        if cv_scores is not None:
            metrics["cv_r2_mean"] = round(float(np.mean(cv_scores)), 4)
            metrics["cv_r2_std"] = round(float(np.std(cv_scores)), 4)

        logger.info(
            "RUL predictor trained on %d samples in %.3fs. R2=%.4f",
            self._training_samples,
            training_time,
            train_score,
        )
        return metrics

    def predict(
        self, features: np.ndarray, cycle: int = 0
    ) -> RULPrediction:
        """
        Predict remaining useful life for a single observation.

        Args:
            features: 1D feature vector (raw sensor features).
            cycle: Current operational cycle number.

        Returns:
            RULPrediction with point estimate and confidence interval.

        Raises:
            RuntimeError: If the model has not been trained.
        """
        if not self._is_trained:
            raise RuntimeError(
                "RUL predictor must be trained before prediction. Call train() first."
            )

        if features.ndim == 1:
            features = features.reshape(1, -1)

        scaled = self._scaler.transform(features)
        rul_cycles = float(self._model.predict(scaled)[0])

        # Ensure non-negative RUL
        rul_cycles = max(0.0, rul_cycles)

        # Confidence interval using residual standard deviation
        from scipy.stats import norm
        z_val = norm.ppf((1 + self.confidence_level) / 2)
        margin = z_val * self._prediction_residual_std

        confidence_lower = max(0.0, rul_cycles - margin)
        confidence_upper = rul_cycles + margin

        # Convert to hours
        rul_hours = rul_cycles / self.cycles_per_hour

        # Compute health index (0 = failed, 1 = healthy)
        # Based on predicted RUL relative to typical equipment life
        max_expected_rul = 200.0
        health_index = min(1.0, max(0.0, rul_cycles / max_expected_rul))

        # Estimate degradation rate from feature magnitudes
        feature_magnitude = float(np.mean(np.abs(features[0])))
        degradation_rate = min(1.0, feature_magnitude / 5.0)

        self._prediction_count += 1

        return RULPrediction(
            rul_cycles=rul_cycles,
            rul_hours=rul_hours,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            health_index=health_index,
            degradation_rate=degradation_rate,
        )

    def predict_batch(
        self, features_batch: np.ndarray
    ) -> List[RULPrediction]:
        """
        Predict RUL for a batch of observations.

        Args:
            features_batch: 2D array of shape (n_samples, n_features).

        Returns:
            List of RULPrediction objects.
        """
        predictions = []
        for i in range(features_batch.shape[0]):
            pred = self.predict(features_batch[i])
            predictions.append(pred)
        return predictions

    def get_feature_importances(self) -> Dict[str, float]:
        """Return feature importance scores."""
        if not self._is_trained:
            return {}
        importances = self._model.feature_importances_
        return {
            f"feature_{i}": round(float(imp), 6)
            for i, imp in enumerate(importances)
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return predictor performance statistics."""
        return {
            "is_trained": self._is_trained,
            "training_samples": self._training_samples,
            "feature_count": self._feature_count,
            "prediction_count": self._prediction_count,
            "residual_std": self._prediction_residual_std,
            "cv_score": self._cv_score,
            "cycles_per_hour": self.cycles_per_hour,
            "confidence_level": self.confidence_level,
        }

"""
Anomaly detection module for IoT sensor data.

Combines Isolation Forest with statistical threshold methods for
real-time anomaly detection on edge devices. Supports training on
historical data and incremental scoring of new observations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger("models.anomaly_detector")


@dataclass
class AnomalyResult:
    """Result of an anomaly detection evaluation."""
    anomaly_score: float
    is_anomaly: bool
    contributing_sensors: List[str]
    confidence: float
    detection_method: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "anomaly_score": round(self.anomaly_score, 6),
            "is_anomaly": self.is_anomaly,
            "contributing_sensors": self.contributing_sensors,
            "confidence": round(self.confidence, 4),
            "detection_method": self.detection_method,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


class AnomalyDetector:
    """
    Hybrid anomaly detection using Isolation Forest and statistical thresholds.

    Combines machine learning-based anomaly scoring with classical
    statistical process control for robust anomaly detection on edge devices.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        z_score_threshold: float = 3.0,
        random_state: int = 42,
        sensor_names: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the anomaly detector.

        Args:
            contamination: Expected proportion of anomalies in data.
            n_estimators: Number of trees in the Isolation Forest.
            z_score_threshold: Z-score threshold for statistical detection.
            random_state: Random seed for reproducibility.
            sensor_names: Names of sensor features for attribution.
        """
        self.contamination = contamination
        self.z_score_threshold = z_score_threshold
        self.sensor_names = sensor_names or []

        self._isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=1,
        )
        self._scaler = StandardScaler()

        self._is_trained = False
        self._training_mean: Optional[np.ndarray] = None
        self._training_std: Optional[np.ndarray] = None
        self._feature_count = 0
        self._detection_count = 0
        self._anomaly_count = 0
        self._training_samples = 0

        logger.info(
            "AnomalyDetector initialized: contamination=%.3f, "
            "n_estimators=%d, z_threshold=%.1f",
            contamination,
            n_estimators,
            z_score_threshold,
        )

    @property
    def is_trained(self) -> bool:
        """Whether the detector has been trained."""
        return self._is_trained

    def train(self, training_data: np.ndarray, sensor_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train the anomaly detector on historical normal data.

        Args:
            training_data: 2D array of shape (n_samples, n_features).
            sensor_names: Optional feature names for sensor attribution.

        Returns:
            Training metrics dictionary.
        """
        start_time = time.time()

        if training_data.ndim == 1:
            training_data = training_data.reshape(-1, 1)

        if sensor_names:
            self.sensor_names = sensor_names

        self._feature_count = training_data.shape[1]
        self._training_samples = training_data.shape[0]

        # Fit scaler and transform
        scaled_data = self._scaler.fit_transform(training_data)

        # Store training statistics for z-score detection
        self._training_mean = np.mean(training_data, axis=0)
        self._training_std = np.std(training_data, axis=0)
        # Prevent division by zero
        self._training_std = np.where(
            self._training_std < 1e-10, 1e-10, self._training_std
        )

        # Train Isolation Forest
        self._isolation_forest.fit(scaled_data)
        self._is_trained = True

        training_time = time.time() - start_time

        # Evaluate on training data for baseline metrics
        train_scores = self._isolation_forest.decision_function(scaled_data)
        train_predictions = self._isolation_forest.predict(scaled_data)
        anomaly_rate = float(np.mean(train_predictions == -1))

        metrics = {
            "training_samples": self._training_samples,
            "feature_count": self._feature_count,
            "training_time_s": round(training_time, 4),
            "anomaly_rate": round(anomaly_rate, 4),
            "mean_score": round(float(np.mean(train_scores)), 6),
            "std_score": round(float(np.std(train_scores)), 6),
        }

        logger.info(
            "AnomalyDetector trained on %d samples (%d features) in %.3fs. "
            "Training anomaly rate: %.2f%%",
            self._training_samples,
            self._feature_count,
            training_time,
            anomaly_rate * 100,
        )
        return metrics

    def _compute_z_scores(self, features: np.ndarray) -> np.ndarray:
        """Compute z-scores relative to training distribution."""
        if self._training_mean is None or self._training_std is None:
            return np.zeros_like(features)
        return np.abs((features - self._training_mean) / self._training_std)

    def _identify_contributing_sensors(
        self, z_scores: np.ndarray, threshold: float = 2.0
    ) -> List[str]:
        """Identify sensors contributing most to the anomaly."""
        contributing = []
        # Sort indices by z-score descending
        sorted_indices = np.argsort(z_scores)[::-1]

        for idx in sorted_indices:
            if z_scores[idx] >= threshold:
                if idx < len(self.sensor_names):
                    contributing.append(self.sensor_names[idx])
                else:
                    contributing.append(f"feature_{idx}")

        return contributing if contributing else ["none"]

    def detect(self, features: np.ndarray) -> AnomalyResult:
        """
        Detect anomalies in a single observation.

        Args:
            features: 1D array of sensor features for one time step.

        Returns:
            AnomalyResult with scores and attribution.

        Raises:
            RuntimeError: If the detector has not been trained.
        """
        if not self._is_trained:
            raise RuntimeError(
                "AnomalyDetector must be trained before detection. Call train() first."
            )

        if features.ndim == 1:
            features = features.reshape(1, -1)

        self._detection_count += 1

        # Isolation Forest scoring
        scaled = self._scaler.transform(features)
        if_score = float(self._isolation_forest.decision_function(scaled)[0])
        if_prediction = int(self._isolation_forest.predict(scaled)[0])

        # Statistical z-score detection
        z_scores = self._compute_z_scores(features[0])
        max_z = float(np.max(z_scores))
        mean_z = float(np.mean(z_scores))
        z_anomaly = max_z > self.z_score_threshold

        # Combine methods: anomaly if either method flags it
        is_anomaly = (if_prediction == -1) or z_anomaly

        # Compute composite anomaly score (0 = normal, 1 = highly anomalous)
        # Normalize IF score: more negative = more anomalous
        if_normalized = max(0.0, min(1.0, -if_score + 0.5))
        z_normalized = max(0.0, min(1.0, mean_z / (2.0 * self.z_score_threshold)))
        anomaly_score = 0.6 * if_normalized + 0.4 * z_normalized

        # Compute confidence based on agreement between methods
        if (if_prediction == -1) == z_anomaly:
            confidence = 0.85 + 0.15 * min(anomaly_score, 1.0)
        else:
            confidence = 0.5 + 0.2 * min(anomaly_score, 1.0)

        # Identify contributing sensors
        contributing = self._identify_contributing_sensors(z_scores)

        # Determine detection method
        if (if_prediction == -1) and z_anomaly:
            method = "isolation_forest+z_score"
        elif if_prediction == -1:
            method = "isolation_forest"
        elif z_anomaly:
            method = "z_score"
        else:
            method = "none"

        if is_anomaly:
            self._anomaly_count += 1

        result = AnomalyResult(
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            contributing_sensors=contributing,
            confidence=confidence,
            detection_method=method,
            details={
                "if_score": round(if_score, 6),
                "if_prediction": if_prediction,
                "max_z_score": round(max_z, 4),
                "mean_z_score": round(mean_z, 4),
                "z_anomaly": z_anomaly,
            },
        )

        return result

    def detect_batch(self, features_batch: np.ndarray) -> List[AnomalyResult]:
        """
        Detect anomalies in a batch of observations.

        Args:
            features_batch: 2D array of shape (n_samples, n_features).

        Returns:
            List of AnomalyResult objects.
        """
        results = []
        for i in range(features_batch.shape[0]):
            result = self.detect(features_batch[i])
            results.append(result)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Return detector performance statistics."""
        anomaly_rate = (
            self._anomaly_count / self._detection_count
            if self._detection_count > 0
            else 0.0
        )
        return {
            "is_trained": self._is_trained,
            "training_samples": self._training_samples,
            "feature_count": self._feature_count,
            "detection_count": self._detection_count,
            "anomaly_count": self._anomaly_count,
            "anomaly_rate": round(anomaly_rate, 4),
            "contamination": self.contamination,
            "z_score_threshold": self.z_score_threshold,
        }

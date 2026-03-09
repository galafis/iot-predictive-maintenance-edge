"""
Edge processor for real-time IoT predictive maintenance.

Orchestrates the complete processing pipeline on edge devices:
sensor ingestion, preprocessing, anomaly detection, RUL prediction,
alert generation, and cloud synchronization batching.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from src.alerts.alert_system import AlertSystem
from src.models.anomaly_detector import AnomalyDetector, AnomalyResult
from src.models.rul_predictor import RemainingUsefulLifePredictor, RULPrediction
from src.sensors.data_ingestion import (
    MachineState,
    SensorDataIngester,
    SensorReading,
    SensorType,
)
from src.sensors.preprocessor import SensorPreprocessor
from src.utils.logger import get_logger

logger = get_logger("edge.edge_processor")


@dataclass
class ProcessingResult:
    """Result from a single edge processing cycle."""
    machine_id: str
    timestamp: datetime
    sensor_readings: Dict[str, float]
    features: Dict[str, float]
    anomaly_result: Optional[Dict[str, Any]]
    rul_prediction: Optional[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    processing_time_ms: float
    cycle_number: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "machine_id": self.machine_id,
            "timestamp": self.timestamp.isoformat(),
            "sensor_readings": self.sensor_readings,
            "features": self.features,
            "anomaly_result": self.anomaly_result,
            "rul_prediction": self.rul_prediction,
            "alerts": self.alerts,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "cycle_number": self.cycle_number,
        }


class EdgeProcessor:
    """
    Edge computing processor for predictive maintenance.

    Manages the complete processing pipeline from sensor data ingestion
    through ML inference to alert generation. Designed for deployment
    on resource-constrained edge devices with batch result buffering.
    """

    def __init__(
        self,
        device_id: str = "edge-node-001",
        batch_size: int = 50,
        window_size: int = 20,
    ) -> None:
        """
        Initialize the edge processor.

        Args:
            device_id: Unique identifier for this edge device.
            batch_size: Number of results to buffer before sync.
            window_size: Sensor data window for feature extraction.
        """
        self.device_id = device_id
        self.batch_size = batch_size
        self.window_size = window_size

        self._preprocessor = SensorPreprocessor(window_size=5)
        self._anomaly_detector = AnomalyDetector(
            contamination=0.05,
            n_estimators=100,
            z_score_threshold=3.0,
        )
        self._rul_predictor = RemainingUsefulLifePredictor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
        )
        self._alert_system = AlertSystem()

        self._ingesters: Dict[str, SensorDataIngester] = {}
        self._result_buffer: List[ProcessingResult] = []
        self._sensor_history: Dict[str, Dict[str, List[float]]] = {}

        self._total_cycles = 0
        self._total_processing_time_ms = 0.0
        self._start_time = time.time()

        self._models_trained = False

        logger.info(
            "EdgeProcessor initialized: device=%s, batch_size=%d, window=%d",
            device_id,
            batch_size,
            window_size,
        )

    @property
    def alert_system(self) -> AlertSystem:
        """Access the alert system."""
        return self._alert_system

    def register_machine(
        self,
        machine_id: str,
        initial_state: MachineState = MachineState.NORMAL,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Register a machine for monitoring.

        Args:
            machine_id: Unique machine identifier.
            initial_state: Initial operating state.
            random_seed: Random seed for reproducible simulation.
        """
        ingester = SensorDataIngester(
            machine_id=machine_id,
            initial_state=initial_state,
            random_seed=random_seed,
        )
        self._ingesters[machine_id] = ingester
        self._sensor_history[machine_id] = {
            st.value: [] for st in SensorType
        }
        logger.info("Registered machine '%s' with state '%s'", machine_id, initial_state.value)

    def set_machine_state(self, machine_id: str, state: MachineState) -> None:
        """Update the operating state of a registered machine."""
        if machine_id in self._ingesters:
            self._ingesters[machine_id].set_state(state)

    def train_models(
        self,
        n_training_machines: int = 30,
        max_life_cycles: int = 200,
    ) -> Dict[str, Any]:
        """
        Train ML models with synthetic degradation data.

        Args:
            n_training_machines: Number of simulated machines for training.
            max_life_cycles: Maximum lifecycle for training data.

        Returns:
            Combined training metrics from both models.
        """
        logger.info("Training edge ML models with %d simulated machines...", n_training_machines)

        # Generate training data for anomaly detector
        n_sensors = 5
        sensor_names = [st.value for st in SensorType]
        feature_names = []
        for name in sensor_names:
            for feat in ["mean", "std", "rms", "peak", "crest", "kurt", "skew", "p2p", "energy"]:
                feature_names.append(f"{name}_{feat}")

        # Generate normal operation data for anomaly training
        rng = np.random.default_rng(42)
        n_normal_samples = 500
        n_features = len(feature_names)
        normal_data = rng.normal(0, 1, size=(n_normal_samples, n_features))
        # Add small correlations between features
        for i in range(1, n_features):
            normal_data[:, i] += 0.3 * normal_data[:, 0]

        anomaly_metrics = self._anomaly_detector.train(
            normal_data, sensor_names=feature_names
        )

        # Generate degradation data for RUL training
        features, rul_labels = RemainingUsefulLifePredictor.generate_degradation_dataset(
            n_machines=n_training_machines,
            max_life_cycles=max_life_cycles,
            n_sensors=n_sensors,
            random_state=42,
        )
        rul_metrics = self._rul_predictor.train(features, rul_labels)

        self._models_trained = True

        logger.info("Models trained successfully.")
        return {
            "anomaly_detector": anomaly_metrics,
            "rul_predictor": rul_metrics,
        }

    def process_cycle(self, machine_id: str) -> Optional[ProcessingResult]:
        """
        Execute one processing cycle for a machine.

        Ingests sensor data, preprocesses signals, runs anomaly detection
        and RUL prediction, and generates alerts if needed.

        Args:
            machine_id: Machine to process.

        Returns:
            ProcessingResult or None if machine is not registered.
        """
        if machine_id not in self._ingesters:
            logger.warning("Machine '%s' is not registered", machine_id)
            return None

        start = time.time()
        ingester = self._ingesters[machine_id]

        # Step 1: Ingest sensor readings
        readings = ingester.generate_readings()
        sensor_values = {}
        for r in readings:
            sensor_values[r.sensor_type.value] = r.value
            self._sensor_history[machine_id][r.sensor_type.value].append(r.value)
            # Keep only recent history
            hist = self._sensor_history[machine_id][r.sensor_type.value]
            if len(hist) > self.window_size:
                self._sensor_history[machine_id][r.sensor_type.value] = hist[-self.window_size:]

        # Step 2: Preprocess and extract features
        features_dict = {}
        sensor_data = {}
        for sensor_type, values in self._sensor_history[machine_id].items():
            if len(values) >= 5:
                signal = np.array(values)
                sensor_data[sensor_type] = signal

        anomaly_result_dict = None
        rul_prediction_dict = None
        alerts = []

        if sensor_data and self._models_trained:
            results = self._preprocessor.process_multi_sensor(sensor_data)

            for stype, res in results.items():
                feat = res["features"]
                features_dict[f"{stype}_mean"] = feat.mean
                features_dict[f"{stype}_std"] = feat.std
                features_dict[f"{stype}_rms"] = feat.rms
                features_dict[f"{stype}_peak"] = feat.peak

            # Step 3: Anomaly detection
            feature_vector = self._preprocessor.create_feature_vector(results)

            # Pad or truncate to match training feature count
            expected_features = self._anomaly_detector._feature_count
            if len(feature_vector) < expected_features:
                feature_vector = np.pad(
                    feature_vector,
                    (0, expected_features - len(feature_vector)),
                )
            elif len(feature_vector) > expected_features:
                feature_vector = feature_vector[:expected_features]

            anomaly = self._anomaly_detector.detect(feature_vector)
            anomaly_result_dict = anomaly.to_dict()

            # Step 4: RUL prediction
            rul_features = self._rul_predictor.engineer_features(
                feature_vector[:5] if len(feature_vector) >= 5 else feature_vector,
                cycle=ingester.cycle_count,
            )
            # Pad/truncate to match RUL model's expected features
            expected_rul = self._rul_predictor._feature_count
            if len(rul_features) < expected_rul:
                rul_features = np.pad(rul_features, (0, expected_rul - len(rul_features)))
            elif len(rul_features) > expected_rul:
                rul_features = rul_features[:expected_rul]

            rul_pred = self._rul_predictor.predict(rul_features, cycle=ingester.cycle_count)
            rul_prediction_dict = rul_pred.to_dict()

            # Step 5: Generate alerts
            new_alerts = self._alert_system.evaluate(
                machine_id=machine_id,
                anomaly_result=anomaly,
                rul_prediction=rul_pred,
                sensor_readings=sensor_values,
            )
            alerts = [a.to_dict() for a in new_alerts]

        processing_time = (time.time() - start) * 1000
        self._total_cycles += 1
        self._total_processing_time_ms += processing_time

        result = ProcessingResult(
            machine_id=machine_id,
            timestamp=datetime.now(timezone.utc),
            sensor_readings=sensor_values,
            features=features_dict,
            anomaly_result=anomaly_result_dict,
            rul_prediction=rul_prediction_dict,
            alerts=alerts,
            processing_time_ms=processing_time,
            cycle_number=ingester.cycle_count,
        )

        self._result_buffer.append(result)
        return result

    def process_all_machines(self) -> List[ProcessingResult]:
        """Process one cycle for all registered machines."""
        results = []
        for machine_id in self._ingesters:
            result = self.process_cycle(machine_id)
            if result:
                results.append(result)
        return results

    def get_buffered_results(self, clear: bool = True) -> List[ProcessingResult]:
        """Retrieve and optionally clear the result buffer."""
        results = list(self._result_buffer)
        if clear:
            self._result_buffer.clear()
        return results

    def get_machine_status(self, machine_id: str) -> Dict[str, Any]:
        """Get current status for a specific machine."""
        if machine_id not in self._ingesters:
            return {"error": f"Machine '{machine_id}' not registered"}

        ingester = self._ingesters[machine_id]
        recent_readings = {}
        for st, values in self._sensor_history[machine_id].items():
            if values:
                recent_readings[st] = round(values[-1], 4)

        return {
            "machine_id": machine_id,
            "state": ingester.state.value,
            "cycle_count": ingester.cycle_count,
            "recent_readings": recent_readings,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return edge processor performance statistics."""
        uptime = time.time() - self._start_time
        avg_processing = (
            self._total_processing_time_ms / self._total_cycles
            if self._total_cycles > 0
            else 0.0
        )

        return {
            "device_id": self.device_id,
            "registered_machines": len(self._ingesters),
            "total_cycles": self._total_cycles,
            "buffer_size": len(self._result_buffer),
            "avg_processing_time_ms": round(avg_processing, 2),
            "uptime_seconds": round(uptime, 1),
            "models_trained": self._models_trained,
            "anomaly_detector_stats": self._anomaly_detector.get_stats(),
            "rul_predictor_stats": self._rul_predictor.get_stats(),
            "alert_stats": self._alert_system.get_stats(),
        }

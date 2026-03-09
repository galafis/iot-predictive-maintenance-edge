"""
Sensor data ingestion module for IoT predictive maintenance.

Simulates realistic IoT sensor readings from industrial equipment with
support for normal operation, degradation patterns, and failure modes.
Uses an MQTT-like message pattern for edge computing compatibility.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("sensors.data_ingestion")


class SensorType(str, Enum):
    """Supported industrial sensor types."""
    VIBRATION = "vibration"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    CURRENT = "current"
    ACOUSTIC_EMISSION = "acoustic_emission"


class MachineState(str, Enum):
    """Operating state of monitored equipment."""
    NORMAL = "normal"
    DEGRADING = "degrading"
    FAILURE = "failure"


@dataclass
class SensorReading:
    """Single sensor measurement with metadata."""
    sensor_id: str
    machine_id: str
    sensor_type: SensorType
    value: float
    unit: str
    timestamp: datetime
    quality: float = 1.0
    reading_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize reading to dictionary for message passing."""
        return {
            "reading_id": self.reading_id,
            "sensor_id": self.sensor_id,
            "machine_id": self.machine_id,
            "sensor_type": self.sensor_type.value,
            "value": round(self.value, 6),
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "quality": round(self.quality, 4),
        }


@dataclass
class SensorConfig:
    """Configuration for a sensor's data generation profile."""
    sensor_type: SensorType
    unit: str
    normal_mean: float
    normal_std: float
    degradation_drift_rate: float
    degradation_noise_multiplier: float
    failure_mean: float
    failure_std: float
    min_value: float
    max_value: float


# Realistic sensor profiles for industrial equipment
DEFAULT_SENSOR_CONFIGS: Dict[SensorType, SensorConfig] = {
    SensorType.VIBRATION: SensorConfig(
        sensor_type=SensorType.VIBRATION,
        unit="mm/s",
        normal_mean=2.5,
        normal_std=0.3,
        degradation_drift_rate=0.02,
        degradation_noise_multiplier=2.5,
        failure_mean=12.0,
        failure_std=3.0,
        min_value=0.0,
        max_value=25.0,
    ),
    SensorType.TEMPERATURE: SensorConfig(
        sensor_type=SensorType.TEMPERATURE,
        unit="celsius",
        normal_mean=65.0,
        normal_std=3.0,
        degradation_drift_rate=0.15,
        degradation_noise_multiplier=1.8,
        failure_mean=120.0,
        failure_std=15.0,
        min_value=-20.0,
        max_value=200.0,
    ),
    SensorType.PRESSURE: SensorConfig(
        sensor_type=SensorType.PRESSURE,
        unit="bar",
        normal_mean=4.5,
        normal_std=0.2,
        degradation_drift_rate=0.01,
        degradation_noise_multiplier=2.0,
        failure_mean=8.0,
        failure_std=1.5,
        min_value=0.0,
        max_value=15.0,
    ),
    SensorType.CURRENT: SensorConfig(
        sensor_type=SensorType.CURRENT,
        unit="ampere",
        normal_mean=15.0,
        normal_std=1.0,
        degradation_drift_rate=0.08,
        degradation_noise_multiplier=2.2,
        failure_mean=35.0,
        failure_std=5.0,
        min_value=0.0,
        max_value=50.0,
    ),
    SensorType.ACOUSTIC_EMISSION: SensorConfig(
        sensor_type=SensorType.ACOUSTIC_EMISSION,
        unit="dB",
        normal_mean=45.0,
        normal_std=5.0,
        degradation_drift_rate=0.3,
        degradation_noise_multiplier=2.0,
        failure_mean=95.0,
        failure_std=10.0,
        min_value=0.0,
        max_value=140.0,
    ),
}


class SensorDataIngester:
    """
    Simulates IoT sensor data ingestion from industrial equipment.

    Generates realistic sensor readings with configurable operating modes
    including normal operation, degradation patterns, and failure scenarios.
    Follows an MQTT-like publish pattern for edge deployment compatibility.
    """

    def __init__(
        self,
        machine_id: str,
        sensor_configs: Optional[Dict[SensorType, SensorConfig]] = None,
        initial_state: MachineState = MachineState.NORMAL,
        degradation_start_cycle: int = 0,
        random_seed: Optional[int] = None,
    ) -> None:
        self.machine_id = machine_id
        self.sensor_configs = sensor_configs or DEFAULT_SENSOR_CONFIGS
        self.state = initial_state
        self.degradation_start_cycle = degradation_start_cycle
        self.cycle_count = 0
        self.rng = np.random.default_rng(random_seed)

        self._sensor_ids: Dict[SensorType, str] = {}
        for sensor_type in self.sensor_configs:
            sid = f"{machine_id}_{sensor_type.value}_{str(uuid.uuid4())[:6]}"
            self._sensor_ids[sensor_type] = sid

        self._message_buffer: List[Dict[str, Any]] = []
        self._total_readings = 0

        logger.info(
            "SensorDataIngester initialized for machine '%s' with %d sensors, "
            "initial state: %s",
            machine_id,
            len(self.sensor_configs),
            initial_state.value,
        )

    @property
    def sensor_ids(self) -> Dict[SensorType, str]:
        """Map of sensor type to unique sensor ID."""
        return dict(self._sensor_ids)

    def set_state(self, state: MachineState) -> None:
        """Update the machine operating state."""
        previous = self.state
        self.state = state
        if state == MachineState.DEGRADING:
            self.degradation_start_cycle = self.cycle_count
        logger.info(
            "Machine '%s' state changed: %s -> %s at cycle %d",
            self.machine_id,
            previous.value,
            state.value,
            self.cycle_count,
        )

    def _generate_value(self, config: SensorConfig) -> float:
        """Generate a sensor value based on the current machine state."""
        if self.state == MachineState.NORMAL:
            value = self.rng.normal(config.normal_mean, config.normal_std)

        elif self.state == MachineState.DEGRADING:
            cycles_degrading = max(1, self.cycle_count - self.degradation_start_cycle)
            drift = config.degradation_drift_rate * cycles_degrading
            noise_std = config.normal_std * config.degradation_noise_multiplier
            base = config.normal_mean + drift
            value = self.rng.normal(base, noise_std)

            # Add occasional spikes during degradation
            if self.rng.random() < 0.05:
                spike = self.rng.uniform(1.5, 3.0) * config.normal_std
                value += spike

        else:  # FAILURE
            value = self.rng.normal(config.failure_mean, config.failure_std)
            # Chaotic behavior in failure mode
            if self.rng.random() < 0.15:
                value *= self.rng.uniform(0.5, 2.0)

        return float(np.clip(value, config.min_value, config.max_value))

    def _compute_quality(self) -> float:
        """Compute signal quality score based on machine state."""
        if self.state == MachineState.NORMAL:
            return float(np.clip(self.rng.normal(0.98, 0.01), 0.9, 1.0))
        elif self.state == MachineState.DEGRADING:
            cycles = max(1, self.cycle_count - self.degradation_start_cycle)
            decay = min(0.3, cycles * 0.001)
            return float(np.clip(self.rng.normal(0.9 - decay, 0.03), 0.5, 1.0))
        else:
            return float(np.clip(self.rng.normal(0.6, 0.1), 0.2, 0.9))

    def generate_readings(
        self,
        sensor_types: Optional[List[SensorType]] = None,
        timestamp: Optional[datetime] = None,
    ) -> List[SensorReading]:
        """
        Generate one set of sensor readings for all (or specified) sensors.

        Args:
            sensor_types: Subset of sensors to read. None means all.
            timestamp: Override timestamp. Defaults to current UTC time.

        Returns:
            List of SensorReading objects with generated values.
        """
        ts = timestamp or datetime.now(timezone.utc)
        types_to_read = sensor_types or list(self.sensor_configs.keys())
        readings = []

        for sensor_type in types_to_read:
            if sensor_type not in self.sensor_configs:
                continue
            config = self.sensor_configs[sensor_type]
            value = self._generate_value(config)
            quality = self._compute_quality()

            reading = SensorReading(
                sensor_id=self._sensor_ids[sensor_type],
                machine_id=self.machine_id,
                sensor_type=sensor_type,
                value=value,
                unit=config.unit,
                timestamp=ts,
                quality=quality,
            )
            readings.append(reading)
            self._total_readings += 1

        self.cycle_count += 1
        return readings

    def generate_batch(
        self,
        num_readings: int,
        interval_seconds: float = 1.0,
        sensor_types: Optional[List[SensorType]] = None,
    ) -> List[List[SensorReading]]:
        """
        Generate a batch of time-series sensor readings.

        Args:
            num_readings: Number of reading cycles to generate.
            interval_seconds: Time interval between cycles.
            sensor_types: Subset of sensors. None means all.

        Returns:
            List of reading cycles, each containing readings for all sensors.
        """
        base_time = datetime.now(timezone.utc)
        batch = []

        for i in range(num_readings):
            ts = datetime.fromtimestamp(
                base_time.timestamp() + i * interval_seconds,
                tz=timezone.utc,
            )
            readings = self.generate_readings(sensor_types=sensor_types, timestamp=ts)
            batch.append(readings)

        logger.debug(
            "Generated batch of %d cycles (%d total readings) for machine '%s'",
            num_readings,
            sum(len(r) for r in batch),
            self.machine_id,
        )
        return batch

    def publish_reading(self, reading: SensorReading) -> Dict[str, Any]:
        """
        Simulate MQTT publish of a sensor reading.

        Returns the message payload that would be published to the
        MQTT topic: sensors/{machine_id}/{sensor_type}/telemetry
        """
        topic = f"sensors/{reading.machine_id}/{reading.sensor_type.value}/telemetry"
        payload = reading.to_dict()

        message = {
            "topic": topic,
            "payload": payload,
            "qos": 1,
            "timestamp": time.time(),
        }

        self._message_buffer.append(message)
        return message

    def get_buffered_messages(self, clear: bool = True) -> List[Dict[str, Any]]:
        """Retrieve and optionally clear the MQTT message buffer."""
        messages = list(self._message_buffer)
        if clear:
            self._message_buffer.clear()
        return messages

    def get_stats(self) -> Dict[str, Any]:
        """Return ingester statistics."""
        return {
            "machine_id": self.machine_id,
            "state": self.state.value,
            "cycle_count": self.cycle_count,
            "total_readings": self._total_readings,
            "sensor_count": len(self.sensor_configs),
            "buffer_size": len(self._message_buffer),
            "degradation_start_cycle": self.degradation_start_cycle,
        }

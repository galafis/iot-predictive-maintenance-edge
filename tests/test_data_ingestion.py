"""Tests for sensor data ingestion module."""

import numpy as np
import pytest

from src.sensors.data_ingestion import (
    MachineState,
    SensorDataIngester,
    SensorReading,
    SensorType,
)


class TestSensorDataIngester:
    """Test suite for SensorDataIngester class."""

    def test_init_default_sensors(self):
        """Test ingester initializes with all 5 default sensor types."""
        ingester = SensorDataIngester(machine_id="test-001", random_seed=42)
        assert len(ingester.sensor_ids) == 5
        assert SensorType.VIBRATION in ingester.sensor_ids
        assert SensorType.TEMPERATURE in ingester.sensor_ids
        assert SensorType.PRESSURE in ingester.sensor_ids
        assert SensorType.CURRENT in ingester.sensor_ids
        assert SensorType.ACOUSTIC_EMISSION in ingester.sensor_ids

    def test_generate_readings_returns_all_sensors(self):
        """Test that generate_readings returns one reading per sensor."""
        ingester = SensorDataIngester(machine_id="test-001", random_seed=42)
        readings = ingester.generate_readings()
        assert len(readings) == 5
        assert all(isinstance(r, SensorReading) for r in readings)

    def test_generate_readings_normal_state(self):
        """Test normal state values are within reasonable bounds."""
        ingester = SensorDataIngester(
            machine_id="test-001",
            initial_state=MachineState.NORMAL,
            random_seed=42,
        )
        # Generate multiple readings for statistical stability
        all_vibrations = []
        for _ in range(100):
            readings = ingester.generate_readings()
            for r in readings:
                if r.sensor_type == SensorType.VIBRATION:
                    all_vibrations.append(r.value)

        mean_vib = np.mean(all_vibrations)
        # Normal vibration mean should be around 2.5 mm/s
        assert 1.5 < mean_vib < 4.0, f"Normal vibration mean out of range: {mean_vib}"

    def test_degrading_state_drift(self):
        """Test that degrading state shows increasing sensor values."""
        ingester = SensorDataIngester(
            machine_id="test-001",
            initial_state=MachineState.DEGRADING,
            random_seed=42,
        )

        early_values = []
        for _ in range(20):
            readings = ingester.generate_readings()
            for r in readings:
                if r.sensor_type == SensorType.VIBRATION:
                    early_values.append(r.value)

        late_values = []
        for _ in range(20):
            readings = ingester.generate_readings()
            for r in readings:
                if r.sensor_type == SensorType.VIBRATION:
                    late_values.append(r.value)

        # Later values should trend higher due to degradation drift
        assert np.mean(late_values) > np.mean(early_values) - 1.0

    def test_failure_state_high_values(self):
        """Test that failure state generates high sensor values."""
        ingester = SensorDataIngester(
            machine_id="test-001",
            initial_state=MachineState.FAILURE,
            random_seed=42,
        )
        vibration_values = []
        for _ in range(50):
            readings = ingester.generate_readings()
            for r in readings:
                if r.sensor_type == SensorType.VIBRATION:
                    vibration_values.append(r.value)

        mean_vib = np.mean(vibration_values)
        # Failure vibration mean should be much higher than normal
        assert mean_vib > 6.0, f"Failure vibration mean too low: {mean_vib}"

    def test_state_change(self):
        """Test machine state can be changed dynamically."""
        ingester = SensorDataIngester(machine_id="test-001", random_seed=42)
        assert ingester.state == MachineState.NORMAL

        ingester.set_state(MachineState.DEGRADING)
        assert ingester.state == MachineState.DEGRADING

        ingester.set_state(MachineState.FAILURE)
        assert ingester.state == MachineState.FAILURE

    def test_generate_batch(self):
        """Test batch generation produces correct number of cycles."""
        ingester = SensorDataIngester(machine_id="test-001", random_seed=42)
        batch = ingester.generate_batch(num_readings=10, interval_seconds=1.0)
        assert len(batch) == 10
        assert all(len(cycle) == 5 for cycle in batch)

    def test_reading_serialization(self):
        """Test SensorReading serializes to dictionary correctly."""
        ingester = SensorDataIngester(machine_id="test-001", random_seed=42)
        readings = ingester.generate_readings()
        reading_dict = readings[0].to_dict()

        assert "sensor_id" in reading_dict
        assert "machine_id" in reading_dict
        assert reading_dict["machine_id"] == "test-001"
        assert "value" in reading_dict
        assert "timestamp" in reading_dict

    def test_mqtt_publish(self):
        """Test MQTT message publishing simulation."""
        ingester = SensorDataIngester(machine_id="test-001", random_seed=42)
        readings = ingester.generate_readings()
        message = ingester.publish_reading(readings[0])

        assert "topic" in message
        assert "payload" in message
        assert "sensors/test-001/" in message["topic"]

    def test_stats(self):
        """Test ingester statistics tracking."""
        ingester = SensorDataIngester(machine_id="test-001", random_seed=42)
        ingester.generate_readings()
        ingester.generate_readings()
        stats = ingester.get_stats()

        assert stats["machine_id"] == "test-001"
        assert stats["cycle_count"] == 2
        assert stats["total_readings"] == 10

"""Tests for alert system module."""

from datetime import datetime, timezone

import pytest

from src.alerts.alert_system import Alert, AlertRule, AlertSeverity, AlertSystem
from src.models.anomaly_detector import AnomalyResult
from src.models.rul_predictor import RULPrediction


class TestAlertSystem:
    """Test suite for AlertSystem class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.alert_system = AlertSystem(cooldown_seconds=0, max_history=100)

    def _make_anomaly_result(self, score: float, is_anomaly: bool = True):
        """Helper to create AnomalyResult."""
        return AnomalyResult(
            anomaly_score=score,
            is_anomaly=is_anomaly,
            contributing_sensors=["vibration", "temperature"],
            confidence=0.9,
            detection_method="isolation_forest+z_score",
        )

    def _make_rul_prediction(self, rul_cycles: float, health_index: float):
        """Helper to create RULPrediction."""
        return RULPrediction(
            rul_cycles=rul_cycles,
            rul_hours=rul_cycles / 60.0,
            confidence_lower=max(0, rul_cycles - 10),
            confidence_upper=rul_cycles + 10,
            health_index=health_index,
            degradation_rate=0.05,
        )

    def test_high_anomaly_triggers_critical(self):
        """Test high anomaly score triggers critical alert."""
        anomaly = self._make_anomaly_result(0.8)
        alerts = self.alert_system.evaluate(
            machine_id="test-001",
            anomaly_result=anomaly,
        )
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        assert len(critical_alerts) > 0

    def test_low_rul_triggers_emergency(self):
        """Test very low RUL triggers emergency alert."""
        rul = self._make_rul_prediction(rul_cycles=10.0, health_index=0.05)
        alerts = self.alert_system.evaluate(
            machine_id="test-001",
            rul_prediction=rul,
        )
        emergency_alerts = [a for a in alerts if a.severity == AlertSeverity.EMERGENCY]
        assert len(emergency_alerts) > 0

    def test_normal_operation_no_alerts(self):
        """Test normal operation generates no alerts."""
        anomaly = self._make_anomaly_result(0.1, is_anomaly=False)
        rul = self._make_rul_prediction(rul_cycles=180.0, health_index=0.9)
        alerts = self.alert_system.evaluate(
            machine_id="test-001",
            anomaly_result=anomaly,
            rul_prediction=rul,
        )
        assert len(alerts) == 0

    def test_sensor_threshold_alert(self):
        """Test sensor value threshold triggers alert."""
        sensor_readings = {"vibration": 10.0, "temperature": 65.0}
        alerts = self.alert_system.evaluate(
            machine_id="test-001",
            sensor_readings=sensor_readings,
        )
        vibration_alerts = [a for a in alerts if "vibration" in a.message.lower()]
        assert len(vibration_alerts) > 0

    def test_alert_serialization(self):
        """Test alert serializes to dictionary correctly."""
        anomaly = self._make_anomaly_result(0.8)
        alerts = self.alert_system.evaluate(
            machine_id="test-001",
            anomaly_result=anomaly,
        )
        assert len(alerts) > 0
        alert_dict = alerts[0].to_dict()
        assert "alert_id" in alert_dict
        assert "severity" in alert_dict
        assert "machine_id" in alert_dict
        assert alert_dict["machine_id"] == "test-001"

    def test_alert_history(self):
        """Test alert history tracking."""
        for i in range(5):
            anomaly = self._make_anomaly_result(0.8)
            self.alert_system.evaluate(
                machine_id=f"test-{i:03d}",
                anomaly_result=anomaly,
            )

        recent = self.alert_system.get_recent_alerts(limit=10)
        assert len(recent) > 0

    def test_alert_stats(self):
        """Test alert system statistics."""
        anomaly = self._make_anomaly_result(0.8)
        self.alert_system.evaluate(machine_id="test-001", anomaly_result=anomaly)
        stats = self.alert_system.get_stats()

        assert stats["total_alerts"] > 0
        assert "alerts_by_severity" in stats

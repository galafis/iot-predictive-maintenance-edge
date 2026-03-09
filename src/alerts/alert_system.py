"""
Alert system for IoT predictive maintenance.

Provides configurable alert generation based on anomaly detection scores,
RUL predictions, and sensor threshold violations. Supports severity levels,
deduplication, cooldown periods, and alert history tracking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, List, Optional

from src.models.anomaly_detector import AnomalyResult
from src.models.rul_predictor import RULPrediction
from src.utils.logger import get_logger

logger = get_logger("alerts.alert_system")


class AlertSeverity(IntEnum):
    """Alert severity levels ordered by priority."""
    INFO = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3


SEVERITY_LABELS = {
    AlertSeverity.INFO: "INFO",
    AlertSeverity.WARNING: "WARNING",
    AlertSeverity.CRITICAL: "CRITICAL",
    AlertSeverity.EMERGENCY: "EMERGENCY",
}

SEVERITY_SYMBOLS = {
    AlertSeverity.INFO: "[i]",
    AlertSeverity.WARNING: "[!]",
    AlertSeverity.CRITICAL: "[!!]",
    AlertSeverity.EMERGENCY: "[!!!]",
}


@dataclass
class Alert:
    """Individual alert record."""
    alert_id: str
    machine_id: str
    severity: AlertSeverity
    title: str
    message: str
    source: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "alert_id": self.alert_id,
            "machine_id": self.machine_id,
            "severity": SEVERITY_LABELS[self.severity],
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "details": self.details,
        }


@dataclass
class AlertRule:
    """Configurable alert rule definition."""
    name: str
    condition_type: str  # 'anomaly_score', 'rul_threshold', 'sensor_threshold'
    threshold: float
    severity: AlertSeverity
    cooldown_seconds: int = 300
    enabled: bool = True


class AlertSystem:
    """
    Configurable alert system for predictive maintenance.

    Evaluates anomaly detection results, RUL predictions, and raw sensor
    readings against configurable rules to generate prioritized alerts
    with deduplication and cooldown management.
    """

    def __init__(
        self,
        cooldown_seconds: int = 60,
        max_history: int = 1000,
    ) -> None:
        """
        Initialize the alert system.

        Args:
            cooldown_seconds: Default cooldown between duplicate alerts.
            max_history: Maximum number of alerts to retain in history.
        """
        self.cooldown_seconds = cooldown_seconds
        self.max_history = max_history

        self._rules = self._default_rules()
        self._alert_history: List[Alert] = []
        self._last_alert_time: Dict[str, float] = {}
        self._alert_counter = 0
        self._total_alerts_generated = 0
        self._alerts_by_severity: Dict[AlertSeverity, int] = {
            s: 0 for s in AlertSeverity
        }

        logger.info(
            "AlertSystem initialized: cooldown=%ds, max_history=%d, rules=%d",
            cooldown_seconds,
            max_history,
            len(self._rules),
        )

    def _default_rules(self) -> List[AlertRule]:
        """Create default alert rules."""
        return [
            # Anomaly score rules
            AlertRule(
                name="anomaly_high",
                condition_type="anomaly_score",
                threshold=0.7,
                severity=AlertSeverity.CRITICAL,
                cooldown_seconds=120,
            ),
            AlertRule(
                name="anomaly_medium",
                condition_type="anomaly_score",
                threshold=0.4,
                severity=AlertSeverity.WARNING,
                cooldown_seconds=300,
            ),
            # RUL rules
            AlertRule(
                name="rul_critical",
                condition_type="rul_cycles",
                threshold=20.0,
                severity=AlertSeverity.EMERGENCY,
                cooldown_seconds=60,
            ),
            AlertRule(
                name="rul_warning",
                condition_type="rul_cycles",
                threshold=50.0,
                severity=AlertSeverity.CRITICAL,
                cooldown_seconds=180,
            ),
            AlertRule(
                name="rul_notice",
                condition_type="rul_cycles",
                threshold=100.0,
                severity=AlertSeverity.WARNING,
                cooldown_seconds=600,
            ),
            # Health index rules
            AlertRule(
                name="health_critical",
                condition_type="health_index",
                threshold=0.3,
                severity=AlertSeverity.EMERGENCY,
                cooldown_seconds=60,
            ),
            AlertRule(
                name="health_warning",
                condition_type="health_index",
                threshold=0.5,
                severity=AlertSeverity.WARNING,
                cooldown_seconds=300,
            ),
            # Sensor threshold rules
            AlertRule(
                name="vibration_high",
                condition_type="sensor_vibration",
                threshold=8.0,
                severity=AlertSeverity.CRITICAL,
                cooldown_seconds=120,
            ),
            AlertRule(
                name="temperature_high",
                condition_type="sensor_temperature",
                threshold=100.0,
                severity=AlertSeverity.CRITICAL,
                cooldown_seconds=120,
            ),
        ]

    def add_rule(self, rule: AlertRule) -> None:
        """Add a custom alert rule."""
        self._rules.append(rule)
        logger.info("Added alert rule: %s (%s)", rule.name, rule.condition_type)

    def _generate_alert_id(self) -> str:
        """Generate a unique alert ID."""
        self._alert_counter += 1
        return f"ALT-{self._alert_counter:06d}"

    def _check_cooldown(self, key: str, cooldown: int) -> bool:
        """Check if an alert is within cooldown period."""
        now = time.time()
        last = self._last_alert_time.get(key, 0)
        return (now - last) < cooldown

    def _record_alert(self, key: str, alert: Alert) -> None:
        """Record an alert and update tracking."""
        self._last_alert_time[key] = time.time()
        self._alert_history.append(alert)
        self._total_alerts_generated += 1
        self._alerts_by_severity[alert.severity] += 1

        # Prune history
        if len(self._alert_history) > self.max_history:
            self._alert_history = self._alert_history[-self.max_history:]

    def evaluate(
        self,
        machine_id: str,
        anomaly_result: Optional[AnomalyResult] = None,
        rul_prediction: Optional[RULPrediction] = None,
        sensor_readings: Optional[Dict[str, float]] = None,
    ) -> List[Alert]:
        """
        Evaluate alert rules against current machine state.

        Args:
            machine_id: Machine identifier.
            anomaly_result: Latest anomaly detection result.
            rul_prediction: Latest RUL prediction.
            sensor_readings: Current sensor values by type.

        Returns:
            List of generated Alert objects (may be empty).
        """
        alerts = []

        for rule in self._rules:
            if not rule.enabled:
                continue

            triggered = False
            value = 0.0
            title = ""
            message = ""
            source = rule.condition_type
            details: Dict[str, Any] = {}

            # Check anomaly score rules
            if rule.condition_type == "anomaly_score" and anomaly_result:
                value = anomaly_result.anomaly_score
                if value >= rule.threshold:
                    triggered = True
                    title = f"Anomaly Detected - {machine_id}"
                    message = (
                        f"Anomaly score {value:.3f} exceeds threshold "
                        f"{rule.threshold:.3f}. Contributing sensors: "
                        f"{', '.join(anomaly_result.contributing_sensors)}"
                    )
                    details = {
                        "anomaly_score": value,
                        "threshold": rule.threshold,
                        "contributing_sensors": anomaly_result.contributing_sensors,
                        "detection_method": anomaly_result.detection_method,
                    }

            # Check RUL rules
            elif rule.condition_type == "rul_cycles" and rul_prediction:
                value = rul_prediction.rul_cycles
                if value <= rule.threshold:
                    triggered = True
                    title = f"Low RUL Warning - {machine_id}"
                    message = (
                        f"Remaining useful life is {value:.0f} cycles "
                        f"({rul_prediction.rul_hours:.1f} hours). "
                        f"Threshold: {rule.threshold:.0f} cycles."
                    )
                    details = {
                        "rul_cycles": value,
                        "rul_hours": rul_prediction.rul_hours,
                        "threshold": rule.threshold,
                        "health_index": rul_prediction.health_index,
                    }

            # Check health index rules
            elif rule.condition_type == "health_index" and rul_prediction:
                value = rul_prediction.health_index
                if value <= rule.threshold:
                    triggered = True
                    title = f"Low Health Index - {machine_id}"
                    message = (
                        f"Equipment health index is {value:.3f} "
                        f"(threshold: {rule.threshold:.3f}). "
                        f"Degradation rate: {rul_prediction.degradation_rate:.4f}"
                    )
                    details = {
                        "health_index": value,
                        "threshold": rule.threshold,
                        "degradation_rate": rul_prediction.degradation_rate,
                    }

            # Check sensor threshold rules
            elif rule.condition_type.startswith("sensor_") and sensor_readings:
                sensor_name = rule.condition_type.replace("sensor_", "")
                if sensor_name in sensor_readings:
                    value = sensor_readings[sensor_name]
                    if value >= rule.threshold:
                        triggered = True
                        title = f"Sensor Threshold Exceeded - {machine_id}"
                        message = (
                            f"Sensor '{sensor_name}' reading is {value:.2f}, "
                            f"exceeding threshold of {rule.threshold:.2f}"
                        )
                        details = {
                            "sensor": sensor_name,
                            "value": value,
                            "threshold": rule.threshold,
                        }

            if triggered:
                cooldown_key = f"{machine_id}:{rule.name}"
                if not self._check_cooldown(cooldown_key, rule.cooldown_seconds):
                    alert = Alert(
                        alert_id=self._generate_alert_id(),
                        machine_id=machine_id,
                        severity=rule.severity,
                        title=title,
                        message=message,
                        source=source,
                        details=details,
                    )
                    self._record_alert(cooldown_key, alert)
                    alerts.append(alert)

                    if rule.severity >= AlertSeverity.CRITICAL:
                        logger.warning(
                            "%s %s: %s",
                            SEVERITY_SYMBOLS[rule.severity],
                            alert.title,
                            alert.message,
                        )

        return alerts

    def get_recent_alerts(
        self,
        machine_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        limit: int = 50,
    ) -> List[Alert]:
        """
        Get recent alerts with optional filtering.

        Args:
            machine_id: Filter by machine.
            severity: Filter by minimum severity.
            limit: Maximum alerts to return.

        Returns:
            List of Alert objects, newest first.
        """
        filtered = self._alert_history
        if machine_id:
            filtered = [a for a in filtered if a.machine_id == machine_id]
        if severity is not None:
            filtered = [a for a in filtered if a.severity >= severity]
        return list(reversed(filtered[-limit:]))

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        for alert in self._alert_history:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Return alert system statistics."""
        return {
            "total_alerts": self._total_alerts_generated,
            "history_size": len(self._alert_history),
            "active_rules": sum(1 for r in self._rules if r.enabled),
            "alerts_by_severity": {
                SEVERITY_LABELS[s]: count
                for s, count in self._alerts_by_severity.items()
            },
            "unacknowledged": sum(
                1 for a in self._alert_history if not a.acknowledged
            ),
        }

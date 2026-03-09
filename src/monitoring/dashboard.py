"""
Monitoring dashboard for IoT predictive maintenance system.

Provides real-time metrics collection and text-based status reporting
for edge device health, model performance, sensor status, and alert
activity. Designed for headless edge environments.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger("monitoring.dashboard")


class DashboardMetrics:
    """
    Collects and reports system metrics for the edge maintenance platform.

    Tracks sensor health, model accuracy, alert counts, edge processing
    latency, and system uptime with text-based reporting.
    """

    def __init__(self, device_id: str = "edge-node-001") -> None:
        """
        Initialize dashboard metrics collection.

        Args:
            device_id: Edge device identifier.
        """
        self.device_id = device_id
        self._start_time = time.time()

        # Sensor health tracking
        self._sensor_health: Dict[str, Dict[str, Any]] = {}

        # Model performance tracking
        self._model_metrics: Dict[str, Dict[str, Any]] = {}

        # Processing latency tracking
        self._latency_samples: List[float] = []
        self._max_latency_samples = 1000

        # Alert tracking
        self._alert_counts: Dict[str, int] = {
            "INFO": 0,
            "WARNING": 0,
            "CRITICAL": 0,
            "EMERGENCY": 0,
        }

        # Machine status tracking
        self._machine_status: Dict[str, Dict[str, Any]] = {}

        # Sync tracking
        self._sync_metrics: Dict[str, Any] = {
            "total_syncs": 0,
            "last_sync": None,
            "bytes_uploaded": 0,
        }

        logger.info("DashboardMetrics initialized for device '%s'", device_id)

    @property
    def uptime_seconds(self) -> float:
        """System uptime in seconds."""
        return time.time() - self._start_time

    @property
    def uptime_formatted(self) -> str:
        """Human-readable uptime string."""
        seconds = int(self.uptime_seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}h {minutes:02d}m {secs:02d}s"

    def update_sensor_health(
        self,
        machine_id: str,
        sensor_type: str,
        value: float,
        quality: float = 1.0,
    ) -> None:
        """Update sensor health metrics."""
        key = f"{machine_id}:{sensor_type}"
        self._sensor_health[key] = {
            "machine_id": machine_id,
            "sensor_type": sensor_type,
            "last_value": round(value, 4),
            "quality": round(quality, 4),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "status": "healthy" if quality > 0.7 else ("degraded" if quality > 0.4 else "poor"),
        }

    def update_model_metrics(
        self,
        model_name: str,
        metrics: Dict[str, float],
    ) -> None:
        """Update model performance metrics."""
        self._model_metrics[model_name] = {
            **metrics,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def record_latency(self, latency_ms: float) -> None:
        """Record a processing latency sample."""
        self._latency_samples.append(latency_ms)
        if len(self._latency_samples) > self._max_latency_samples:
            self._latency_samples = self._latency_samples[-self._max_latency_samples:]

    def record_alert(self, severity: str) -> None:
        """Record an alert occurrence by severity."""
        if severity in self._alert_counts:
            self._alert_counts[severity] += 1

    def update_machine_status(
        self,
        machine_id: str,
        state: str,
        health_index: float,
        rul_cycles: float,
        anomaly_score: float,
    ) -> None:
        """Update comprehensive machine status."""
        self._machine_status[machine_id] = {
            "state": state,
            "health_index": round(health_index, 4),
            "rul_cycles": round(rul_cycles, 1),
            "anomaly_score": round(anomaly_score, 4),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def update_sync_metrics(
        self,
        syncs: int = 0,
        bytes_uploaded: int = 0,
    ) -> None:
        """Update cloud sync metrics."""
        self._sync_metrics["total_syncs"] += syncs
        self._sync_metrics["bytes_uploaded"] += bytes_uploaded
        self._sync_metrics["last_sync"] = datetime.now(timezone.utc).isoformat()

    def get_latency_stats(self) -> Dict[str, float]:
        """Compute latency statistics."""
        if not self._latency_samples:
            return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}

        import numpy as np
        arr = np.array(self._latency_samples)
        return {
            "avg": round(float(np.mean(arr)), 2),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
            "max": round(float(np.max(arr)), 2),
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all dashboard metrics as a dictionary."""
        return {
            "device_id": self.device_id,
            "uptime": self.uptime_formatted,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "sensor_health": self._sensor_health,
            "model_metrics": self._model_metrics,
            "latency": self.get_latency_stats(),
            "alert_counts": dict(self._alert_counts),
            "machine_status": dict(self._machine_status),
            "sync": dict(self._sync_metrics),
        }

    def generate_status_report(self) -> str:
        """
        Generate a comprehensive text-based status report.

        Returns:
            Formatted status report string.
        """
        sep = "=" * 72
        thin_sep = "-" * 72
        lines = []

        lines.append(sep)
        lines.append("   IOT PREDICTIVE MAINTENANCE - EDGE DASHBOARD")
        lines.append(f"   Device: {self.device_id} | Uptime: {self.uptime_formatted}")
        lines.append(f"   Report Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(sep)

        # Machine Health Overview
        lines.append("")
        lines.append("  MACHINE HEALTH OVERVIEW")
        lines.append(thin_sep)

        if self._machine_status:
            lines.append(
                f"  {'Machine':<20} {'State':<12} {'Health':>8} "
                f"{'RUL (cyc)':>10} {'Anomaly':>9} {'Status':<10}"
            )
            lines.append(thin_sep)

            for machine_id, status in sorted(self._machine_status.items()):
                health = status["health_index"]
                rul = status["rul_cycles"]
                anomaly = status["anomaly_score"]

                if health >= 0.7:
                    health_status = "GOOD"
                elif health >= 0.4:
                    health_status = "FAIR"
                elif health >= 0.2:
                    health_status = "POOR"
                else:
                    health_status = "CRITICAL"

                # Build health bar
                bar_len = 10
                filled = int(health * bar_len)
                bar = "#" * filled + "." * (bar_len - filled)

                lines.append(
                    f"  {machine_id:<20} {status['state']:<12} "
                    f"[{bar}] "
                    f"{rul:>10.0f} {anomaly:>9.3f} {health_status:<10}"
                )
        else:
            lines.append("  No machines registered yet.")

        # Alert Summary
        lines.append("")
        lines.append("  ALERT SUMMARY")
        lines.append(thin_sep)
        total_alerts = sum(self._alert_counts.values())
        lines.append(f"  Total Alerts: {total_alerts}")
        for severity, count in self._alert_counts.items():
            indicator = "*" * min(count, 20)
            lines.append(f"    {severity:<12} : {count:>5}  {indicator}")

        # Processing Performance
        lines.append("")
        lines.append("  PROCESSING PERFORMANCE")
        lines.append(thin_sep)
        latency = self.get_latency_stats()
        lines.append(f"  Avg Latency     : {latency['avg']:>8.2f} ms")
        lines.append(f"  P50 Latency     : {latency['p50']:>8.2f} ms")
        lines.append(f"  P95 Latency     : {latency['p95']:>8.2f} ms")
        lines.append(f"  P99 Latency     : {latency['p99']:>8.2f} ms")
        lines.append(f"  Max Latency     : {latency['max']:>8.2f} ms")
        lines.append(f"  Total Samples   : {len(self._latency_samples):>8d}")

        # Model Status
        if self._model_metrics:
            lines.append("")
            lines.append("  MODEL STATUS")
            lines.append(thin_sep)
            for model_name, metrics in self._model_metrics.items():
                lines.append(f"  {model_name}:")
                for key, value in metrics.items():
                    if key != "last_updated":
                        if isinstance(value, float):
                            lines.append(f"    {key:<30} : {value:>10.4f}")
                        else:
                            lines.append(f"    {key:<30} : {str(value):>10}")

        # Cloud Sync Status
        lines.append("")
        lines.append("  CLOUD SYNC STATUS")
        lines.append(thin_sep)
        lines.append(f"  Total Syncs     : {self._sync_metrics['total_syncs']:>8d}")
        lines.append(f"  Bytes Uploaded   : {self._sync_metrics['bytes_uploaded']:>8d}")
        last_sync = self._sync_metrics.get("last_sync", "Never")
        lines.append(f"  Last Sync        : {last_sync}")

        lines.append("")
        lines.append(sep)

        return "\n".join(lines)

    def generate_machine_detail_report(self, machine_id: str) -> str:
        """
        Generate a detailed report for a specific machine.

        Args:
            machine_id: Machine identifier.

        Returns:
            Formatted machine detail report.
        """
        lines = []
        thin_sep = "-" * 50

        lines.append(f"  Machine Detail: {machine_id}")
        lines.append(thin_sep)

        if machine_id in self._machine_status:
            status = self._machine_status[machine_id]
            for key, value in status.items():
                if isinstance(value, float):
                    lines.append(f"  {key:<25} : {value:>10.4f}")
                else:
                    lines.append(f"  {key:<25} : {str(value):>10}")

        # Sensor readings
        lines.append("")
        lines.append("  Sensor Readings:")
        lines.append(thin_sep)
        for key, sensor in self._sensor_health.items():
            if sensor["machine_id"] == machine_id:
                lines.append(
                    f"  {sensor['sensor_type']:<20} : "
                    f"{sensor['last_value']:>10.4f}  "
                    f"(quality: {sensor['quality']:.2f}, "
                    f"status: {sensor['status']})"
                )

        return "\n".join(lines)

"""Tests for monitoring dashboard module."""

import pytest

from src.monitoring.dashboard import DashboardMetrics


class TestDashboardMetrics:
    """Test suite for DashboardMetrics class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.dashboard = DashboardMetrics(device_id="test-edge-001")

    def test_uptime_tracking(self):
        """Test uptime is tracked from initialization."""
        assert self.dashboard.uptime_seconds >= 0
        assert isinstance(self.dashboard.uptime_formatted, str)
        assert "h" in self.dashboard.uptime_formatted

    def test_update_machine_status(self):
        """Test machine status updates are recorded."""
        self.dashboard.update_machine_status(
            machine_id="M-001",
            state="normal",
            health_index=0.95,
            rul_cycles=150.0,
            anomaly_score=0.1,
        )
        metrics = self.dashboard.get_all_metrics()
        assert "M-001" in metrics["machine_status"]
        assert metrics["machine_status"]["M-001"]["health_index"] == 0.95

    def test_record_latency(self):
        """Test latency recording and statistics."""
        for ms in [1.0, 2.0, 3.0, 4.0, 5.0]:
            self.dashboard.record_latency(ms)

        stats = self.dashboard.get_latency_stats()
        assert stats["avg"] == 3.0
        assert stats["max"] == 5.0

    def test_record_alert(self):
        """Test alert counting."""
        self.dashboard.record_alert("WARNING")
        self.dashboard.record_alert("WARNING")
        self.dashboard.record_alert("CRITICAL")

        metrics = self.dashboard.get_all_metrics()
        assert metrics["alert_counts"]["WARNING"] == 2
        assert metrics["alert_counts"]["CRITICAL"] == 1

    def test_generate_status_report(self):
        """Test status report generation produces non-empty string."""
        self.dashboard.update_machine_status(
            "M-001", "normal", 0.9, 150.0, 0.1
        )
        self.dashboard.record_latency(2.5)
        self.dashboard.record_alert("INFO")

        report = self.dashboard.generate_status_report()
        assert len(report) > 100
        assert "MACHINE HEALTH" in report
        assert "M-001" in report

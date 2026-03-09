"""Tests for cloud sync module."""

import pytest

from src.sync.cloud_sync import CloudSyncManager, SyncStatus


class TestCloudSyncManager:
    """Test suite for CloudSyncManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sync = CloudSyncManager(
            device_id="test-edge-001",
            batch_size=10,
            max_buffer_size=100,
            sync_interval_seconds=0,
            simulate_latency_ms=1.0,
        )

    def test_initial_status_connected(self):
        """Test initial connection status is connected."""
        assert self.sync.status == SyncStatus.CONNECTED

    def test_buffer_data(self):
        """Test data buffering."""
        result = self.sync.buffer_data({"value": 42}, data_type="telemetry")
        assert result is True
        assert self.sync.buffer_size == 1

    def test_buffer_alert(self):
        """Test alert buffering."""
        self.sync.buffer_alert({"severity": "CRITICAL", "message": "test"})
        assert self.sync.buffer_size == 1

    def test_sync_upload(self):
        """Test data sync upload."""
        for i in range(5):
            self.sync.buffer_data({"reading": i})

        sync_record = self.sync.sync_upload(force=True)
        assert sync_record is not None
        assert sync_record.record_count == 5
        assert sync_record.status == "success"

    def test_sync_empty_buffer(self):
        """Test sync with empty buffer returns None."""
        result = self.sync.sync_upload(force=True)
        assert result is None

    def test_sync_disconnected(self):
        """Test sync when disconnected returns None."""
        self.sync.buffer_data({"value": 1})
        self.sync.set_connection_status(False)
        result = self.sync.sync_upload(force=True)
        assert result is None
        assert self.sync.status == SyncStatus.DISCONNECTED

    def test_stats(self):
        """Test sync statistics."""
        self.sync.buffer_data({"value": 1})
        self.sync.sync_upload(force=True)
        stats = self.sync.get_stats()

        assert stats["device_id"] == "test-edge-001"
        assert stats["total_uploaded"] == 1
        assert stats["total_syncs"] == 1

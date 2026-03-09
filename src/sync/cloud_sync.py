"""
Edge-to-cloud synchronization manager.

Handles batched data upload, model update downloads, and offline
buffering for reliable edge-to-cloud communication. Simulates
cloud connectivity for demonstration purposes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger("sync.cloud_sync")


class SyncStatus(str, Enum):
    """Synchronization status states."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    SYNCING = "syncing"
    ERROR = "error"


class SyncDirection(str, Enum):
    """Direction of data synchronization."""
    UPLOAD = "upload"
    DOWNLOAD = "download"


@dataclass
class SyncRecord:
    """Record of a synchronization operation."""
    sync_id: str
    direction: SyncDirection
    data_type: str
    record_count: int
    size_bytes: int
    timestamp: datetime
    status: str
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "sync_id": self.sync_id,
            "direction": self.direction.value,
            "data_type": self.data_type,
            "record_count": self.record_count,
            "size_bytes": self.size_bytes,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
        }


class CloudSyncManager:
    """
    Manages edge-to-cloud data synchronization.

    Handles batched upload of processed sensor data, anomaly results,
    and alerts to the cloud. Supports offline buffering when cloud
    connectivity is unavailable and model update retrieval.
    """

    def __init__(
        self,
        cloud_endpoint: str = "https://cloud.iot-maintenance.example.com/api/v1",
        device_id: str = "edge-node-001",
        batch_size: int = 100,
        max_buffer_size: int = 10000,
        sync_interval_seconds: int = 30,
        simulate_latency_ms: float = 50.0,
    ) -> None:
        """
        Initialize the cloud sync manager.

        Args:
            cloud_endpoint: Cloud API endpoint URL.
            device_id: Edge device identifier.
            batch_size: Records per sync batch.
            max_buffer_size: Maximum offline buffer size.
            sync_interval_seconds: Minimum interval between syncs.
            simulate_latency_ms: Simulated network latency for demo.
        """
        self.cloud_endpoint = cloud_endpoint
        self.device_id = device_id
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size
        self.sync_interval_seconds = sync_interval_seconds
        self.simulate_latency_ms = simulate_latency_ms

        self._status = SyncStatus.CONNECTED
        self._upload_buffer: List[Dict[str, Any]] = []
        self._alert_buffer: List[Dict[str, Any]] = []
        self._sync_history: List[SyncRecord] = []
        self._sync_counter = 0
        self._total_uploaded = 0
        self._total_downloaded = 0
        self._total_bytes_uploaded = 0
        self._last_sync_time = 0.0
        self._model_updates_available: List[Dict[str, Any]] = []
        self._connection_failures = 0

        logger.info(
            "CloudSyncManager initialized: endpoint=%s, device=%s, "
            "batch_size=%d",
            cloud_endpoint,
            device_id,
            batch_size,
        )

    @property
    def status(self) -> SyncStatus:
        """Current sync status."""
        return self._status

    @property
    def buffer_size(self) -> int:
        """Current upload buffer size."""
        return len(self._upload_buffer) + len(self._alert_buffer)

    def set_connection_status(self, connected: bool) -> None:
        """
        Set the cloud connection status.

        Args:
            connected: True if connected, False if disconnected.
        """
        if connected:
            self._status = SyncStatus.CONNECTED
            self._connection_failures = 0
            logger.info("Cloud connection established")
        else:
            self._status = SyncStatus.DISCONNECTED
            self._connection_failures += 1
            logger.warning(
                "Cloud connection lost (failures: %d)",
                self._connection_failures,
            )

    def buffer_data(self, data: Dict[str, Any], data_type: str = "telemetry") -> bool:
        """
        Buffer data for cloud upload.

        Args:
            data: Data record to buffer.
            data_type: Type of data ('telemetry', 'anomaly', 'prediction').

        Returns:
            True if buffered successfully, False if buffer is full.
        """
        if self.buffer_size >= self.max_buffer_size:
            logger.warning(
                "Buffer full (%d records). Dropping oldest data.",
                self.buffer_size,
            )
            # Drop oldest 10% to make room
            drop_count = self.max_buffer_size // 10
            self._upload_buffer = self._upload_buffer[drop_count:]

        record = {
            "data": data,
            "data_type": data_type,
            "device_id": self.device_id,
            "buffered_at": datetime.now(timezone.utc).isoformat(),
        }
        self._upload_buffer.append(record)
        return True

    def buffer_alert(self, alert: Dict[str, Any]) -> bool:
        """
        Buffer an alert for priority upload.

        Alerts are synced before regular telemetry data.
        """
        self._alert_buffer.append({
            "alert": alert,
            "device_id": self.device_id,
            "buffered_at": datetime.now(timezone.utc).isoformat(),
        })
        return True

    def sync_upload(self, force: bool = False) -> Optional[SyncRecord]:
        """
        Synchronize buffered data to the cloud.

        Args:
            force: Force sync even within cooldown period.

        Returns:
            SyncRecord if sync occurred, None if skipped.
        """
        now = time.time()
        if not force and (now - self._last_sync_time) < self.sync_interval_seconds:
            return None

        if self._status == SyncStatus.DISCONNECTED:
            logger.debug("Sync skipped: disconnected from cloud")
            return None

        if not self._upload_buffer and not self._alert_buffer:
            return None

        self._status = SyncStatus.SYNCING
        start = time.time()

        # Priority: sync alerts first
        alerts_to_sync = self._alert_buffer[:self.batch_size]
        data_to_sync = self._upload_buffer[:self.batch_size - len(alerts_to_sync)]

        total_records = len(alerts_to_sync) + len(data_to_sync)

        # Simulate network upload
        payload = json.dumps({
            "device_id": self.device_id,
            "alerts": alerts_to_sync,
            "data": data_to_sync,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        size_bytes = len(payload.encode("utf-8"))

        # Simulate latency
        time.sleep(self.simulate_latency_ms / 1000.0)

        # Remove synced items from buffers
        self._alert_buffer = self._alert_buffer[len(alerts_to_sync):]
        self._upload_buffer = self._upload_buffer[len(data_to_sync):]

        duration_ms = (time.time() - start) * 1000
        self._total_uploaded += total_records
        self._total_bytes_uploaded += size_bytes
        self._last_sync_time = now

        self._sync_counter += 1
        sync_record = SyncRecord(
            sync_id=f"SYNC-{self._sync_counter:06d}",
            direction=SyncDirection.UPLOAD,
            data_type="mixed",
            record_count=total_records,
            size_bytes=size_bytes,
            timestamp=datetime.now(timezone.utc),
            status="success",
            duration_ms=duration_ms,
        )
        self._sync_history.append(sync_record)

        self._status = SyncStatus.CONNECTED

        logger.info(
            "Cloud sync completed: %d records (%d bytes) in %.1fms",
            total_records,
            size_bytes,
            duration_ms,
        )
        return sync_record

    def check_model_updates(self) -> List[Dict[str, Any]]:
        """
        Check for available model updates from the cloud.

        Returns:
            List of available model update descriptors.
        """
        if self._status == SyncStatus.DISCONNECTED:
            return []

        # Simulate checking for updates
        return list(self._model_updates_available)

    def download_model_update(self, update_id: str) -> Optional[Dict[str, Any]]:
        """
        Download a model update from the cloud.

        Args:
            update_id: Identifier of the model update to download.

        Returns:
            Model update data or None if not available.
        """
        if self._status == SyncStatus.DISCONNECTED:
            logger.warning("Cannot download model: disconnected")
            return None

        for update in self._model_updates_available:
            if update.get("update_id") == update_id:
                self._total_downloaded += 1
                self._sync_counter += 1
                sync_record = SyncRecord(
                    sync_id=f"SYNC-{self._sync_counter:06d}",
                    direction=SyncDirection.DOWNLOAD,
                    data_type="model_update",
                    record_count=1,
                    size_bytes=update.get("size_bytes", 0),
                    timestamp=datetime.now(timezone.utc),
                    status="success",
                    duration_ms=self.simulate_latency_ms * 2,
                )
                self._sync_history.append(sync_record)
                return update

        return None

    def simulate_model_update_available(
        self,
        model_name: str,
        version: str,
        metrics: Dict[str, float],
    ) -> None:
        """Register a simulated model update for testing."""
        self._model_updates_available.append({
            "update_id": f"UPD-{len(self._model_updates_available) + 1:04d}",
            "model_name": model_name,
            "version": version,
            "metrics": metrics,
            "size_bytes": 1024 * 256,
            "published_at": datetime.now(timezone.utc).isoformat(),
        })

    def get_sync_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent synchronization history."""
        return [r.to_dict() for r in self._sync_history[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        """Return sync manager statistics."""
        return {
            "status": self._status.value,
            "device_id": self.device_id,
            "upload_buffer_size": len(self._upload_buffer),
            "alert_buffer_size": len(self._alert_buffer),
            "total_uploaded": self._total_uploaded,
            "total_downloaded": self._total_downloaded,
            "total_bytes_uploaded": self._total_bytes_uploaded,
            "total_syncs": self._sync_counter,
            "connection_failures": self._connection_failures,
            "model_updates_available": len(self._model_updates_available),
        }

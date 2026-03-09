"""
Edge model lifecycle manager.

Handles model loading, versioning, updates, rollback, and performance
monitoring for ML models deployed on edge devices. Supports model
hot-swapping without service interruption.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("edge.model_manager")


@dataclass
class ModelVersion:
    """Metadata for a deployed model version."""
    model_name: str
    version: str
    deployed_at: datetime
    model_hash: str
    metrics: Dict[str, float]
    is_active: bool = True
    inference_count: int = 0
    total_inference_time_ms: float = 0.0
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        avg_inference = (
            self.total_inference_time_ms / self.inference_count
            if self.inference_count > 0
            else 0.0
        )
        return {
            "model_name": self.model_name,
            "version": self.version,
            "deployed_at": self.deployed_at.isoformat(),
            "model_hash": self.model_hash,
            "metrics": self.metrics,
            "is_active": self.is_active,
            "inference_count": self.inference_count,
            "avg_inference_ms": round(avg_inference, 3),
            "error_count": self.error_count,
        }


class EdgeModelManager:
    """
    Manages ML model lifecycle on edge devices.

    Supports model versioning, performance tracking, automatic rollback
    on degradation, and model update management from cloud deployments.
    """

    def __init__(
        self,
        model_dir: str = "models/",
        max_versions: int = 3,
        validation_threshold: float = 0.85,
        rollback_on_failure: bool = True,
    ) -> None:
        """
        Initialize the model manager.

        Args:
            model_dir: Directory for storing model artifacts.
            max_versions: Maximum model versions to keep locally.
            validation_threshold: Minimum accuracy to deploy a model.
            rollback_on_failure: Auto-rollback on inference failure.
        """
        self.model_dir = Path(model_dir)
        self.max_versions = max_versions
        self.validation_threshold = validation_threshold
        self.rollback_on_failure = rollback_on_failure

        self._models: Dict[str, List[ModelVersion]] = {}
        self._active_versions: Dict[str, str] = {}
        self._performance_history: Dict[str, List[Dict[str, Any]]] = {}

        logger.info(
            "EdgeModelManager initialized: dir=%s, max_versions=%d, "
            "threshold=%.2f",
            model_dir,
            max_versions,
            validation_threshold,
        )

    def register_model(
        self,
        model_name: str,
        version: str,
        metrics: Dict[str, float],
        model_data: Optional[bytes] = None,
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model_name: Name of the model (e.g., 'anomaly_detector').
            version: Version string (e.g., '1.0.0').
            metrics: Training/validation metrics.
            model_data: Optional serialized model bytes for hashing.

        Returns:
            ModelVersion object for the registered model.
        """
        model_hash = hashlib.sha256(
            model_data or f"{model_name}:{version}:{time.time()}".encode()
        ).hexdigest()[:16]

        model_version = ModelVersion(
            model_name=model_name,
            version=version,
            deployed_at=datetime.now(timezone.utc),
            model_hash=model_hash,
            metrics=metrics,
            is_active=True,
        )

        if model_name not in self._models:
            self._models[model_name] = []

        # Deactivate previous versions
        for mv in self._models[model_name]:
            mv.is_active = False

        self._models[model_name].append(model_version)
        self._active_versions[model_name] = version

        # Prune old versions
        if len(self._models[model_name]) > self.max_versions:
            self._models[model_name] = self._models[model_name][-self.max_versions:]

        logger.info(
            "Model registered: %s v%s (hash: %s)",
            model_name,
            version,
            model_hash,
        )
        return model_version

    def get_active_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the currently active model version."""
        if model_name not in self._models:
            return None
        for mv in reversed(self._models[model_name]):
            if mv.is_active:
                return mv
        return None

    def get_version_history(self, model_name: str) -> List[Dict[str, Any]]:
        """Get version history for a model."""
        if model_name not in self._models:
            return []
        return [mv.to_dict() for mv in self._models[model_name]]

    def record_inference(
        self,
        model_name: str,
        inference_time_ms: float,
        success: bool = True,
    ) -> None:
        """
        Record an inference execution for performance tracking.

        Args:
            model_name: Name of the model.
            inference_time_ms: Time taken for inference in milliseconds.
            success: Whether the inference succeeded.
        """
        active = self.get_active_version(model_name)
        if active is None:
            return

        active.inference_count += 1
        active.total_inference_time_ms += inference_time_ms

        if not success:
            active.error_count += 1
            error_rate = active.error_count / active.inference_count
            if (
                self.rollback_on_failure
                and error_rate > 0.1
                and active.inference_count >= 10
            ):
                logger.warning(
                    "High error rate (%.1f%%) for %s v%s. Attempting rollback.",
                    error_rate * 100,
                    model_name,
                    active.version,
                )
                self.rollback(model_name)

        # Track performance history
        if model_name not in self._performance_history:
            self._performance_history[model_name] = []
        self._performance_history[model_name].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inference_time_ms": inference_time_ms,
            "success": success,
            "version": active.version,
        })
        # Keep only recent history
        if len(self._performance_history[model_name]) > 1000:
            self._performance_history[model_name] = (
                self._performance_history[model_name][-500:]
            )

    def rollback(self, model_name: str) -> bool:
        """
        Rollback to the previous model version.

        Args:
            model_name: Name of the model to rollback.

        Returns:
            True if rollback succeeded, False if no previous version exists.
        """
        if model_name not in self._models or len(self._models[model_name]) < 2:
            logger.warning("Cannot rollback %s: insufficient version history", model_name)
            return False

        # Deactivate current
        for mv in self._models[model_name]:
            mv.is_active = False

        # Activate previous
        previous = self._models[model_name][-2]
        previous.is_active = True
        previous.error_count = 0
        previous.inference_count = 0
        previous.total_inference_time_ms = 0.0
        self._active_versions[model_name] = previous.version

        logger.info(
            "Model %s rolled back to v%s", model_name, previous.version
        )
        return True

    def check_model_health(self, model_name: str) -> Dict[str, Any]:
        """
        Check the health status of a deployed model.

        Args:
            model_name: Name of the model to check.

        Returns:
            Health status dictionary.
        """
        active = self.get_active_version(model_name)
        if active is None:
            return {"status": "not_found", "model_name": model_name}

        avg_inference = (
            active.total_inference_time_ms / active.inference_count
            if active.inference_count > 0
            else 0.0
        )
        error_rate = (
            active.error_count / active.inference_count
            if active.inference_count > 0
            else 0.0
        )

        if error_rate > 0.1:
            status = "degraded"
        elif avg_inference > 500:
            status = "slow"
        elif active.inference_count == 0:
            status = "idle"
        else:
            status = "healthy"

        return {
            "status": status,
            "model_name": model_name,
            "version": active.version,
            "inference_count": active.inference_count,
            "avg_inference_ms": round(avg_inference, 3),
            "error_rate": round(error_rate, 4),
            "uptime_hours": round(
                (datetime.now(timezone.utc) - active.deployed_at).total_seconds() / 3600,
                2,
            ),
        }

    def validate_model_update(
        self, metrics: Dict[str, float]
    ) -> bool:
        """
        Validate if a model update meets deployment criteria.

        Args:
            metrics: Validation metrics for the candidate model.

        Returns:
            True if the model meets the validation threshold.
        """
        accuracy_keys = ["accuracy", "r2_score", "f1_score", "cv_r2_mean"]
        for key in accuracy_keys:
            if key in metrics:
                if metrics[key] >= self.validation_threshold:
                    return True
                else:
                    logger.warning(
                        "Model update rejected: %s=%.4f < threshold=%.4f",
                        key,
                        metrics[key],
                        self.validation_threshold,
                    )
                    return False
        # If no recognized metric, accept by default
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Return model manager statistics."""
        model_summaries = {}
        for name, versions in self._models.items():
            active = self.get_active_version(name)
            model_summaries[name] = {
                "total_versions": len(versions),
                "active_version": active.version if active else None,
                "total_inferences": sum(v.inference_count for v in versions),
            }

        return {
            "total_models": len(self._models),
            "max_versions": self.max_versions,
            "validation_threshold": self.validation_threshold,
            "models": model_summaries,
        }

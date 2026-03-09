"""
Application settings and configuration management.

Uses Pydantic for validation and supports environment variable overrides.
All configuration sections are independently configurable and composable
into the main AppSettings for full system configuration.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class InferenceRuntime(str, Enum):
    """Supported inference runtimes for edge deployment."""
    TFLITE = "tflite"
    ONNX = "onnx"


class EdgeConfig(BaseModel):
    """Configuration for the edge inference engine."""

    device_id: str = Field(default="edge-node-001", description="Unique identifier for this edge device")
    max_memory_mb: int = Field(default=512, ge=64, le=8192, description="Maximum memory for inference in MB")
    inference_timeout_ms: int = Field(default=500, ge=10, le=10000, description="Timeout per inference call in ms")
    preferred_runtime: InferenceRuntime = Field(default=InferenceRuntime.TFLITE)
    model_dir: str = Field(default="models/", description="Directory for storing model artifacts")
    enable_hot_swap: bool = Field(default=True, description="Enable model hot-swap without downtime")
    batch_size: int = Field(default=1, ge=1, le=64, description="Inference batch size")
    num_threads: int = Field(default=2, ge=1, le=16, description="Number of inference threads")

    @field_validator("model_dir")
    @classmethod
    def ensure_model_dir_trailing_slash(cls, v: str) -> str:
        return v if v.endswith("/") else f"{v}/"


class MQTTConfig(BaseModel):
    """Configuration for the MQTT client and message ingestion."""

    broker_host: str = Field(default="localhost", description="MQTT broker hostname")
    broker_port: int = Field(default=1883, ge=1, le=65535)
    username: Optional[str] = Field(default=None, description="MQTT auth username")
    password: Optional[str] = Field(default=None, description="MQTT auth password")
    client_id: str = Field(default="iot-maintenance-edge")
    topics: List[str] = Field(
        default=["sensors/+/telemetry", "sensors/+/status"],
        description="MQTT topic subscriptions with wildcard support",
    )
    qos: int = Field(default=1, ge=0, le=2, description="MQTT Quality of Service level")
    keepalive: int = Field(default=60, ge=10, le=3600, description="MQTT keepalive interval in seconds")
    use_tls: bool = Field(default=False, description="Enable TLS for MQTT connection")
    ca_cert_path: Optional[str] = Field(default=None, description="Path to CA certificate for TLS")
    offline_buffer_size: int = Field(default=10000, ge=100, le=1000000, description="Messages to buffer when offline")
    reconnect_delay_s: int = Field(default=5, ge=1, le=300)
    max_reconnect_attempts: int = Field(default=10, ge=1, le=100)


class ModelConfig(BaseModel):
    """Configuration for model management and versioning."""

    registry_url: Optional[str] = Field(default=None, description="Remote model registry URL")
    update_check_interval_s: int = Field(default=3600, ge=60, description="How often to check for model updates")
    max_model_versions: int = Field(default=3, ge=1, le=10, description="Max versions to keep locally")
    validation_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="Min accuracy to deploy a model")
    compression_enabled: bool = Field(default=True, description="Enable model quantization/pruning")
    rollback_on_failure: bool = Field(default=True, description="Auto-rollback on inference failure")
    ota_enabled: bool = Field(default=True, description="Enable Over-The-Air model updates")
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket for model artifacts")


class AnomalyConfig(BaseModel):
    """Configuration for anomaly detection algorithms."""

    isolation_forest_contamination: float = Field(
        default=0.05, ge=0.001, le=0.5, description="Expected anomaly fraction"
    )
    isolation_forest_n_estimators: int = Field(default=100, ge=10, le=1000)
    z_score_threshold: float = Field(default=3.0, ge=1.0, le=10.0, description="Z-score threshold for anomaly flag")
    sliding_window_size: int = Field(default=100, ge=10, le=10000, description="Sliding window length in samples")
    spc_control_limit_sigma: float = Field(default=3.0, ge=1.0, le=6.0, description="SPC control limit in sigma")
    correlation_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Multi-sensor correlation threshold")
    min_samples_for_detection: int = Field(default=50, ge=10, description="Min samples before anomaly detection starts")
    alert_cooldown_s: int = Field(default=300, ge=0, description="Minimum seconds between repeated alerts")


class RULConfig(BaseModel):
    """Configuration for Remaining Useful Life prediction."""

    prediction_horizon_hours: int = Field(default=720, ge=1, description="Max RUL prediction horizon in hours")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence interval level")
    weibull_shape_prior: float = Field(default=2.0, ge=0.1, le=10.0, description="Weibull distribution shape prior")
    weibull_scale_prior: float = Field(default=1000.0, ge=1.0, description="Weibull distribution scale prior")
    health_index_weights: Dict[str, float] = Field(
        default={"vibration": 0.3, "temperature": 0.25, "pressure": 0.2, "current": 0.15, "acoustic": 0.1},
        description="Sensor weights for composite health index",
    )
    degradation_trend_window: int = Field(default=48, ge=6, description="Window in hours for degradation trending")
    critical_health_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Health index below this triggers alert")
    warning_health_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class StorageConfig(BaseModel):
    """Configuration for Cassandra time-series and Redis cache storage."""

    cassandra_hosts: List[str] = Field(default=["localhost"], description="Cassandra contact points")
    cassandra_port: int = Field(default=9042, ge=1, le=65535)
    cassandra_keyspace: str = Field(default="iot_maintenance")
    cassandra_replication_factor: int = Field(default=1, ge=1, le=5)
    cassandra_consistency_level: str = Field(default="LOCAL_QUORUM")
    ttl_raw_data_days: int = Field(default=30, ge=1, description="TTL for raw sensor data in days")
    ttl_aggregated_data_days: int = Field(default=365, ge=1, description="TTL for downsampled data in days")
    downsample_interval_s: int = Field(default=300, ge=60, description="Downsample interval in seconds")

    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379, ge=1, le=65535)
    redis_db: int = Field(default=0, ge=0, le=15)
    redis_password: Optional[str] = Field(default=None)
    prediction_cache_ttl_s: int = Field(default=600, ge=10, description="Cache TTL for predictions in seconds")
    health_index_cache_ttl_s: int = Field(default=120, ge=10, description="Cache TTL for health indices")


class APIConfig(BaseModel):
    """Configuration for the FastAPI REST interface."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=2, ge=1, le=16)
    cors_origins: List[str] = Field(default=["*"])
    api_prefix: str = Field(default="/api/v1")
    enable_docs: bool = Field(default=True, description="Enable Swagger/ReDoc endpoints")
    rate_limit_per_minute: int = Field(default=120, ge=1, le=10000)
    request_timeout_s: int = Field(default=30, ge=1, le=300)


class AppSettings(BaseModel):
    """Root application settings composing all sub-configurations."""

    app_name: str = Field(default="IoT Predictive Maintenance Edge")
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    log_file: Optional[str] = Field(default=None)

    edge: EdgeConfig = Field(default_factory=EdgeConfig)
    mqtt: MQTTConfig = Field(default_factory=MQTTConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    rul: RULConfig = Field(default_factory=RULConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "AppSettings":
        """Load settings from a YAML configuration file with env var overrides."""
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        env_overrides = {
            "environment": os.getenv("APP_ENV"),
            "log_level": os.getenv("LOG_LEVEL"),
        }
        for key, value in env_overrides.items():
            if value is not None:
                raw[key] = value

        mqtt_env = {
            "broker_host": os.getenv("MQTT_BROKER_HOST"),
            "broker_port": os.getenv("MQTT_BROKER_PORT"),
            "username": os.getenv("MQTT_USERNAME"),
            "password": os.getenv("MQTT_PASSWORD"),
        }
        if "mqtt" not in raw:
            raw["mqtt"] = {}
        for key, value in mqtt_env.items():
            if value is not None:
                raw["mqtt"][key] = int(value) if key == "broker_port" else value

        storage_env = {
            "redis_host": os.getenv("REDIS_HOST"),
            "redis_port": os.getenv("REDIS_PORT"),
            "redis_password": os.getenv("REDIS_PASSWORD"),
        }
        if "storage" not in raw:
            raw["storage"] = {}
        for key, value in storage_env.items():
            if value is not None:
                raw["storage"][key] = int(value) if key == "redis_port" else value

        return cls(**raw)


@lru_cache(maxsize=1)
def get_settings(config_path: str = "config/maintenance_config.yaml") -> AppSettings:
    """
    Get cached application settings singleton.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated AppSettings instance.
    """
    return AppSettings.from_yaml(config_path)

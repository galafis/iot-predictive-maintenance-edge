"""
Sensor data preprocessing and feature extraction for edge computing.

Provides signal processing utilities including noise removal, normalization,
outlier detection, and statistical feature extraction optimized for
real-time edge deployment with minimal memory footprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.utils.logger import get_logger

logger = get_logger("sensors.preprocessor")


@dataclass
class StatisticalFeatures:
    """Extracted statistical features from a sensor signal window."""
    mean: float
    std: float
    rms: float
    peak: float
    crest_factor: float
    kurtosis: float
    skewness: float
    min_value: float
    max_value: float
    peak_to_peak: float
    variance: float
    energy: float

    def to_dict(self) -> Dict[str, float]:
        """Serialize features to dictionary."""
        return {
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "rms": round(self.rms, 6),
            "peak": round(self.peak, 6),
            "crest_factor": round(self.crest_factor, 6),
            "kurtosis": round(self.kurtosis, 6),
            "skewness": round(self.skewness, 6),
            "min_value": round(self.min_value, 6),
            "max_value": round(self.max_value, 6),
            "peak_to_peak": round(self.peak_to_peak, 6),
            "variance": round(self.variance, 6),
            "energy": round(self.energy, 6),
        }


class SensorPreprocessor:
    """
    Signal preprocessing pipeline for IoT sensor data.

    Applies sequential processing steps: noise removal, normalization,
    outlier detection, and feature extraction. Optimized for real-time
    edge computing with configurable window sizes.
    """

    def __init__(
        self,
        window_size: int = 5,
        iqr_multiplier: float = 1.5,
        normalization_method: str = "z-score",
    ) -> None:
        """
        Initialize the preprocessor.

        Args:
            window_size: Window size for moving average noise removal.
            iqr_multiplier: IQR multiplier for outlier detection threshold.
            normalization_method: 'z-score' or 'min-max'.
        """
        self.window_size = window_size
        self.iqr_multiplier = iqr_multiplier
        self.normalization_method = normalization_method

        self._running_stats: Dict[str, Dict[str, float]] = {}
        self._processed_count = 0

        logger.info(
            "SensorPreprocessor initialized: window=%d, iqr_mult=%.1f, norm=%s",
            window_size,
            iqr_multiplier,
            normalization_method,
        )

    def remove_noise(self, signal: np.ndarray, window_size: Optional[int] = None) -> np.ndarray:
        """
        Remove noise using a moving average filter.

        Args:
            signal: Raw sensor signal array.
            window_size: Override default window size.

        Returns:
            Smoothed signal array of the same length.
        """
        ws = window_size or self.window_size
        if len(signal) < ws:
            return signal.copy()

        kernel = np.ones(ws) / ws
        # Use 'same' mode to preserve array length
        smoothed = np.convolve(signal, kernel, mode="same")

        # Fix edge effects by using partial windows
        half = ws // 2
        for i in range(half):
            smoothed[i] = np.mean(signal[: i + half + 1])
            smoothed[-(i + 1)] = np.mean(signal[-(i + half + 1):])

        return smoothed

    def normalize(
        self,
        signal: np.ndarray,
        method: Optional[str] = None,
        reference_mean: Optional[float] = None,
        reference_std: Optional[float] = None,
    ) -> np.ndarray:
        """
        Normalize sensor signal values.

        Args:
            signal: Input signal array.
            method: 'z-score' or 'min-max'. Uses instance default if None.
            reference_mean: Reference mean for z-score (for consistent scaling).
            reference_std: Reference std for z-score.

        Returns:
            Normalized signal array.
        """
        norm_method = method or self.normalization_method

        if norm_method == "z-score":
            mean = reference_mean if reference_mean is not None else np.mean(signal)
            std = reference_std if reference_std is not None else np.std(signal)
            if std < 1e-10:
                return np.zeros_like(signal)
            return (signal - mean) / std

        elif norm_method == "min-max":
            min_val = np.min(signal)
            max_val = np.max(signal)
            range_val = max_val - min_val
            if range_val < 1e-10:
                return np.zeros_like(signal)
            return (signal - min_val) / range_val

        else:
            raise ValueError(f"Unknown normalization method: {norm_method}")

    def detect_outliers(
        self,
        signal: np.ndarray,
        iqr_multiplier: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outliers using the Interquartile Range (IQR) method.

        Args:
            signal: Input signal array.
            iqr_multiplier: Override default IQR multiplier.

        Returns:
            Tuple of (outlier_mask, cleaned_signal) where outlier_mask is
            a boolean array (True = outlier) and cleaned_signal has outliers
            replaced with interpolated values.
        """
        mult = iqr_multiplier or self.iqr_multiplier

        q1 = np.percentile(signal, 25)
        q3 = np.percentile(signal, 75)
        iqr = q3 - q1

        lower_bound = q1 - mult * iqr
        upper_bound = q3 + mult * iqr

        outlier_mask = (signal < lower_bound) | (signal > upper_bound)

        # Replace outliers with linear interpolation
        cleaned = signal.copy()
        if np.any(outlier_mask):
            indices = np.arange(len(signal))
            valid_mask = ~outlier_mask
            if np.sum(valid_mask) >= 2:
                cleaned[outlier_mask] = np.interp(
                    indices[outlier_mask],
                    indices[valid_mask],
                    signal[valid_mask],
                )
            else:
                median = np.median(signal)
                cleaned[outlier_mask] = median

        return outlier_mask, cleaned

    def extract_statistical_features(self, signal: np.ndarray) -> StatisticalFeatures:
        """
        Extract comprehensive statistical features from a signal window.

        Args:
            signal: Input signal array (typically one window of data).

        Returns:
            StatisticalFeatures dataclass with all computed features.
        """
        if len(signal) == 0:
            return StatisticalFeatures(
                mean=0.0, std=0.0, rms=0.0, peak=0.0,
                crest_factor=0.0, kurtosis=0.0, skewness=0.0,
                min_value=0.0, max_value=0.0, peak_to_peak=0.0,
                variance=0.0, energy=0.0,
            )

        mean = float(np.mean(signal))
        std = float(np.std(signal))
        rms = float(np.sqrt(np.mean(signal ** 2)))
        peak = float(np.max(np.abs(signal)))
        crest_factor = peak / rms if rms > 1e-10 else 0.0
        min_val = float(np.min(signal))
        max_val = float(np.max(signal))
        peak_to_peak = max_val - min_val
        variance = float(np.var(signal))
        energy = float(np.sum(signal ** 2))

        # Use scipy for robust kurtosis and skewness
        if len(signal) >= 4:
            kurt = float(scipy_stats.kurtosis(signal, fisher=True))
            skew = float(scipy_stats.skew(signal))
        else:
            kurt = 0.0
            skew = 0.0

        return StatisticalFeatures(
            mean=mean,
            std=std,
            rms=rms,
            peak=peak,
            crest_factor=crest_factor,
            kurtosis=kurt,
            skewness=skew,
            min_value=min_val,
            max_value=max_val,
            peak_to_peak=peak_to_peak,
            variance=variance,
            energy=energy,
        )

    def process_signal(
        self,
        signal: np.ndarray,
        remove_noise_flag: bool = True,
        normalize_flag: bool = True,
        remove_outliers: bool = True,
    ) -> Tuple[np.ndarray, StatisticalFeatures, np.ndarray]:
        """
        Apply the full preprocessing pipeline to a sensor signal.

        Args:
            signal: Raw sensor signal array.
            remove_noise_flag: Apply moving average smoothing.
            normalize_flag: Apply normalization.
            remove_outliers: Detect and replace outliers.

        Returns:
            Tuple of (processed_signal, features, outlier_mask).
        """
        processed = signal.copy().astype(np.float64)

        if remove_outliers:
            outlier_mask, processed = self.detect_outliers(processed)
        else:
            outlier_mask = np.zeros(len(signal), dtype=bool)

        if remove_noise_flag:
            processed = self.remove_noise(processed)

        # Extract features before normalization (on physical values)
        features = self.extract_statistical_features(processed)

        if normalize_flag:
            processed = self.normalize(processed)

        self._processed_count += 1
        return processed, features, outlier_mask

    def process_multi_sensor(
        self,
        sensor_data: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple sensor signals and extract features from each.

        Args:
            sensor_data: Dict mapping sensor_type to signal arrays.

        Returns:
            Dict mapping sensor_type to processing results including
            processed signal, features, and outlier info.
        """
        results = {}
        for sensor_type, signal in sensor_data.items():
            processed, features, outlier_mask = self.process_signal(signal)
            results[sensor_type] = {
                "processed_signal": processed,
                "features": features,
                "outlier_count": int(np.sum(outlier_mask)),
                "outlier_ratio": float(np.mean(outlier_mask)),
            }
        return results

    def create_feature_vector(
        self,
        multi_sensor_results: Dict[str, Dict[str, Any]],
    ) -> np.ndarray:
        """
        Create a flat feature vector from multi-sensor processing results.

        Args:
            multi_sensor_results: Output from process_multi_sensor().

        Returns:
            1D numpy array with all features concatenated.
        """
        feature_vectors = []
        for sensor_type in sorted(multi_sensor_results.keys()):
            result = multi_sensor_results[sensor_type]
            features = result["features"]
            feature_vectors.extend([
                features.mean,
                features.std,
                features.rms,
                features.peak,
                features.crest_factor,
                features.kurtosis,
                features.skewness,
                features.peak_to_peak,
                features.energy,
            ])
        return np.array(feature_vectors)

    def create_feature_dataframe(
        self,
        multi_sensor_results: Dict[str, Dict[str, Any]],
        machine_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Create a feature DataFrame from multi-sensor processing results.

        Args:
            multi_sensor_results: Output from process_multi_sensor().
            machine_id: Optional machine identifier column.

        Returns:
            DataFrame with one row per sensor, features as columns.
        """
        rows = []
        for sensor_type in sorted(multi_sensor_results.keys()):
            result = multi_sensor_results[sensor_type]
            features_dict = result["features"].to_dict()
            features_dict["sensor_type"] = sensor_type
            features_dict["outlier_count"] = result["outlier_count"]
            features_dict["outlier_ratio"] = result["outlier_ratio"]
            if machine_id:
                features_dict["machine_id"] = machine_id
            rows.append(features_dict)

        return pd.DataFrame(rows)

    def get_stats(self) -> Dict[str, Any]:
        """Return preprocessor usage statistics."""
        return {
            "processed_count": self._processed_count,
            "window_size": self.window_size,
            "iqr_multiplier": self.iqr_multiplier,
            "normalization_method": self.normalization_method,
        }

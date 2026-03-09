"""Tests for sensor data preprocessor module."""

import numpy as np
import pytest

from src.sensors.preprocessor import SensorPreprocessor, StatisticalFeatures


class TestSensorPreprocessor:
    """Test suite for SensorPreprocessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = SensorPreprocessor(
            window_size=5, iqr_multiplier=1.5, normalization_method="z-score"
        )
        self.rng = np.random.default_rng(42)

    def test_remove_noise_smoothing(self):
        """Test that noise removal reduces signal variance."""
        # Create noisy signal
        signal = np.sin(np.linspace(0, 4 * np.pi, 100)) + self.rng.normal(0, 0.5, 100)
        smoothed = self.preprocessor.remove_noise(signal)

        assert len(smoothed) == len(signal)
        # Smoothed signal should have lower variance than noisy input
        assert np.var(smoothed) < np.var(signal)

    def test_normalize_zscore(self):
        """Test z-score normalization produces zero mean and unit std."""
        signal = self.rng.normal(50, 10, 200)
        normalized = self.preprocessor.normalize(signal, method="z-score")

        assert abs(np.mean(normalized)) < 0.01
        assert abs(np.std(normalized) - 1.0) < 0.01

    def test_normalize_minmax(self):
        """Test min-max normalization produces range [0, 1]."""
        signal = self.rng.normal(50, 10, 200)
        normalized = self.preprocessor.normalize(signal, method="min-max")

        assert np.min(normalized) >= -0.01
        assert np.max(normalized) <= 1.01
        assert abs(np.min(normalized)) < 0.01
        assert abs(np.max(normalized) - 1.0) < 0.01

    def test_detect_outliers_iqr(self):
        """Test IQR outlier detection identifies injected outliers."""
        signal = self.rng.normal(0, 1, 100)
        # Inject outliers
        signal[10] = 20.0
        signal[50] = -15.0

        outlier_mask, cleaned = self.preprocessor.detect_outliers(signal)

        assert outlier_mask[10] == True
        assert outlier_mask[50] == True
        assert len(cleaned) == len(signal)
        # Cleaned values should be interpolated (not the outlier values)
        assert abs(cleaned[10]) < 10.0
        assert abs(cleaned[50]) < 10.0

    def test_extract_statistical_features(self):
        """Test feature extraction produces all expected features."""
        signal = self.rng.normal(5.0, 1.5, 200)
        features = self.preprocessor.extract_statistical_features(signal)

        assert isinstance(features, StatisticalFeatures)
        assert abs(features.mean - 5.0) < 0.5
        assert features.std > 0
        assert features.rms > 0
        assert features.peak > 0
        assert features.crest_factor > 0
        assert features.peak_to_peak > 0
        assert features.variance > 0
        assert features.energy > 0

    def test_features_serialization(self):
        """Test statistical features serialize to dictionary."""
        signal = self.rng.normal(5.0, 1.5, 200)
        features = self.preprocessor.extract_statistical_features(signal)
        features_dict = features.to_dict()

        assert len(features_dict) == 12
        assert "mean" in features_dict
        assert "std" in features_dict
        assert "rms" in features_dict
        assert "kurtosis" in features_dict
        assert "skewness" in features_dict

    def test_process_signal_full_pipeline(self):
        """Test the full preprocessing pipeline."""
        signal = self.rng.normal(10, 2, 100)
        signal[25] = 50.0  # Inject outlier

        processed, features, outlier_mask = self.preprocessor.process_signal(signal)

        assert len(processed) == len(signal)
        assert isinstance(features, StatisticalFeatures)
        assert outlier_mask[25] == True

    def test_process_multi_sensor(self):
        """Test multi-sensor processing."""
        sensor_data = {
            "vibration": self.rng.normal(2.5, 0.3, 50),
            "temperature": self.rng.normal(65, 3, 50),
            "pressure": self.rng.normal(4.5, 0.2, 50),
        }

        results = self.preprocessor.process_multi_sensor(sensor_data)

        assert len(results) == 3
        assert "vibration" in results
        assert "features" in results["vibration"]
        assert "outlier_count" in results["vibration"]

    def test_create_feature_vector(self):
        """Test feature vector creation from multi-sensor results."""
        sensor_data = {
            "vibration": self.rng.normal(2.5, 0.3, 50),
            "temperature": self.rng.normal(65, 3, 50),
        }
        results = self.preprocessor.process_multi_sensor(sensor_data)
        vector = self.preprocessor.create_feature_vector(results)

        # 9 features per sensor, 2 sensors
        assert len(vector) == 18
        assert isinstance(vector, np.ndarray)

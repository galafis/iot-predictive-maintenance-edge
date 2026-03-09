"""
IoT Predictive Maintenance Edge Computing - Main Demo

Simulates a factory floor with 5 industrial machines, each equipped
with multiple sensors. Demonstrates the complete edge computing
pipeline: data ingestion, preprocessing, anomaly detection, RUL
prediction, alert generation, cloud synchronization, and dashboard
monitoring.

Usage:
    python main.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone

from src.edge.edge_processor import EdgeProcessor
from src.edge.model_manager import EdgeModelManager
from src.monitoring.dashboard import DashboardMetrics
from src.sensors.data_ingestion import MachineState
from src.sync.cloud_sync import CloudSyncManager
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger(name="iot_maintenance", level="INFO")


def print_header() -> None:
    """Print the application header."""
    header = """
================================================================================
     ___ ___ _____   ___              _ _      _   _
    |_ _/ _ |_   _| | _ \\_ _ ___ __| (_)__ _| |_(_)_ _____
     | | (_) || |   |  _/ '_/ -_) _` | / _|  _| \\ V / -_)
    |___\\___/ |_|   |_| |_| \\___\\__,_|_\\__|\\__|_|\\_/\\___|

    __  __      _       _
   |  \\/  |__ _(_)_ __ | |_ ___ _ _  __ _ _ _  __ ___
   | |\\/| / _` | | '  \\|  _/ -_) ' \\/ _` | ' \\/ _/ -_)
   |_|  |_\\__,_|_|_||_| \\__\\___|_||_\\__,_|_||_\\__\\___|

                    Edge Computing Platform
================================================================================
"""
    print(header)


def run_demo() -> None:
    """
    Run the complete predictive maintenance demo.

    Simulates 5 machines with different health states:
    - Machine 1-2: Normal operation (healthy)
    - Machine 3: Degrading bearings
    - Machine 4: Advanced degradation (near failure)
    - Machine 5: Normal initially, degrades during simulation
    """
    print_header()
    print("  Starting IoT Predictive Maintenance Edge Demo...")
    print("  Simulating factory floor with 5 industrial machines\n")

    # ---- Initialize edge components ----
    print("  [1/6] Initializing edge processor...")
    edge = EdgeProcessor(
        device_id="edge-gateway-factory-01",
        batch_size=50,
        window_size=20,
    )

    model_manager = EdgeModelManager(
        model_dir="models/",
        max_versions=3,
        validation_threshold=0.80,
    )

    cloud_sync = CloudSyncManager(
        device_id="edge-gateway-factory-01",
        batch_size=100,
        simulate_latency_ms=10.0,
    )

    dashboard = DashboardMetrics(device_id="edge-gateway-factory-01")

    # ---- Register machines ----
    print("  [2/6] Registering machines...\n")

    machines = {
        "CNC-MILL-001": {"state": MachineState.NORMAL, "desc": "CNC Milling Machine - Bay 1", "seed": 101},
        "CNC-MILL-002": {"state": MachineState.NORMAL, "desc": "CNC Milling Machine - Bay 2", "seed": 102},
        "PUMP-HYD-003": {"state": MachineState.DEGRADING, "desc": "Hydraulic Pump - Line 3", "seed": 103},
        "COMPRESSOR-004": {"state": MachineState.DEGRADING, "desc": "Air Compressor - Utility", "seed": 104},
        "CONVEYOR-005": {"state": MachineState.NORMAL, "desc": "Conveyor Belt Drive - Assembly", "seed": 105},
    }

    for machine_id, config in machines.items():
        edge.register_machine(
            machine_id=machine_id,
            initial_state=config["state"],
            random_seed=config["seed"],
        )
        status_label = config["state"].value.upper()
        print(f"    + {machine_id:<20} | {config['desc']:<40} | State: {status_label}")

    # ---- Train ML models ----
    print("\n  [3/6] Training ML models on synthetic degradation data...")
    start = time.time()
    training_metrics = edge.train_models(
        n_training_machines=30,
        max_life_cycles=200,
    )
    train_time = time.time() - start

    # Register models in model manager
    model_manager.register_model(
        model_name="anomaly_detector",
        version="1.0.0",
        metrics=training_metrics["anomaly_detector"],
    )
    model_manager.register_model(
        model_name="rul_predictor",
        version="1.0.0",
        metrics=training_metrics["rul_predictor"],
    )

    dashboard.update_model_metrics("anomaly_detector", training_metrics["anomaly_detector"])
    dashboard.update_model_metrics("rul_predictor", training_metrics["rul_predictor"])

    print(f"    Models trained in {train_time:.2f}s")
    print(f"    Anomaly Detector - Training anomaly rate: "
          f"{training_metrics['anomaly_detector']['anomaly_rate']:.2%}")
    rul_score = training_metrics['rul_predictor'].get('train_r2_score', 0)
    print(f"    RUL Predictor    - R2 Score: {rul_score:.4f}")

    # ---- Run processing cycles ----
    print("\n  [4/6] Running edge processing pipeline...")
    print(f"    Simulating {30} processing cycles per machine...\n")

    total_cycles = 30
    all_alerts = []

    for cycle in range(total_cycles):
        # Simulate Machine 4 getting worse over time
        if cycle == 10:
            edge.set_machine_state("COMPRESSOR-004", MachineState.FAILURE)

        # Simulate Machine 5 starting to degrade
        if cycle == 15:
            edge.set_machine_state("CONVEYOR-005", MachineState.DEGRADING)

        results = edge.process_all_machines()

        for result in results:
            # Record latency
            dashboard.record_latency(result.processing_time_ms)

            # Update machine status on dashboard
            if result.rul_prediction and result.anomaly_result:
                dashboard.update_machine_status(
                    machine_id=result.machine_id,
                    state=machines[result.machine_id]["state"].value
                    if result.machine_id in machines
                    else "unknown",
                    health_index=result.rul_prediction.get("health_index", 1.0),
                    rul_cycles=result.rul_prediction.get("rul_cycles", 999),
                    anomaly_score=result.anomaly_result.get("anomaly_score", 0.0),
                )

            # Update sensor health
            for sensor_type, value in result.sensor_readings.items():
                dashboard.update_sensor_health(
                    machine_id=result.machine_id,
                    sensor_type=sensor_type,
                    value=value,
                )

            # Record alerts
            for alert in result.alerts:
                dashboard.record_alert(alert.get("severity", "INFO"))
                all_alerts.append(alert)

            # Buffer data for cloud sync
            cloud_sync.buffer_data(result.to_dict(), data_type="telemetry")
            for alert in result.alerts:
                cloud_sync.buffer_alert(alert)

        # Periodic sync
        if (cycle + 1) % 10 == 0:
            sync_result = cloud_sync.sync_upload(force=True)
            if sync_result:
                dashboard.update_sync_metrics(
                    syncs=1,
                    bytes_uploaded=sync_result.size_bytes,
                )

        # Progress indicator
        if (cycle + 1) % 10 == 0:
            print(f"    Cycle {cycle + 1}/{total_cycles} completed "
                  f"({len(all_alerts)} alerts generated)")

    # ---- Display Results ----
    print("\n  [5/6] Processing complete. Generating reports...\n")

    # Print dashboard
    report = dashboard.generate_status_report()
    print(report)

    # Print recent critical alerts
    print("\n  RECENT ALERTS (last 15)")
    print("-" * 72)
    recent_alerts = all_alerts[-15:] if all_alerts else []
    if recent_alerts:
        for alert in recent_alerts:
            severity = alert.get("severity", "INFO")
            machine = alert.get("machine_id", "unknown")
            title = alert.get("title", "No title")
            print(f"  [{severity:<9}] {machine:<20} | {title}")
    else:
        print("  No alerts generated during this simulation.")

    # ---- Final Stats ----
    print("\n  [6/6] Final Statistics")
    print("-" * 72)
    edge_stats = edge.get_stats()
    sync_stats = cloud_sync.get_stats()
    model_stats = model_manager.get_stats()

    print(f"  Total Processing Cycles  : {edge_stats['total_cycles']}")
    print(f"  Avg Processing Time      : {edge_stats['avg_processing_time_ms']:.2f} ms")
    print(f"  Total Alerts Generated   : {len(all_alerts)}")
    print(f"  Cloud Syncs Completed    : {sync_stats['total_syncs']}")
    print(f"  Data Uploaded            : {sync_stats['total_bytes_uploaded']} bytes")
    print(f"  Models Deployed          : {model_stats['total_models']}")
    print(f"  System Uptime            : {dashboard.uptime_formatted}")

    print("\n" + "=" * 72)
    print("  Demo completed successfully.")
    print("  In production, this pipeline runs continuously on edge gateways")
    print("  monitoring industrial equipment for predictive maintenance.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n  Demo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

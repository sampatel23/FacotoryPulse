"""
FactoryPulse - Industrial Sensor Producer

Purpose:
1. Generate simulated industrial machine sensor readings.
2. Simulate gradual machine degradation.
3. Publish each reading as a JSON message to Kafka topic sensor.raw.

IMPORTANT: All readings are simulated, not collected from a real machine.
"""

import json
import math
import random
import time
from datetime import datetime, timezone

from kafka import KafkaProducer
from kafka.errors import KafkaError


# Kafka is exposed to Windows on port 29092 in compose.yaml.
KAFKA_BOOTSTRAP_SERVERS = "127.0.0.1:29092"
KAFKA_TOPIC = "sensor.raw"
MACHINE_IDS = ["MACHINE-001", "MACHINE-002", "MACHINE-003"]


def create_producer():
    """Connect to Kafka and serialize Python dictionaries as JSON bytes."""
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        acks="all",  # Wait for the broker to acknowledge each message.
        retries=5,   # Retry transient send failures.
    )


def generate_reading(tick, machine_id):
    """Generate simulated readings for a particular machine."""

    # Give each simulated machine a different operating profile.
    profiles = {
        "MACHINE-001": {"temp": 0.0, "rpm": 0.0, "torque": 0.0},
        "MACHINE-002": {"temp": 3.0, "rpm": -80.0, "torque": 5.0},
        "MACHINE-003": {"temp": 6.0, "rpm": -150.0, "torque": 10.0},
    }

    profile = profiles[machine_id]

    # Increase degradation gradually, then reset to simulate maintenance.
    degradation = (tick % 120) / 120.0

    air_temp = random.uniform(295.0, 299.0) + degradation * 4.0 + profile["temp"]
    process_temp = air_temp + random.uniform(8.0, 12.0) + degradation * 5.0
    rpm = random.uniform(1400.0, 1600.0) - degradation * 180.0 + profile["rpm"]
    torque = random.uniform(35.0, 45.0) + degradation * 22.0 + profile["torque"]
    tool_wear = min(240.0, tick * 0.5 + random.uniform(0.0, 1.0))

    vibration = random.uniform(1.0, 2.5) + degradation * 5.0
    hydraulic_pressure = random.uniform(85.0, 100.0) - degradation * 20.0

    if process_temp > 310.0:
        fault_label = "Heat Dissipation"
    elif torque > 55.0 or hydraulic_pressure < 75.0:
        fault_label = "Mechanical Overstrain"
    elif tool_wear > 200.0 or vibration > 6.0:
        fault_label = "Bearing/Tool Wear"
    elif rpm < 1300.0:
        fault_label = "Power/Speed Instability"
    else:
        fault_label = "NORMAL"

    return {
        "machine_id": machine_id,
        "machine_type": "M",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "air_temp_K": round(air_temp, 2),
        "process_temp_K": round(process_temp, 2),
        "rotational_speed_rpm": round(rpm, 2),
        "torque_nm": round(torque, 2),
        "tool_wear_min": round(tool_wear, 2),
        "vibration_mm_s": round(vibration, 2),
        "hydraulic_pressure_bar": round(hydraulic_pressure, 2),
        "degradation_index": round(degradation, 3),
        "fault_label": fault_label,
    }


def main():
    """Publish one sensor reading per second until interrupted."""
    producer = None

    try:
        producer = create_producer()
        print(f"Connected to Kafka: {KAFKA_BOOTSTRAP_SERVERS}")
        print(f"Publishing readings to topic: {KAFKA_TOPIC}")
        print("Press Ctrl+C to stop.\n")

        tick = 0

        while True:
            for machine_id in MACHINE_IDS:
                reading = generate_reading(tick, machine_id)

            future = producer.send(
                KAFKA_TOPIC,
                key=reading["machine_id"].encode("utf-8"),
                value=reading,
            )

            metadata = future.get(timeout=10)

            print(
                f"Machine={machine_id} | "
                f"topic={metadata.topic} | "
                f"offset={metadata.offset} | "
                f"temp={reading['process_temp_K']} K | "
                f"RPM={reading['rotational_speed_rpm']} | "
                f"status={reading['fault_label']}"
            )

            tick += 1

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping sensor producer...")

    except (KafkaError, OSError) as error:
        print(f"Kafka connection/send error: {error}")
        print("Check that the Kafka container is running.")

    finally:
        if producer is not None:
            producer.flush(timeout=10)
            producer.close()
        print("Sensor producer stopped.")


if __name__ == "__main__":
    main()
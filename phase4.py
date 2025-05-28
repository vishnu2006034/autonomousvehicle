import time
import threading
import random
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Sensor simulation functions
def get_lidar_data():
    """Simulates LIDAR data with random distance readings."""
    return {'distance_to_object': round(random.uniform(0.5, 10.0), 2)}

def get_camera_data():
    """Simulates camera object detection."""
    return {'object_detected': random.choice(['car', 'pedestrian', 'none'])}

def get_gps_data():
    """Returns static GPS coordinates (mock)."""
    return {'latitude': 37.7749, 'longitude': -122.4194}

def get_imu_data():
    """Simulates IMU sensor data."""
    return {
        'acceleration': round(random.uniform(-3, 3), 2),
        'rotation': round(random.uniform(-180, 180), 2)
    }

# Sensor fusion
def fuse_sensor_data():
    """Combines all sensor data into one dictionary."""
    return {
        'lidar': get_lidar_data(),
        'camera': get_camera_data(),
        'gps': get_gps_data(),
        'imu': get_imu_data()
    }

# Decision-making logic
def make_decision(fused_data, safety_distance=2.0):
    """Decides movement based on LIDAR and camera input."""
    distance = fused_data['lidar']['distance_to_object']
    object_detected = fused_data['camera']['object_detected']

    if distance < safety_distance or object_detected in ['car', 'pedestrian']:
        return "Stop"
    return "Move Forward"

# Mock encryption
def encrypt_data(data):
    """Mock encryption using SHA-256 hashing (symbolic)."""
    return hashlib.sha256(str(data).encode('utf-8')).hexdigest()

# Control loop interface
def control_interface(run_time=10):
    logging.info("Autonomous Navigation System Started.")
    for cycle in range(run_time):
        data = fuse_sensor_data()
        decision = make_decision(data)

        logging.info(f"Cycle {cycle+1}")
        logging.info(f"Sensor Data: {data}")
        logging.info(f"Encrypted Log: {encrypt_data(data)}")
        logging.info(f"System Decision: {decision}\n")

        time.sleep(1)

# Test simulation
def run_autonomous_test():
    logging.info("Running simulated autonomous test with live data streams...")
    control_thread = threading.Thread(target=control_interface)
    control_thread.start()
    control_thread.join()

# Entry point
if __name__ == "__main__":
    run_autonomous_test()

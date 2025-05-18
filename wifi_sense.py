import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import ctypes
import os

# Configuration parameters
SCAN_DURATION = 30      # Seconds to scan Wi-Fi signals
GRID_SIZE = (10, 10)    # 10x10 grid for heatmap
SIGNAL_THRESHOLD = 5.0  # Variance threshold for motion/presence detection
MOVING_AVG_WINDOW = 5   # Window size for smoothing signal data
ROOM_DIM = (5, 5)       # Assumed room dimensions in meters
PATH_LOSS_EXPONENT = 2.5  # Path-loss exponent for distance estimation

# Global variables
signal_data = []        # Store signal strength values
timestamps = []         # Store timestamps
grid_signal = np.zeros(GRID_SIZE)  # Grid for heatmap

def is_admin():
    """Check if script is running with admin privileges."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def scan_wifi_networks_netsh():
    """Scan Wi-Fi networks using netsh and return signal strengths."""
    try:
        output = subprocess.check_output("netsh wlan show networks mode=bssid", shell=True, text=True, stderr=subprocess.STDOUT)
        networks = []
        current_ssid = None
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("SSID"):
                current_ssid = line.split(":", 1)[1].strip()
            elif line.startswith("Signal") and current_ssid:
                try:
                    signal_percent = int(line.split(":", 1)[1].strip().replace("%", ""))
                    if signal_percent > 0:
                        rssi = signal_percent / 2 - 100  # Approximate RSSI (100% = -50 dBm, 0% = -100 dBm)
                        networks.append(rssi)
                        print(f"Network: SSID={current_ssid}, Signal={signal_percent}%, RSSI={rssi:.2f} dBm")
                    else:
                        print(f"Network: SSID={current_ssid}, Signal=0% (skipped)")
                except ValueError:
                    print(f"Network: SSID={current_ssid}, Invalid signal value (skipped)")
        print(f"Netsh found {len(networks)} networks with valid signal strength.")
        return networks
    except subprocess.CalledProcessError as e:
        print(f"Netsh command failed: {e.output}")
        return []
    except Exception as e:
        print(f"Netsh scan failed: {e}")
        return []

def calculate_moving_average(data, window):
    """Smooth signal data with a moving average."""
    return np.convolve(data, np.ones(window)/window, mode='valid')

def detect_motion_or_presence(signal_values, threshold, window):
    """Detect motion or presence based on signal variance."""
    if len(signal_values) < window:
        return False
    moving_avg = calculate_moving_average(signal_values, window)
    variance = np.var(signal_values[-window:])
    print(f"Signal Variance: {variance:.2f}")
    return variance > threshold

def estimate_distance(rssi, reference_rssi=-40, reference_distance=1):
    """Estimate distance using signal strength (coarse)."""
    try:
        distance = reference_distance * 10 ** ((reference_rssi - rssi) / (10 * PATH_LOSS_EXPONENT))
        return min(max(distance, 0.1), 20)
    except:
        return None

def simulate_grid_mapping(signal_values, grid_size, room_dim):
    """Map signal variance to a 2D grid."""
    grid = np.zeros(grid_size)
    if len(signal_values) < MOVING_AVG_WINDOW:
        return grid
    variance = np.var(signal_values[-MOVING_AVG_WINDOW:])
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            distance = np.sqrt((i - grid_size[0]/2)**2 + (j - grid_size[1]/2)**2)
            grid[i, j] = variance * np.exp(-distance / 3)
    return grid

def generate_heatmap(grid, filename="rssi_heatmap.png"):
    """Generate a 2D heatmap of signal variance."""
    plt.figure(figsize=(8, 8))
    sns.heatmap(grid, cmap="viridis", annot=True, fmt=".2f",
                xticklabels=np.linspace(0, ROOM_DIM[0], GRID_SIZE[0]),
                yticklabels=np.linspace(0, ROOM_DIM[1], GRID_SIZE[1]))
    plt.title("Wi-Fi Signal Disruption Heatmap (Signal Variance)")
    plt.xlabel("Room Width (meters)")
    plt.ylabel("Room Height (meters)")
    plt.savefig(filename)
    plt.close()
    print(f"Heatmap saved as {filename}")

def estimate_room_dimensions(signal_values):
    """Roughly estimate room dimensions."""
    if not signal_values:
        return None
    avg_signal = np.mean(signal_values)
    distance = estimate_distance(avg_signal)
    if distance:
        print(f"Estimated distance to access point: {distance:.2f} meters")
        return (2 * distance, 2 * distance)
    return None

def main():
    """Capture Wi-Fi signals, detect motion/presence, and visualize results."""
    if not is_admin():
        print("Error: This script requires administrative privileges. Run as administrator.")
        return
    
    print("Checking Wi-Fi status...")
    try:
        output = subprocess.check_output("netsh wlan show interfaces", shell=True, text=True, stderr=subprocess.STDOUT)
        if "State : connected" in output or "State : disconnected" in output:
            print("Wi-Fi adapter is enabled.")
        else:
            print("Wi-Fi adapter may be disabled. Enable Wi-Fi and try again.")
            return
    except subprocess.CalledProcessError as e:
        print(f"Failed to check Wi-Fi status: {e.output}")
        return
    except Exception as e:
        print(f"Failed to check Wi-Fi status: {e}")
        return
    
    print(f"Scanning Wi-Fi signals for {SCAN_DURATION} seconds...")
    start_time = time.time()
    retry_count = 0
    max_retries = 3
    while time.time() - start_time < SCAN_DURATION:
        networks = scan_wifi_networks_netsh()
        if networks:
            avg_signal = np.mean(networks)
            signal_data.append(avg_signal)
            timestamps.append(datetime.now())
            print(f"Average Signal Strength: {avg_signal:.2f} dBm")
            retry_count = 0  # Reset retries on success
        else:
            print("No networks detected in this scan.")
            retry_count += 1
            if retry_count >= max_retries:
                print("Multiple failed scans. Check Wi-Fi connectivity or move closer to access points.")
                break
        time.sleep(2)  # 2-second delay for reliable scanning
    
    if not signal_data:
        print("No signal data captured after scanning.")
        print("Ensure Wi-Fi is enabled and networks are visible. Run: netsh wlan show networks")
        print("Check: Wi-Fi on, adapter enabled, admin privileges, updated drivers.")
        return
    
    print("\nAnalyzing signal data...")
    motion_detected = detect_motion_or_presence(signal_data, SIGNAL_THRESHOLD, MOVING_AVG_WINDOW)
    if motion_detected:
        print("Motion or presence detected!")
    else:
        print("No significant motion or presence detected.")
    
    print("\nGenerating spatial mapping...")
    global grid_signal
    grid_signal = simulate_grid_mapping(signal_data, GRID_SIZE, ROOM_DIM)
    generate_heatmap(grid_signal)
    
    print("\nEstimating room dimensions (experimental)...")
    room_dims = estimate_room_dimensions(signal_data)
    if room_dims:
        print(f"Estimated room dimensions: {room_dims[0]:.2f}m x {room_dims[1]:.2f}m")
    else:
        print("Unable to estimate room dimensions.")
    
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, signal_data, label="Signal Strength")
    if len(signal_data) >= MOVING_AVG_WINDOW:
        moving_avg = calculate_moving_average(signal_data, MOVING_AVG_WINDOW)
        plt.plot(timestamps[MOVING_AVG_WINDOW-1:], moving_avg, label="Moving Average")
    plt.xlabel("Time")
    plt.ylabel("Signal Strength (dBm)")
    plt.title("Wi-Fi Signal Strength Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("signal_time_plot.png")
    print("Signal time plot saved as signal_time_plot.png")

if __name__ == "__main__":
    main()
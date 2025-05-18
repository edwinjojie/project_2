import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal
import time

# Configuration parameters
SAMPLE_RATE = 44100          # Audio sampling rate (Hz)
CHIRP_DURATION = 0.01        # Chirp duration (seconds)
CHIRP_FREQ_START = 18000     # Start frequency (Hz)
CHIRP_FREQ_END = 20000       # End frequency (Hz)
RECORD_DURATION = 0.1        # Record duration (seconds)
SPEED_OF_SOUND = 343         # Speed of sound (m/s)
MAX_DISTANCE = 10            # Max detectable distance (m)
MIN_DISTANCE = 0.3           # Min detectable distance (m)
CHIRP_INTERVAL = 0.5         # Interval between chirps (seconds)
MAX_CHIRPS = 50              # Total chirps to emit
UPDATE_INTERVAL = 10         # Update plot every 10 chirps
GRID_RESOLUTION = 20         # Voxel grid resolution (20x20x20)
LAPTOP_HEIGHT = 0.5          # Laptop height (m, e.g., on a table)

# Global variables
all_distances = []           # Store all detected distances
all_points_3d = []           # Store all 3D points (x, y, z)
room_dimensions = [0, 0, 0]  # [length, width, height]
laptop_pos = [0, 0, LAPTOP_HEIGHT]  # Laptop position (x, y, z)

def generate_chirp(duration, sample_rate, f_start, f_end):
    """Generate a linear chirp signal."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    chirp = scipy.signal.chirp(t, f_start, duration, f_end, method='linear')
    return chirp / np.max(np.abs(chirp))

def play_and_record(chirp, sample_rate, record_duration):
    """Play chirp and record reflections."""
    print("Playing chirp and recording reflections (keep room quiet)...")
    sd.play(chirp, sample_rate)
    recording = sd.rec(int(record_duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return recording.flatten()

def detect_reflections(chirp, recording, sample_rate):
    """Detect reflection times using cross-correlation."""
    corr = scipy.signal.correlate(recording, chirp, mode='full')
    lags = scipy.signal.correlation_lags(len(recording), len(chirp), mode='full')
    corr = corr / np.max(np.abs(corr))
    times = lags / sample_rate
    peaks, _ = scipy.signal.find_peaks(corr, height=0.05, distance=int(0.002 * sample_rate))
    reflection_times = times[peaks]
    valid_times = [t for t in reflection_times if 0 < t < (2 * MAX_DISTANCE / SPEED_OF_SOUND)]
    return valid_times

def time_to_distance(times):
    """Convert reflection times to distances."""
    return [(t * SPEED_OF_SOUND) / 2 for t in times]

def estimate_room_dimensions(distances, iteration):
    """Estimate room dimensions and laptop position using detected distances."""
    if not distances:
        return [5, 5, 5], [0, 0, LAPTOP_HEIGHT]  # Fallback
    
    # Cluster distances to find walls (largest consistent distances in opposite directions)
    distances = sorted(distances)
    max_dist = distances[-1]  # Potential diagonal
    
    # Assume laptop is not centered; find min/max distances in each direction
    x_pos, x_neg, y_pos, y_neg, z_pos = [], [], [], [], []
    for d in distances:
        if d < MIN_DISTANCE or d > MAX_DISTANCE:
            continue
        # Simulate directional scanning based on iteration
        angle = ((iteration % 360) + (distances.index(d) * 45)) * np.pi / 180
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        z = (d / max_dist) * (max_dist / 2) if max_dist > 0 else d / 2
        if x > 0:
            x_pos.append(abs(x))
        else:
            x_neg.append(abs(x))
        if y > 0:
            y_pos.append(abs(y))
        else:
            y_neg.append(abs(y))
        if z > LAPTOP_HEIGHT:
            z_pos.append(z - LAPTOP_HEIGHT)
    
    # Estimate room dimensions (double the distance to account for walls on both sides)
    length = (max(x_pos, default=2) + max(x_neg, default=2)) if x_pos or x_neg else 4
    width = (max(y_pos, default=2) + max(y_neg, default=2)) if y_pos or y_neg else 4
    height = max(z_pos, default=2) + LAPTOP_HEIGHT if z_pos else 2.5
    
    # Ensure dimensions are reasonable (cap at max distance)
    length = min(length, max_dist)
    width = min(width, max_dist)
    height = min(height, max_dist)
    
    # Estimate laptop position (offset based on min distances)
    laptop_x = max(x_neg, default=length/2) - max(x_pos, default=length/2)
    laptop_y = max(y_neg, default=width/2) - max(y_pos, default=width/2)
    
    return [length, width, height], [laptop_x, laptop_y, LAPTOP_HEIGHT]

def map_to_3d(distances, iteration, laptop_pos, room_dim):
    """Map distances to 3D points relative to laptop position."""
    points = []
    max_dist = max(distances) if distances else 5
    for i, d in enumerate(distances):
        if MIN_DISTANCE <= d <= MAX_DISTANCE:
            # Simulate directional scanning
            angle = ((iteration % 360) + (i * 45)) * np.pi / 180
            x = laptop_pos[0] + d * np.cos(angle)
            y = laptop_pos[1] + d * np.sin(angle)
            z = laptop_pos[2] + (d / max_dist) * (room_dim[2] - laptop_pos[2])
            # Ensure points are within room bounds
            if (abs(x) <= room_dim[0]/2 and
                abs(y) <= room_dim[1]/2 and
                0 <= z <= room_dim[2]):
                points.append((x, y, z))
    return points

def create_voxel_grid(points, resolution, room_dim):
    """Create a 3D voxel grid for density visualization."""
    if not points:
        return np.zeros((resolution, resolution, resolution))
    x, y, z = zip(*points)
    grid = np.zeros((resolution, resolution, resolution))
    for i in range(len(x)):
        # Scale coordinates to grid indices
        xi = int(((x[i] + room_dim[0]/2) / room_dim[0]) * (resolution - 1))
        yi = int(((y[i] + room_dim[1]/2) / room_dim[1]) * (resolution - 1))
        zi = int((z[i] / room_dim[2]) * (resolution - 1))
        if 0 <= xi < resolution and 0 <= yi < resolution and 0 <= zi < resolution:
            grid[xi, yi, zi] += 1
    return grid

def plot_3d_map(points, iteration, room_dim, laptop_pos, filename_prefix="room_3d_map"):
    """Generate a 3D voxel plot of the room."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot room boundaries
    x = [-room_dim[0]/2, room_dim[0]/2]
    y = [-room_dim[1]/2, room_dim[1]/2]
    z = [0, room_dim[2]]
    ax.plot3D(x, [y[0], y[0]], [z[0], z[0]], 'b-', alpha=0.2)
    ax.plot3D(x, [y[1], y[1]], [z[0], z[0]], 'b-', alpha=0.2)
    ax.plot3D(x, [y[0], y[0]], [z[1], z[1]], 'b-', alpha=0.2)
    ax.plot3D(x, [y[1], y[1]], [z[1], z[1]], 'b-', alpha=0.2)
    ax.plot3D([x[0], x[0]], y, [z[0], z[0]], 'b-', alpha=0.2)
    ax.plot3D([x[1], x[1]], y, [z[0], z[0]], 'b-', alpha=0.2)
    ax.plot3D([x[0], x[0]], y, [z[1], z[1]], 'b-', alpha=0.2)
    ax.plot3D([x[1], x[1]], y, [z[1], z[1]], 'b-', alpha=0.2)
    ax.plot3D([x[0], x[0]], [y[0], y[0]], z, 'b-', alpha=0.2)
    ax.plot3D([x[0], x[0]], [y[1], y[1]], z, 'b-', alpha=0.2)
    ax.plot3D([x[1], x[1]], [y[0], y[0]], z, 'b-', alpha=0.2)
    ax.plot3D([x[1], x[1]], [y[1], y[1]], z, 'b-', alpha=0.2)
    # Plot laptop position
    ax.scatter([laptop_pos[0]], [laptop_pos[1]], [laptop_pos[2]], c='r', marker='o', s=100, label='Laptop')
    # Create voxel grid
    voxel_grid = create_voxel_grid(points, GRID_RESOLUTION, room_dim)
    nonzero = voxel_grid > 0
    if np.any(nonzero):
        x, y, z = np.indices(voxel_grid.shape)
        x = (x[nonzero] / (GRID_RESOLUTION - 1)) * room_dim[0] - room_dim[0]/2
        y = (y[nonzero] / (GRID_RESOLUTION - 1)) * room_dim[1] - room_dim[1]/2
        z = (z[nonzero] / (GRID_RESOLUTION - 1)) * room_dim[2]
        intensities = voxel_grid[nonzero]
        ax.scatter(x, y, z, c=intensities, cmap='viridis', s=50, alpha=0.6, label='Density')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_xlim(-room_dim[0]/2, room_dim[0]/2)
    ax.set_ylim(-room_dim[1]/2, room_dim[1]/2)
    ax.set_zlim(0, room_dim[2])
    plt.title(f"3D Room Map (Chirp {iteration})")
    plt.legend()
    filename = f"{filename_prefix}_{iteration}.png"
    plt.savefig(filename)
    plt.close()
    print(f"3D map saved as {filename}")

def plot_reflections(chirp, recording, reflection_times, sample_rate, iteration, filename_prefix="reflection_time_plot"):
    """Plot emitted and recorded signals with reflection times."""
    t = np.linspace(0, len(recording)/sample_rate, len(recording))
    plt.figure(figsize=(10, 6))
    plt.plot(t, recording, label="Recorded Signal")
    plt.plot(t[:len(chirp)], chirp, label="Emitted Chirp", alpha=0.5)
    for rt in reflection_times:
        plt.axvline(x=rt, color='r', linestyle='--', alpha=0.5, label="Reflection" if rt == reflection_times[0] else "")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(f"Chirp and Reflection Times (Chirp {iteration})")
    plt.legend()
    plt.grid(True)
    filename = f"{filename_prefix}_{iteration}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Reflection time plot saved as {filename}")

def main():
    """Continuously emit chirps, capture reflections, and map room in 3D."""
    print("Checking audio devices...")
    try:
        sd.check_output_settings(device=None, channels=1, dtype='float32', samplerate=SAMPLE_RATE)
        sd.check_input_settings(device=None, channels=1, dtype='float32', samplerate=SAMPLE_RATE)
    except Exception as e:
        print(f"Audio device error: {e}")
        print("Ensure speaker and microphone are enabled.")
        return
    
    global all_distances, all_points_3d, room_dimensions, laptop_pos
    
    for chirp_num in range(1, MAX_CHIRPS + 1):
        print(f"\nChirp {chirp_num}/{MAX_CHIRPS}")
        # Generate chirp
        f_start = CHIRP_FREQ_START + (chirp_num % 5) * 100
        f_end = CHIRP_FREQ_END + (chirp_num % 5) * 100
        chirp = generate_chirp(CHIRP_DURATION, SAMPLE_RATE, f_start, f_end)
        
        # Play chirp and record
        recording = play_and_record(chirp, SAMPLE_RATE, RECORD_DURATION)
        
        # Detect reflections
        reflection_times = detect_reflections(chirp, recording, SAMPLE_RATE)
        if not reflection_times:
            print("No reflections detected. Ensure room is quiet and place objects nearby.")
            continue
        
        # Convert times to distances
        distances = time_to_distance(reflection_times)
        valid_distances = [d for d in distances if MIN_DISTANCE <= d <= MAX_DISTANCE]
        if not valid_distances:
            print("No valid distances detected.")
            continue
        
        print("Detected distances:")
        for i, d in enumerate(valid_distances):
            print(f"Surface {i+1}: {d:.2f} meters")
        all_distances.extend(valid_distances)
        
        # Estimate room dimensions and laptop position
        room_dimensions, laptop_pos = estimate_room_dimensions(all_distances, chirp_num)
        print(f"Estimated room: {room_dimensions[0]:.1f}m x {room_dimensions[1]:.1f}m x {room_dimensions[2]:.1f}m")
        print(f"Laptop position: ({laptop_pos[0]:.1f}, {laptop_pos[1]:.1f}, {laptop_pos[2]:.1f})")
        
        # Map distances to 3D points
        points_3d = map_to_3d(valid_distances, chirp_num, laptop_pos, room_dimensions)
        all_points_3d.extend(points_3d)
        
        # Generate visualizations every UPDATE_INTERVAL chirps
        if chirp_num % UPDATE_INTERVAL == 0 or chirp_num == MAX_CHIRPS:
            plot_3d_map(all_points_3d, chirp_num, room_dimensions, laptop_pos)
            plot_reflections(chirp, recording, reflection_times, SAMPLE_RATE, chirp_num)
        
        # Delay between chirps
        time.sleep(CHIRP_INTERVAL)
    
    if not all_distances:
        print("No reflections detected during the session. Check audio setup and room conditions.")

if __name__ == "__main__":
    main()
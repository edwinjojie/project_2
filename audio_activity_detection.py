import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal
import time
from sklearn.cluster import DBSCAN

# Configuration parameters
SAMPLE_RATE = 44100          # Audio sampling rate (Hz)
CHIRP_DURATION = 0.02        # Chirp duration (seconds)
CHIRP_FREQ_START = 8000      # Start frequency (Hz)
CHIRP_FREQ_END = 12000       # End frequency (Hz)
RECORD_DURATION = 0.15       # Record duration (seconds)
SPEED_OF_SOUND = 343         # Speed of sound (m/s)
MAX_DISTANCE = 5             # Max detectable distance (m)
MIN_DISTANCE = 0.3           # Min detectable distance (m)
MOVE_DISTANCE = 0.1          # Distance to move laptop (meters, 10 cm)
CHIRP_PAIR_INTERVAL = 2.0    # Time between chirps in a pair (seconds)
MAX_PAIRS = 10               # Number of chirp pairs (20 chirps total)
UPDATE_INTERVAL = 5          # Update plot every 5 pairs
LAPTOP_HEIGHT = 0.5          # Laptop height (m)

# Global variables
all_distances = []
all_amplitudes = []
all_points_3d = []
room_dimensions = [0, 0, 0]
laptop_pos = [0, 0, LAPTOP_HEIGHT]

def generate_chirp(duration, sample_rate, f_start, f_end):
    """Generate a chirp signal."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    chirp = scipy.signal.chirp(t, f_start, duration, f_end, method='linear')
    return chirp / np.max(np.abs(chirp))

def play_and_record(chirp, sample_rate, record_duration, pair_num, position):
    """Play chirp, record reflections, and save raw recording for debugging."""
    print("Playing chirp and recording reflections (keep room quiet)...")
    sd.play(chirp, sample_rate)
    recording = sd.rec(int(record_duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    recording = recording.flatten()
    # Debug: Save raw recording
    t = np.linspace(0, record_duration, len(recording))
    plt.figure(figsize=(10, 6))
    plt.plot(t, recording, label="Raw Recording")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(f"Raw Recording (Pair {pair_num}, Position {position})")
    plt.legend()
    plt.grid(True)
    filename = f"raw_recording_{pair_num}_{position}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Raw recording plot saved as {filename}")
    return recording

def detect_reflections(chirp, recording, sample_rate, pair_num, position):
    """Detect reflection times and amplitudes, with debug output."""
    corr = scipy.signal.correlate(recording, chirp, mode='full')
    lags = scipy.signal.correlation_lags(len(recording), len(chirp), mode='full')
    times = lags / sample_rate
    if np.max(np.abs(corr)) == 0:
        print("Cross-correlation is zero. No signal detected.")
        return [], []
    corr = corr / np.max(np.abs(corr))
    # Debug: Save cross-correlation plot
    plt.figure(figsize=(10, 6))
    plt.plot(times, corr, label="Cross-Correlation")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Correlation")
    plt.title(f"Cross-Correlation (Pair {pair_num}, Position {position})")
    plt.legend()
    plt.grid(True)
    filename = f"cross_correlation_{pair_num}_{position}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Cross-correlation plot saved as {filename}")
    
    peaks, properties = scipy.signal.find_peaks(corr, height=0.01, distance=int(0.001 * sample_rate))
    reflection_times = times[peaks]
    amplitudes = properties['peak_heights']
    valid_indices = [i for i, t in enumerate(reflection_times) if 0 < t < (2 * MAX_DISTANCE / SPEED_OF_SOUND)]
    return [reflection_times[i] for i in valid_indices], [amplitudes[i] for i in valid_indices]

def time_to_distance(times):
    """Convert reflection times to distances."""
    return [(t * SPEED_OF_SOUND) / 2 for t in times]

def cluster_reflections(distances, amplitudes):
    """Cluster reflections to identify surfaces and objects."""
    if not distances:
        return [], [], []
    
    # Cluster based on distances
    X = np.array(distances).reshape(-1, 1)
    clustering = DBSCAN(eps=0.1, min_samples=2).fit(X)
    labels = clustering.labels_
    
    # Group distances and amplitudes by cluster
    clusters = {}
    cluster_amplitudes = {}
    for label, dist, amp in zip(labels, distances, amplitudes):
        if label == -1:  # Noise/outliers treated as individual objects
            label = f"outlier_{dist}"
        if label not in clusters:
            clusters[label] = []
            cluster_amplitudes[label] = []
        clusters[label].append(dist)
        cluster_amplitudes[label].append(amp)
    
    # Average distances and amplitudes per cluster
    cluster_distances = []
    cluster_avg_amplitudes = []
    cluster_sizes = []
    for label in clusters:
        avg_dist = np.mean(clusters[label])
        avg_amp = np.mean(cluster_amplitudes[label])
        cluster_distances.append(avg_dist)
        cluster_avg_amplitudes.append(avg_amp)
        cluster_sizes.append(len(clusters[label]))
    
    return cluster_distances, cluster_avg_amplitudes, cluster_sizes

def calculate_tdoa(times_a, times_b, amplitudes_a, amplitudes_b):
    """Calculate TDOA to estimate azimuth angles after clustering."""
    if not times_a or not times_b:
        return [], [], [], []
    
    distances_a = time_to_distance(times_a)
    distances_b = time_to_distance(times_b)
    
    # Cluster reflections at both positions
    cluster_distances_a, cluster_amplitudes_a, _ = cluster_reflections(distances_a, amplitudes_a)
    cluster_distances_b, cluster_amplitudes_b, _ = cluster_reflections(distances_b, amplitudes_b)
    
    # Match clusters between positions
    matched_pairs = []
    for i, (da, aa) in enumerate(zip(cluster_distances_a, cluster_amplitudes_a)):
        for j, (db, ab) in enumerate(zip(cluster_distances_b, cluster_amplitudes_b)):
            if abs(da - db) < 0.1 and abs(aa - ab) < 0.2:
                matched_pairs.append((i, j, da, aa))
                break
    
    # Calculate TDOA for matched clusters
    tdoas = []
    matched_distances = []
    matched_amplitudes = []
    for i, j, d, a in matched_pairs:
        # Find representative ToA for each cluster (e.g., median)
        cluster_times_a = [t for t, d in zip(times_a, distances_a) if abs(d - cluster_distances_a[i]) < 0.1]
        cluster_times_b = [t for t, d in zip(times_b, distances_b) if abs(d - cluster_distances_b[j]) < 0.1]
        if cluster_times_a and cluster_times_b:
            tdoa = np.median(cluster_times_b) - np.median(cluster_times_a)
            tdoas.append(tdoa)
            matched_distances.append(d)
            matched_amplitudes.append(a)
    
    # Calculate azimuth angles
    azimuths = []
    for tdoa in tdoas:
        sin_theta = (tdoa * SPEED_OF_SOUND) / MOVE_DISTANCE
        sin_theta = np.clip(sin_theta, -1, 1)
        theta = np.arcsin(sin_theta) * 180 / np.pi
        azimuths.append(theta)
    
    return matched_distances, matched_amplitudes, tdoas, azimuths

def infer_elevation(amplitudes):
    """Infer elevation angles using amplitude (simplified)."""
    amplitudes = np.array(amplitudes) / np.max(amplitudes)
    elevations = [30 * (1 - amp) for amp in amplitudes]  # 0° to 30°
    return elevations

def estimate_room_dimensions(distances, angles):
    """Estimate room dimensions and laptop position."""
    if not distances:
        return [5, 5, 5], [0, 0, LAPTOP_HEIGHT]
    
    x_pos, x_neg, y_pos, y_neg, z_pos = [], [], [], [], []
    for d, (azimuth, elevation) in zip(distances, angles):
        if d < MIN_DISTANCE or d > MAX_DISTANCE:
            continue
        azimuth_rad = azimuth * np.pi / 180
        elevation_rad = elevation * np.pi / 180
        x = d * np.cos(azimuth_rad) * np.cos(elevation_rad)
        y = d * np.sin(azimuth_rad) * np.cos(elevation_rad)
        z = d * np.sin(elevation_rad) + LAPTOP_HEIGHT
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
    
    length = (max(x_pos, default=2) + max(x_neg, default=2)) if x_pos or x_neg else 4
    width = (max(y_pos, default=2) + max(y_neg, default=2)) if y_pos or y_neg else 4
    height = max(z_pos, default=2) + LAPTOP_HEIGHT if z_pos else 2.5
    
    max_dist = max(distances) if distances else 5
    length = min(length, max_dist)
    width = min(width, max_dist)
    height = min(height, max_dist)
    
    laptop_x = max(x_neg, default=length/2) - max(x_pos, default=length/2)
    laptop_y = max(y_neg, default=width/2) - max(y_pos, default=width/2)
    
    return [length, width, height], [laptop_x, laptop_y, LAPTOP_HEIGHT]

def map_to_3d(distances, angles, cluster_sizes, laptop_pos, room_dim):
    """Map distances to 3D points, distinguishing walls and objects."""
    walls = []  # For large clusters (walls, floor, ceiling)
    objects = []  # For small clusters (furniture)
    for d, (azimuth, elevation), size in zip(distances, angles, cluster_sizes):
        if MIN_DISTANCE <= d <= MAX_DISTANCE:
            azimuth_rad = azimuth * np.pi / 180
            elevation_rad = elevation * np.pi / 180
            x = laptop_pos[0] + d * np.cos(azimuth_rad) * np.cos(elevation_rad)
            y = laptop_pos[1] + d * np.sin(azimuth_rad) * np.cos(elevation_rad)
            z = laptop_pos[2] + d * np.sin(elevation_rad)
            if (abs(x) <= room_dim[0]/2 and
                abs(y) <= room_dim[1]/2 and
                0 <= z <= room_dim[2]):
                if size >= 5:  # Large clusters are walls/floor/ceiling
                    walls.append((x, y, z, d))
                else:  # Small clusters are objects
                    objects.append((x, y, z))
    return walls, objects

def plot_wireframe(walls, objects, room_dim, laptop_pos, pair_num):
    """Generate a 3D wireframe representation of the room and objects."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw room (four walls, floor, ceiling) as a wireframe box
    x = [-room_dim[0]/2, room_dim[0]/2]
    y = [-room_dim[1]/2, room_dim[1]/2]
    z = [0, room_dim[2]]
    # Floor
    ax.plot3D(x, [y[0], y[0]], [z[0], z[0]], 'b-')
    ax.plot3D(x, [y[1], y[1]], [z[0], z[0]], 'b-')
    ax.plot3D([x[0], x[0]], y, [z[0], z[0]], 'b-')
    ax.plot3D([x[1], x[1]], y, [z[0], z[0]], 'b-')
    # Ceiling
    ax.plot3D(x, [y[0], y[0]], [z[1], z[1]], 'b-')
    ax.plot3D(x, [y[1], y[1]], [z[1], z[1]], 'b-')
    ax.plot3D([x[0], x[0]], y, [z[1], z[1]], 'b-')
    ax.plot3D([x[1], x[1]], y, [z[1], z[1]], 'b-')
    # Vertical edges
    ax.plot3D([x[0], x[0]], [y[0], y[0]], z, 'b-')
    ax.plot3D([x[0], x[0]], [y[1], y[1]], z, 'b-')
    ax.plot3D([x[1], x[1]], [y[0], y[0]], z, 'b-')
    ax.plot3D([x[1], x[1]], [y[1], y[1]], z, 'b-')
    
    # Plot detected walls as planes (simplified as lines along the room edges)
    for x, y, z, d in walls:
        # Determine which surface this is based on position
        if abs(z - LAPTOP_HEIGHT) < 0.1:  # Near laptop height, likely a wall
            if abs(x - (-room_dim[0]/2)) < 0.5:  # Left wall
                ax.plot3D([-room_dim[0]/2, -room_dim[0]/2], y, [0, room_dim[2]], 'g-', linewidth=2)
            elif abs(x - (room_dim[0]/2)) < 0.5:  # Right wall
                ax.plot3D([room_dim[0]/2, room_dim[0]/2], y, [0, room_dim[2]], 'g-', linewidth=2)
            elif abs(y - (-room_dim[1]/2)) < 0.5:  # Front wall
                ax.plot3D(x, [-room_dim[1]/2, -room_dim[1]/2], [0, room_dim[2]], 'g-', linewidth=2)
            elif abs(y - (room_dim[1]/2)) < 0.5:  # Back wall
                ax.plot3D(x, [room_dim[1]/2, room_dim[1]/2], [0, room_dim[2]], 'g-', linewidth=2)
        elif abs(z - 0) < 0.5:  # Floor
            ax.plot3D(x, y, [0, 0], 'g-', linewidth=2)
        elif abs(z - room_dim[2]) < 0.5:  # Ceiling
            ax.plot3D(x, y, [room_dim[2], room_dim[2]], 'g-', linewidth=2)
    
    # Plot objects as small wireframe boxes
    for x, y, z in objects:
        # Simplified: Draw a small cube (0.2m sides) to represent an object
        r = 0.1  # Half-side length
        ax.plot3D([x-r, x+r], [y-r, y-r], [z-r, z-r], 'r-')
        ax.plot3D([x-r, x+r], [y+r, y+r], [z-r, z-r], 'r-')
        ax.plot3D([x-r, x-r], [y-r, y+r], [z-r, z-r], 'r-')
        ax.plot3D([x+r, x+r], [y-r, y+r], [z-r, z-r], 'r-')
        ax.plot3D([x-r, x+r], [y-r, y-r], [z+r, z+r], 'r-')
        ax.plot3D([x-r, x+r], [y+r, y+r], [z+r, z+r], 'r-')
        ax.plot3D([x-r, x-r], [y-r, y+r], [z+r, z+r], 'r-')
        ax.plot3D([x+r, x+r], [y-r, y+r], [z+r, z+r], 'r-')
        ax.plot3D([x-r, x-r], [y-r, y-r], [z-r, z+r], 'r-')
        ax.plot3D([x+r, x+r], [y-r, y-r], [z-r, z+r], 'r-')
        ax.plot3D([x-r, x-r], [y+r, y+r], [z-r, z+r], 'r-')
        ax.plot3D([x+r, x+r], [y+r, y+r], [z-r, z+r], 'r-')
    
    # Plot laptop position
    ax.scatter([laptop_pos[0]], [laptop_pos[1]], [laptop_pos[2]], c='r', marker='o', s=100, label='Laptop')
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_xlim(-room_dim[0]/2, room_dim[0]/2)
    ax.set_ylim(-room_dim[1]/2, room_dim[1]/2)
    ax.set_zlim(0, room_dim[2])
    plt.title(f"Wireframe Room Map (Pair {pair_num})")
    plt.legend()
    filename = f"wireframe_map_{pair_num}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Wireframe map saved as {filename}")

def plot_reflections(chirp, recording, reflection_times, amplitudes, sample_rate, pair_num, position):
    """Plot emitted and recorded signals with reflection times."""
    t = np.linspace(0, len(recording)/sample_rate, len(recording))
    plt.figure(figsize=(10, 6))
    plt.plot(t, recording, label="Recorded Signal")
    plt.plot(t[:len(chirp)], chirp, label="Emitted Chirp", alpha=0.5)
    for rt, amp in zip(reflection_times, amplitudes):
        plt.axvline(x=rt, color='r', linestyle='--', alpha=amp, label="Reflection" if rt == reflection_times[0] else "")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(f"Chirp and Reflection Times (Pair {pair_num}, Position {position})")
    plt.legend()
    plt.grid(True)
    filename = f"reflection_time_plot_{pair_num}_{position}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Reflection time plot saved as {filename}")

def main():
    """Emit chirp pairs, prompt user movement, and map room in 3D."""
    print("Checking audio devices...")
    try:
        sd.check_output_settings(device=None, channels=1, dtype='float32', samplerate=SAMPLE_RATE)
        sd.check_input_settings(device=None, channels=1, dtype='float32', samplerate=SAMPLE_RATE)
    except Exception as e:
        print(f"Audio device error: {e}")
        print("Ensure speaker and microphone are enabled.")
        return
    
    global all_distances, all_amplitudes, all_points_3d, room_dimensions, laptop_pos
    
    print("\nInstructions:")
    print("For each pair:")
    print("1. Position A: Keep the laptop still and press Enter to emit the first chirp.")
    print("2. Position B: Move the laptop 10 cm to the right, then press Enter to emit the second chirp.")
    print("Repeat for 10 pairs (20 chirps total). Keep movements consistent.")
    
    for pair_num in range(1, MAX_PAIRS + 1):
        print(f"\nChirp Pair {pair_num}/{MAX_PAIRS}")
        
        # Position A
        input("Position A: Keep laptop still and press Enter to emit chirp...")
        chirp_a = generate_chirp(CHIRP_DURATION, SAMPLE_RATE, CHIRP_FREQ_START, CHIRP_FREQ_END)
        recording_a = play_and_record(chirp_a, SAMPLE_RATE, RECORD_DURATION, pair_num, "A")
        times_a, amplitudes_a = detect_reflections(chirp_a, recording_a, SAMPLE_RATE, pair_num, "A")
        if not times_a:
            print("No reflections detected at Position A. Skipping pair.")
            continue
        
        # Prompt user to move
        input(f"Position B: Move laptop 10 cm to the right, then press Enter to emit chirp...")
        
        # Position B
        chirp_b = generate_chirp(CHIRP_DURATION, SAMPLE_RATE, CHIRP_FREQ_START, CHIRP_FREQ_END)
        recording_b = play_and_record(chirp_b, SAMPLE_RATE, RECORD_DURATION, pair_num, "B")
        times_b, amplitudes_b = detect_reflections(chirp_b, recording_b, SAMPLE_RATE, pair_num, "B")
        if not times_b:
            print("No reflections detected at Position B. Skipping pair.")
            continue
        
        # Calculate TDOA and angles
        distances, amplitudes, tdoas, azimuths = calculate_tdoa(times_a, times_b, amplitudes_a, amplitudes_b)
        if not distances:
            print("No matched reflections between positions. Skipping pair.")
            continue
        
        # Cluster sizes for distinguishing walls vs. objects
        _, _, cluster_sizes_a = cluster_reflections(time_to_distance(times_a), amplitudes_a)
        _, _, cluster_sizes_b = cluster_reflections(time_to_distance(times_b), amplitudes_b)
        cluster_sizes = cluster_sizes_a[:len(distances)]  # Simplified matching
        
        elevations = infer_elevation(amplitudes)
        
        print("Detected distances, TDOA, and angles:")
        for i, (d, tdoa, az, el) in enumerate(zip(distances, tdoas, azimuths, elevations)):
            print(f"Surface/Object {i+1}: Distance: {d:.2f}m, TDOA: {tdoa:.6f}s, Azimuth: {az:.1f}°, Elevation: {el:.1f}°")
        all_distances.extend(distances)
        all_amplitudes.extend(amplitudes)
        
        angles = list(zip(azimuths, elevations))
        
        # Estimate room dimensions
        room_dimensions, laptop_pos = estimate_room_dimensions(distances, angles)
        print(f"Estimated room: {room_dimensions[0]:.1f}m x {room_dimensions[1]:.1f}m x {room_dimensions[2]:.1f}m")
        print(f"Laptop position: ({laptop_pos[0]:.1f}, {laptop_pos[1]:.1f}, {laptop_pos[2]:.1f})")
        
        # Map to 3D
        walls, objects = map_to_3d(distances, angles, cluster_sizes, laptop_pos, room_dimensions)
        all_points_3d.extend([(x, y, z) for x, y, z, _ in walls] + objects)
        
        # Visualize as wireframe
        if pair_num % UPDATE_INTERVAL == 0 or pair_num == MAX_PAIRS:
            plot_wireframe(walls, objects, room_dimensions, laptop_pos, pair_num)
            plot_reflections(chirp_a, recording_a, times_a, amplitudes_a, SAMPLE_RATE, pair_num, "A")
            plot_reflections(chirp_b, recording_b, times_b, amplitudes_b, SAMPLE_RATE, pair_num, "B")
        
        time.sleep(CHIRP_PAIR_INTERVAL)
    
    if not all_distances:
        print("No reflections detected during the session. Check audio setup and room conditions.")

if __name__ == "__main__":
    main()
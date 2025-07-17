import os
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# 🔧 Settings
log_dirs = {
    "Nodes 4": r"C:\Users\rkozl\Documents\PythonProjects\PPO_PSRP\tppo_tensorboard\radomized_GNN_TRPO_batch-4048_nodes-4_veh-2_1",
    "Nodes 6": r"C:\Users\rkozl\Documents\PythonProjects\PPO_PSRP\tppo_tensorboard\radomized_GNN_TRPO_batch-4048_nodes-6_veh-2_3",
    "Nodes 8": r"C:\Users\rkozl\Documents\PythonProjects\PPO_PSRP\tppo_tensorboard\radomized_GNN_TRPO_batch-4048_nodes-8_veh-4_1",
}
metrics_to_plot = [
    "rollout/ep_rew_mean",
    "episode_info/dry_runs",
    "episode_info/total_travel_distance",
    "train/explained_variance"
]
output_dir = "tensorboard_metric_plots"
smoothing_weight = 0.9
max_step = 1_800_000
outlier_threshold = 3  # std deviations

os.makedirs(output_dir, exist_ok=True)

def smooth(scalars, weight):
    if len(scalars) == 0:
        return np.array([])
    smoothed = [scalars[0]]
    for i in range(1, len(scalars)):
        smoothed.append(weight * smoothed[-1] + (1 - weight) * scalars[i])
    return np.array(smoothed)

def remove_outliers(steps, values, window=10, threshold=3):
    if len(values) < window:
        return steps, values
    mask = []
    for i in range(len(values)):
        start = max(0, i - window)
        window_values = values[start:i+1]
        mean = np.mean(window_values)
        std = np.std(window_values)
        if std == 0 or abs(values[i] - mean) <= threshold * std:
            mask.append(True)
        else:
            mask.append(False)
    return steps[mask], values[mask]

for metric in metrics_to_plot:
    plt.figure(figsize=(10, 6))
    for run_name, log_path in log_dirs.items():
        if not os.path.exists(log_path):
            print(f"[!] Warning: Log path '{log_path}' does not exist.")
            continue

        try:
            ea = event_accumulator.EventAccumulator(log_path)
            ea.Reload()

            if metric not in ea.Tags()["scalars"]:
                print(f"[!] Warning: Metric '{metric}' not found in '{run_name}'")
                continue

            events = ea.Scalars(metric)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])

            # Filter by max step
            mask = steps <= max_step
            steps = steps[mask]
            values = values[mask]

            # Remove outliers
            steps, values = remove_outliers(steps, values, window=10, threshold=outlier_threshold)

            # Smooth and calculate std
            smoothed_values = smooth(values, smoothing_weight)
            window = 10
            std_dev = np.array([
                np.std(values[max(0, i - window):i + 1]) for i in range(len(values))
            ])

            plt.plot(steps, smoothed_values, label=run_name)
            plt.fill_between(steps, smoothed_values - std_dev, smoothed_values + std_dev, alpha=0.2)

        except Exception as e:
            print(f"[!] Error processing '{run_name}': {e}")

    short_metric_name = metric.split("/")[-1]
    plt.title(f"{short_metric_name.replace('_', ' ').capitalize()} vs Steps")
    plt.xlabel("Training step")
    plt.ylabel(short_metric_name.replace('_', ' ').capitalize())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{short_metric_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[+] Saved: {save_path}")

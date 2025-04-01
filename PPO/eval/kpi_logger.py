import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def add_vertical_lines(ax, new_day_series, alpha=0.3):
    for idx, day_end in enumerate(new_day_series):
        if day_end:
            ax.axvline(x=idx, color='red', linestyle='--', alpha=alpha)


def log_kpi_metrics(df, capacities_list, args):
    fig, axes = plt.subplots(4, 1, figsize=(14*1.5, 12*1.5))
    axes = axes.flatten()

    title_fontsize = 30
    label_fontsize = 26
    legend_fontsize = 20
    tick_fontsize = 26

    axes[0].plot(df.iloc[:, 0], label="Capacity reward")
    axes[0].set_title("Capacity reward", fontsize=title_fontsize)
    axes[0].grid()
    axes[0].tick_params(axis='x', labelsize=tick_fontsize)
    axes[0].tick_params(axis='y', labelsize=tick_fontsize)
    axes[0].set_ylim(0.5, 1.1)
    add_vertical_lines(axes[0], df['new_day'])
    axes[0].legend(loc="upper left", fontsize=legend_fontsize)

    axes[1].plot(df.iloc[:, 1], label="Distance reward")
    axes[1].set_title("Distance reward", fontsize=title_fontsize)
    axes[1].grid()
    axes[1].tick_params(axis='x', labelsize=tick_fontsize)
    axes[1].tick_params(axis='y', labelsize=tick_fontsize)
    axes[1].set_ylim(-0.5, 0.1)
    add_vertical_lines(axes[1], df['new_day'])
    axes[1].legend(loc="upper left", fontsize=legend_fontsize)

    for i in range(2, 7):
        axes[2].plot(df.iloc[:, i], label=df.columns[i].replace("_penalties", ""))
    # axes[2].plot(df['dry_runs_penalty'], label='dry_runs')  # Добавлено
    axes[2].set_title("Penalties", fontsize=title_fontsize)
    axes[2].grid()
    axes[2].tick_params(axis='x', labelsize=tick_fontsize)
    axes[2].tick_params(axis='y', labelsize=tick_fontsize)
    add_vertical_lines(axes[2], df['new_day'])
    axes[2].legend(loc="upper left", fontsize=legend_fontsize)

    pd.DataFrame(capacities_list, columns=[f'Station {i+1}' for i in range(args.n - 1)]).plot(ax=axes[3])
    axes[3].set_title("Station capacity", fontsize=title_fontsize)
    axes[3].grid()
    axes[3].set_ylabel("Liters", fontsize=label_fontsize)
    axes[3].set_xlabel("Steps", fontsize=label_fontsize)
    axes[3].tick_params(axis='x', labelsize=tick_fontsize)
    axes[3].tick_params(axis='y', labelsize=tick_fontsize)
    add_vertical_lines(axes[3], df['new_day'])
    axes[3].legend(loc="upper left", fontsize=legend_fontsize)  # Включаем легенду с номерами станций

    plt.tight_layout()
    output_dir = f"results/{args.n}_{args.n_steps}_{args.veh}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/rewards_{args.n}_{args.n_steps}_{args.veh}.png")
    df.to_csv(f"{output_dir}/results.csv", index=False)
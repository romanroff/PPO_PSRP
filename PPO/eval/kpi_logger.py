import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def add_vertical_lines(ax, new_day_series, alpha=0.3):
    for idx, day_end in enumerate(new_day_series):
        if day_end:
            ax.axvline(x=idx, color='red', linestyle='--', alpha=alpha)

def log_kpi_metrics(df, capacities_list, args):
    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(len(df.columns) - 1, 1, height_ratios=[1] * (len(df.columns) - 2) + [2])
    for i, column in enumerate(df.columns[:-2]):
        ax = fig.add_subplot(gs[i])
        ax.plot(df[column], label=column)
        ax.set_title(column)
        ax.grid()
        add_vertical_lines(ax, df['new_day'])
        ax.legend()

    last_ax = fig.add_subplot(gs[-1])
    pd.DataFrame(capacities_list, columns=[f'{i}' for i in range(args.n)]).plot(ax=last_ax)
    last_ax.set_title('Capacities Plot')
    last_ax.grid()
    add_vertical_lines(last_ax, df['new_day'])
    plt.tight_layout()

    # Создание папки results, если она не существует
    if not os.path.exists("results"):
        os.makedirs("results")

    # Сохранение графика
    plt.savefig(f"results/rewards_{args.n}_{args.n_steps}_{args.veh}.png")
    df.to_csv('results/results.csv', index=False)


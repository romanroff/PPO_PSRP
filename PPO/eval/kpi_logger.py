import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def log_kpi_metrics(df, capacities_list, args):
    # Проверки входных данных
    if df is None or df.empty:
        raise ValueError("Пустой DataFrame — нечего логировать.")
    if not capacities_list or not isinstance(capacities_list, list):
        raise ValueError("Список capacities_list пустой или невалидный.")
    if not hasattr(args, 'n') or not hasattr(args, 'n_steps') or not hasattr(args, 'veh'):
        raise ValueError("Аргументы args должны содержать поля: n, n_steps, veh")

    # Настройки графика
    title_fontsize = 30
    label_fontsize = 26
    legend_fontsize = 20
    tick_fontsize = 26

    fig, ax = plt.subplots(figsize=(21, 18))  # 14 * 1.5, 12 * 1.5

    try:
        # Построение графика по емкостям
        capacities_df = pd.DataFrame(
            capacities_list,
            columns=[f'Station {i + 1}' for i in range(args.n - 1)]
        )

        capacities_df.plot(ax=ax, linewidth=5.0)  # ✅ Толщина линий = 5

        ax.set_title("Station capacity", fontsize=title_fontsize)
        ax.set_xlabel("Steps", fontsize=label_fontsize)
        ax.set_ylabel("Liters", fontsize=label_fontsize)
        ax.grid(True, which='both', linestyle='--', linewidth=1)  # ✅ Сетка включена

        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))  # ✅ Шаг по оси Y = 10

        ax.tick_params(axis='x', labelsize=tick_fontsize)
        ax.tick_params(axis='y', labelsize=tick_fontsize)

        if 'new_day' in df.columns:
            add_vertical_lines(ax, df['new_day'])

        ax.legend(loc="upper left", fontsize=legend_fontsize)

        plt.tight_layout()

        # Создание директории для вывода
        output_dir = os.path.join(
            "C:/Users/rkozl/Documents/PythonProjects/PPO_PSRP/results",
            f"{args.n}_{args.n_steps}_{args.veh}"
        )
        os.makedirs(output_dir, exist_ok=True)

        # Сохранение графика и таблицы
        image_path = os.path.join(output_dir, f"rewards_{args.n}_{args.n_steps}_{args.veh}.png")
        csv_path = os.path.join(output_dir, "results.csv")

        plt.savefig(image_path)
        df.to_csv(csv_path, index=False)

        print(f"График сохранен: {image_path}")
        print(f"CSV сохранен: {csv_path}")

    except Exception as e:
        print(f"[Ошибка] При логировании KPI: {e}")
    finally:
        plt.close(fig)


def add_vertical_lines(ax, new_day_series, alpha=0.3):
    for idx, day_end in enumerate(new_day_series):
        if day_end:
            ax.axvline(x=idx, color='red', linestyle='--', alpha=alpha, linewidth=4)

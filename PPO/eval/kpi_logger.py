# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker


# def log_kpi_metrics(df, capacities_list, args, min_capacity=None, max_capacity=None):
#     # Проверки входных данных
#     if df is None or df.empty:
#         raise ValueError("Пустой DataFrame — нечего логировать.")
#     if not capacities_list or not isinstance(capacities_list, list):
#         raise ValueError("Список capacities_list пустой или невалидный.")
#     if not hasattr(args, 'n') or not hasattr(args, 'n_steps') or not hasattr(args, 'veh'):
#         raise ValueError("Аргументы args должны содержать поля: n, n_steps, veh")

#     # Настройки графика
#     title_fontsize = 30
#     label_fontsize = 26
#     legend_fontsize = 20
#     tick_fontsize = 26

#     fig, ax = plt.subplots(figsize=(21, 18))  # 14 * 1.5, 12 * 1.5

#     try:
#         # Построение графика по емкостям
#         capacities_df = pd.DataFrame(
#             capacities_list,
#             columns=[f'Station {i + 1}' for i in range(args.n - 1)]
#         )

#         # Получаем линии и цвета
#         lines = capacities_df.plot(ax=ax, linewidth=5.0)
#         colors = [line.get_color() for line in ax.get_lines()]  # Сохраняем цвета станций

#         # Добавляем горизонтальные линии min/max capacities
#         if min_capacity is not None and max_capacity is not None:
#             for i in range(args.n - 1):
#                 color = colors[i]
#                 ax.axhline(y=min_capacity[i], color=color, linestyle='--', linewidth=2.5, alpha=0.6)
#                 ax.axhline(y=max_capacity[i], color=color, linestyle='--', linewidth=2.5, alpha=0.6)

#         ax.set_title("Station capacity", fontsize=title_fontsize)
#         ax.set_xlabel("Steps", fontsize=label_fontsize)
#         ax.set_ylabel("Liters", fontsize=label_fontsize)
#         ax.grid(True, which='both', linestyle='--', linewidth=1)

#         ax.set_ylim(0, 100)
#         ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

#         ax.tick_params(axis='x', labelsize=tick_fontsize)
#         ax.tick_params(axis='y', labelsize=tick_fontsize)

#         if 'new_day' in df.columns:
#             add_vertical_lines(ax, df['new_day'])

#         ax.legend(loc="upper left", fontsize=legend_fontsize)
#         plt.tight_layout()

#         # Создание директории для вывода
#         output_dir = os.path.join(
#             "C:/Users/rkozl/Documents/PythonProjects/PPO_PSRP/results",
#             f"{args.n}_{args.n_steps}_{args.veh}"
#         )
#         os.makedirs(output_dir, exist_ok=True)

#         # Сохранение графика и таблицы
#         image_path = os.path.join(output_dir, f"rewards_{args.n}_{args.n_steps}_{args.veh}.png")
#         csv_path = os.path.join(output_dir, "results.csv")

#         plt.savefig(image_path)
#         df.to_csv(csv_path, index=False)

#         print(f"График сохранен: {image_path}")
#         print(f"CSV сохранен: {csv_path}")

#     except Exception as e:
#         print(f"[Ошибка] При логировании KPI: {e}")
#     finally:
#         plt.close(fig)


# def add_vertical_lines(ax, new_day_series, alpha=0.3):
#     for idx, day_end in enumerate(new_day_series):
#         if day_end:
#             ax.axvline(x=idx, color='red', linestyle='--', alpha=alpha, linewidth=4)

            
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import numpy as np

def log_kpi_metrics(df, capacities_list, args, min_capacity=None, max_capacity=None):
    # Проверки входных данных
    if df is None or df.empty:
        raise ValueError("Пустой DataFrame — нечего логировать.")
    if not capacities_list or not isinstance(capacities_list, list):
        raise ValueError("Список capacities_list пустой или невалидный.")
    if not hasattr(args, 'n') or not hasattr(args, 'n_steps') or not hasattr(args, 'veh'):
        raise ValueError("Аргументы args должны содержать поля: n, n_steps, veh")
    if 'new_day' not in df.columns:
        raise ValueError("DataFrame должен содержать столбец 'new_day'")

    # Преобразуем значения в столбце 'new_day' из тензоров в булевы
    df['new_day'] = df['new_day'].apply(lambda x: x.item() if isinstance(x, torch.Tensor) else bool(x))

    # Параметр: сколько искусственных шагов на день
    steps_per_day = getattr(args, 'steps_per_day', 10)

    # Настройки графика
    title_fontsize = 30
    label_fontsize = 26
    legend_fontsize = 20
    tick_fontsize = 26

    fig, ax = plt.subplots(figsize=(21, 18))

    try:
        # Создаем DataFrame из исходных вместимостей
        capacities_df = pd.DataFrame(
            capacities_list,
            columns=[f'Station {i + 1}' for i in range(args.n - 1)]
        )

        # Получаем индексы окончаний дней
        new_day_indices = df[df['new_day']].index
        if len(new_day_indices) == 0:
            raise ValueError("Нет данных об окончании дня в столбце 'new_day'")
        new_day_indices = np.insert(new_day_indices, 0, 0)
        # Отбираем данные на конец дня
        raw_end_of_day_capacities = capacities_df.loc[new_day_indices].reset_index(drop=True)

        # Растягиваем каждый день на steps_per_day одинаковых значений
        end_of_day_capacities = pd.DataFrame(
            raw_end_of_day_capacities.loc[raw_end_of_day_capacities.index.repeat(steps_per_day)].values,
            columns=raw_end_of_day_capacities.columns
        ).reset_index(drop=True)

        # Построение графика
        lines = end_of_day_capacities.plot(ax=ax, linewidth=5.0)
        colors = [line.get_color() for line in ax.get_lines()]

        # Добавляем горизонтальные линии min/max capacities
        if min_capacity is not None and max_capacity is not None:
            for i in range(args.n - 1):
                color = colors[i]
                min_val = min_capacity[i]
                max_val = max_capacity[i]
                # Проверяем, совпадают ли min и max значения
                if min_val == max_val:
                    # Добавляем сдвиг: min_val - 1, max_val + 1
                    ax.axhline(y=min_val - 1, color=color, linestyle='--', linewidth=2.5, alpha=0.6)
                    ax.axhline(y=max_val + 1, color=color, linestyle=':', linewidth=2.5, alpha=0.6)
                else:
                    # Без сдвига, если значения не совпадают
                    ax.axhline(y=min_val, color=color, linestyle='--', linewidth=2.5, alpha=0.6)
                    ax.axhline(y=max_val, color=color, linestyle='--', linewidth=2.5, alpha=0.6)

        # Настройка осей и подписей
        ax.set_title("Station capacity", fontsize=title_fontsize)
        ax.set_xlabel("Days", fontsize=label_fontsize)
        ax.set_ylabel("Liters", fontsize=label_fontsize)
        ax.grid(True, which='both', linestyle='--', linewidth=1)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

        # Настройка оси X: показываем только реальные дни, начиная с 0
        total_days = len(raw_end_of_day_capacities)
        tick_positions = [i * steps_per_day for i in range(total_days)]
        tick_labels = [f'{i}' for i in range(total_days)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

        ax.tick_params(axis='x', labelsize=tick_fontsize)
        ax.tick_params(axis='y', labelsize=tick_fontsize)
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
        raw_end_of_day_capacities.to_csv(csv_path, index=False)

        print(f"График сохранен: {image_path}")
        print(f"CSV сохранен: {csv_path}")

    except Exception as e:
        print(f"[Ошибка] При логировании KPI: {e}")
    finally:
        plt.close(fig)
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
import torch

class RenderUtils:
    def __init__(self):
        self.figsize = (16, 10)
        self.graph_width = 0.6
        self.info_width = 0.4
        self.bar_height = 0.3

    def render(self, env, probs=None, mode='rgb_array'):
        """
        Возвращает список кадров для каждого положения машины на основе последнего шага в render_steps.
        """
        frames = []

        # Вычисляем начальные позиции машин (все в депо, если нет шагов)
        if not env.render_steps:
            vehicle_positions = torch.full((env.k_vehicles,), env.depots.item(), dtype=torch.long, device=env.device)
            frames.append(self._render_single_frame(env, [], vehicle_positions, probs))
            return frames

        # Определяем текущую машину и последний шаг
        current_vehicle = env.action_history[-1][0] if env.action_history else 0
        last_step = env.render_steps[-1]
        start = last_step['start']
        end = last_step['end']
        end_day_flag = last_step['end_day_flag']
        vehicle = last_step['vehicle']

        # Вычисляем текущие позиции машин на основе всех шагов в render_steps
        vehicle_positions = torch.full((env.k_vehicles,), env.depots.item(), dtype=torch.long, device=env.device)
        for step in env.render_steps:
            v = step['vehicle']
            vehicle_positions[v] = step['end']
            if step['end_day_flag'] and step['end'] != env.depots.item():
                vehicle_positions[v] = env.depots.item()

        # Копия позиций для промежуточных состояний
        temp_positions = vehicle_positions.clone()

        # Кадр 1: Перемещение на станцию (start -> end)
        if start != end and vehicle == current_vehicle:
            render_history = [(start, end, vehicle, False)]
            temp_positions[vehicle] = end
            frames.append(self._render_single_frame(env, render_history, temp_positions.clone(), probs))

        # Кадр 2: Возврат в депо (end -> depot), если end_day_flag и не в депо
        if end_day_flag and end != env.depots.item() and vehicle == current_vehicle:
            render_history = [(end, env.depots.item(), vehicle, True)]
            temp_positions[vehicle] = env.depots.item()
            frames.append(self._render_single_frame(env, render_history, temp_positions.clone(), probs))

        # Если день закончился полностью, показываем все машины в депо
        if env.day_end and len(frames) > 0:
            render_history = []  # Пустой граф, так как все действия завершены
            temp_positions[:] = env.depots.item()
            frames.append(self._render_single_frame(env, render_history, temp_positions.clone(), probs))

        # Если нет новых действий, возвращаем один кадр с текущими позициями
        if not frames:
            frames.append(self._render_single_frame(env, [], temp_positions.clone(), probs))

        # Очистка render_steps в конце дня
        if env.day_end:
            env.render_steps.clear()

        return frames

    def _render_single_frame(self, env, render_history, vehicle_positions, probs):
        """Вспомогательный метод для создания одного кадра."""
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(3, 2, width_ratios=[self.info_width, self.graph_width], 
                              height_ratios=[0.5, 0.2, self.bar_height])

        ax_graph = fig.add_subplot(gs[0, 1])
        ax_info = fig.add_subplot(gs[0:2, 0])
        ax_hist = fig.add_subplot(gs[1, 1])
        ax_probs = fig.add_subplot(gs[2, :])

        self._draw_graph(ax_graph, env, render_history, vehicle_positions)
        self.draw_info(ax_info, env)
        self.draw_station_histograms(ax_hist, env, vehicle_positions)
        if probs is not None:
            self.draw_probs(ax_probs, probs, env.num_nodes)

        ax_info.axis('off')
        ax_hist.axis('off')
        if probs is None:
            ax_probs.axis('off')

        plt.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return Image.fromarray(img)

    def _draw_graph(self, ax, env, render_history, vehicle_positions):
        G = nx.DiGraph()

        # Добавляем узлы
        for i in range(env.num_nodes):
            G.add_node(i, pos=env.positions[i].cpu().numpy())

        # Добавляем ребра из render_history
        for start, end, vehicle, is_return in render_history:
            G.add_edge(start, end, vehicle=vehicle, is_return=is_return, 
                       weight=int(env.get_distance(start, end).cpu().item() / 60))

        # Отрисовка узлов
        pos = nx.get_node_attributes(G, 'pos')
        depot = env.depots.item()
        nx.draw_networkx_nodes(G, pos, nodelist=[depot], node_color='skyblue', node_size=500, ax=ax, label='Depot')
        nx.draw_networkx_nodes(G, pos, nodelist=[i for i in range(env.num_nodes) if i != depot], 
                               node_color='lightgreen', node_size=300, ax=ax, label='Stations')

        # Отрисовка ребер
        edges_regular = [(u, v) for (u, v, d) in G.edges(data=True) if not d['is_return']]
        edges_return = [(u, v) for (u, v, d) in G.edges(data=True) if d['is_return']]
        nx.draw_networkx_edges(G, pos, edgelist=edges_regular, ax=ax, edge_color='gray', width=2)
        nx.draw_networkx_edges(G, pos, edgelist=edges_return, ax=ax, edge_color='orange', width=2, style='dashed')

        # Подписи узлов и ребер
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        edge_labels = {(u, v): f"{d['weight']} min" for (u, v, d) in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

        # Отрисовка машин
        vehicle_colors = ['red', 'green', 'blue', 'yellow']
        for v in range(env.k_vehicles):
            loc = vehicle_positions[v].item()
            x, y = pos[loc]
            ax.scatter(x, y, c=vehicle_colors[v % len(vehicle_colors)], s=100, marker='s', 
                       label=f'Vehicle {v}' if v == 0 else None)
            ax.text(x, y - 0.05, f"V{v}", fontsize=8, ha='center', va='bottom', 
                    color=vehicle_colors[v % len(vehicle_colors)])

        ax.legend()
        ax.set_title("Agent Movements (Current Step)")

    def draw_info(self, ax, env):
        vehicle = env.action_history[-1][0] if env.action_history else 0
        temp_load_str = ' '.join([str(round(e, 2)) for e in env.temp_load[vehicle].tolist()])
        temp_hour = env.working_hours[vehicle].item() - env.cur_remaining_time[vehicle].item() / (60 * 60)
        temp_hour = int(temp_hour)
        last_action = env.action_history[-1] if env.action_history else (0, 0, torch.zeros(env.products_count), 0)
        station_idx, delivery_percents, end_day_flag = last_action[1], last_action[2], last_action[3]
        delivery_str = ' '.join([f"P{i+1}: {int(p)}%" for i, p in enumerate(delivery_percents) if p > 0])

        # Вычисляем позиции машин для текста из render_steps
        vehicle_positions = torch.full((env.k_vehicles,), env.depots.item(), dtype=torch.long, device=env.device)
        for step in env.render_steps:
            v = step['vehicle']
            vehicle_positions[v] = step['end']
            if step['end_day_flag'] and step['end'] != env.depots.item():
                vehicle_positions[v] = env.depots.item()
        vehicle_locs = ' '.join([f"V{i}:{loc.item()}" for i, loc in enumerate(vehicle_positions)])

        info_text = (
            f"Load: {temp_load_str}\n"
            f"Step: {env.step_count}\n"
            f"Day: {env.cur_day[0].item()}\n"
            f"Time Left: {int(env.cur_remaining_time[vehicle].item())} s\n"
            f"Vehicle Locs: {vehicle_locs}\n"
            f"Distance: {int(env.total_travel_distance[0].item() / 60)} min\n"
            f"Dry Runs: {int(env.total_dry_runs[0].item())}\n"
            f"Hour: {temp_hour}\n"
            f"End Day: {'Yes' if env.day_end else 'No'}\n"
            f"Delivery: {delivery_str if delivery_str else 'None'}"
        )

        kpis = env.get_kpis()
        rewards_text = (
            f"\nRewards & Penalties:\n"
            f"Distance: {kpis['distance_rewards']:.2f}\n"
            f"Capacity: {kpis['capacity_rewards']:.2f}\n"
            f"Dry Runs: {kpis['dry_runs_penalty']:.2f}\n"
            f"Time End: {kpis['time_end_penalties']:.2f}\n"
            f"Empty Load: {kpis['empty_load_penalties']:.2f}\n"
            f"Restricted: {kpis['restricted_station_penalties']:.2f}\n"
            # f"Revisit: {kpis['revisit_penalties']:.2f}"
        )

        full_text = info_text + rewards_text
        ax.text(0.1, 0.95, full_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')

    def draw_station_histograms(self, ax, env, vehicle_positions):
        num_stations = env.num_stations
        vehicle = env.action_history[-1][0] if env.action_history else 0
        current_loc = vehicle_positions[vehicle].item()

        for i in range(num_stations + 1):
            if i == env.depots.item():
                continue
            station_idx = i - 1
            init_cap = env.init_capacities[station_idx].cpu().numpy()
            max_cap = env.max_capacities[station_idx].cpu().numpy()
            delivery = env.delivery.cpu().numpy() if i == current_loc else np.zeros_like(init_cap)

            sub_ax = ax.inset_axes([0.2 * (station_idx % 5), 0.5 - 0.5 * (station_idx // 5), 0.18, 0.45])
            products = np.arange(env.products_count)
            sub_ax.bar(products, init_cap, width=0.4, color='skyblue', label='Current')
            sub_ax.bar(products, delivery, width=0.4, bottom=init_cap, color='lightcoral', label='Delivered')
            sub_ax.plot(products, max_cap, 'r--', label='Max')
            sub_ax.set_ylim(0, max(max_cap) * 1.3)
            sub_ax.set_xticks(products)
            sub_ax.set_xticklabels([f"P{j+1}" for j in products], fontsize=6)
            sub_ax.set_title(f"Station {i}", fontsize=8)
            if station_idx == 0:
                sub_ax.legend(fontsize=6)

    def draw_probs(self, ax, probs, num_nodes):
        actions = np.arange(num_nodes)
        labels = ['Depot'] + [f'S{i+1}' for i in range(num_nodes - 1)]
        ax.bar(actions, probs, color=['salmon'] + ['cornflowerblue'] * (num_nodes - 1), edgecolor='gray')
        ax.set_xlabel('Actions')
        ax.set_ylabel('Probability')
        ax.set_title('Action Probabilities')
        ax.set_ylim(0, 1)
        ax.set_xticks(actions)
        ax.set_xticklabels(labels, rotation=45)
        ax.grid(True, linestyle='--', alpha=0.5)
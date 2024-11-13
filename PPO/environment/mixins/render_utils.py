import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

class RenderUtils:
    def draw_distance_matrix(self, draw, font):
        matrix_size = 200
        matrix_pos = (600, 0)  # Position in the top right corner
        cell_size = matrix_size // self.num_nodes

        current_day = self.cur_day.item()
        current_vehicle = self.temp_vehicle.item()
        current_matrix = self.daily_matrixes[current_day, current_vehicle].cpu().numpy()

        # Find the minimum distance (excluding self-distances)
        min_distance = np.min(current_matrix[current_matrix > 0])

        # Get the current and previous actions
        current_action = self.current_location.item()
        previous_action = self.action_history[-2] if len(self.action_history) > 1 else None

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                x = matrix_pos[0] + j * cell_size
                y = matrix_pos[1] + i * cell_size
                value = current_matrix[i, j]

                # Choose color based on whether it's the minimum distance, the agent's action, or both
                if previous_action is not None and i == previous_action and j == current_action:
                    if value == min_distance:
                        color = (128, 0, 128)  # Purple for agent's action on minimum distance
                    else:
                        color = (0, 255, 0)  # Green for agent's action
                elif value == min_distance:
                    color = (255, 0, 0)  # Red for minimum distance
                elif i == j:
                    color = (200, 200, 200)  # Grey for self-distance
                else:
                    color = (255, 255, 255)  # White for other distances

                draw.rectangle([x, y, x + cell_size, y + cell_size], fill=color, outline=(0, 0, 0))

                # Draw the distance value
                text = f"{value:.1f}"
                text_width = draw.textlength(text, font=font)
                text_height = font.size
                text_x = x + (cell_size - text_width) / 2
                text_y = y + (cell_size - text_height) / 2
                draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

        # Draw matrix title
        title = f"Distance Matrix (Day {current_day + 1}, Vehicle {current_vehicle + 1})"
        draw.text((matrix_pos[0], matrix_pos[1] - 20), title, fill=(0, 0, 0), font=font)

        # Draw legend
        legend_y = matrix_pos[1] + matrix_size + 10
        draw.rectangle([matrix_pos[0], legend_y, matrix_pos[0] + 20, legend_y + 20], fill=(255, 0, 0),
                       outline=(0, 0, 0))
        draw.text((matrix_pos[0] + 25, legend_y), "Min Distance", fill=(0, 0, 0), font=font)

        draw.rectangle([matrix_pos[0], legend_y + 25, matrix_pos[0] + 20, legend_y + 45], fill=(0, 255, 0),
                       outline=(0, 0, 0))
        draw.text((matrix_pos[0] + 25, legend_y + 25), "Agent Action", fill=(0, 0, 0), font=font)

        draw.rectangle([matrix_pos[0], legend_y + 50, matrix_pos[0] + 20, legend_y + 70], fill=(128, 0, 128),
                       outline=(0, 0, 0))
        draw.text((matrix_pos[0] + 25, legend_y + 50), "Agent Action on Min Distance", fill=(0, 0, 0), font=font)

        # Draw agent action information
        if previous_action is not None:
            action_distance = current_matrix[previous_action, current_action]
            action_type = "min distance" if action_distance == min_distance else "normal"
            action_info = f"Agent moved from {previous_action} to {current_action} ({action_type})"
            draw.text((matrix_pos[0], legend_y + 75), action_info, fill=(0, 0, 0), font=font)

    def draw_text(self, draw, font):
        temp_load_str = ''
        for e in torch.round(self.temp_load, decimals=2).tolist():
            temp_load_str += str(e)
        temp_hour = self.working_hours[self.temp_vehicle] - self.cur_remaining_time / (60 * 60)
        temp_hour = int(temp_hour.item())
        text_info = [
            f"Temp load: {temp_load_str}",
            f"Step: {self.step_count}",
            f"Current Day: {self.cur_day[0].item()}",
            f"Remaining Time: {int(self.cur_remaining_time[0].item())}",
            f"Vehicles Left: {int(self.vehicles[0].item())}",
            f"Total Travel Distance: {int(self.total_travel_distance[0].item() / (60))}",
            f"Total Dry Runs: {int(self.total_dry_runs[0].item())}",
            f"Temp hour: {temp_hour}",
        ]

        for i, text in enumerate(text_info):
            draw.text((10, 10 + i * 20), text, fill=(0, 0, 0), font=font)

    def draw_nodes(self, draw, scale, font):
        for i in range(self.num_nodes):
            x, y = self.positions[i].cpu().numpy() * scale
            x += 50
            y += 50
            color = (0, 0, 255) if i == self.depots[0].item() else (0, 255, 0)
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color)

            init_cap = self.init_capacities[i].cpu().numpy()
            delivery = self.delivery.cpu().numpy().squeeze()
            info_str = " ".join([f"{int(ic)}({int(cr)}), " for ic, cr in zip(init_cap, delivery)])

            draw.text((x - 40, y - 25), f"№{i}   "+info_str, fill=(0, 0, 0), font=font)

    def draw_edges(self, draw, scale, font):
        if len(self.action_history) == 1:
            start = 0
        else:
            start = self.action_history[-2]
        end = self.action_history[-1]

        dist = f"{int(self.get_distance(start, end).cpu().item() / (60))}"
        start_pos = self.positions[start].cpu().numpy() * scale + 50
        end_pos = self.positions[end].cpu().numpy() * scale + 50
        self.draw_line_with_text(draw, tuple(start_pos), tuple(end_pos), dist, font)

    def draw_line_with_text(self, draw, start, end, text, font, line_color=(255, 0, 0), text_color=(0, 0, 0)):
        draw.line([start, end], fill=line_color, width=2)

        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2

        text_width = int(draw.textlength(text, font=font))
        text_height = int(font.size)

        text_x = int(mid_x - text_width / 2)
        text_y = int(mid_y - text_height - 5)

        draw.text((text_x, text_y), text, font=font, fill=text_color)

    def plot_action_probabilities_on_image(self, img, probs):
        fig, ax = plt.subplots(figsize=(2, 2))  # Устанавливаем размер графика
        actions = np.arange(len(probs))
        ax.bar(actions, probs, color='blue')
        ax.set_xlabel('Actions')
        ax.set_ylabel('Probability')
        ax.set_title('Action Probabilities')
        ax.set_ylim(0, 1)  # Вероятности в диапазоне [0, 1]
        
        fig.canvas.draw()

        # Конвертируем график в изображение и помещаем его на img
        action_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        action_img = action_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        action_img = Image.fromarray(action_img)
        
        img.paste(action_img.resize((200, 200)), (10, 10))  # Вставляем график в левый верхний угол

        plt.close(fig)  # Закрываем график, чтобы освободить ресурсы

        return img

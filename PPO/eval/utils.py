
import torch as th
from PIL import Image
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor
from matplotlib import pyplot as plt

def predict_proba(model, state, lstm_states, episode_start):
    states = (th.tensor(lstm_states[0], dtype=th.float32, device=model.policy.device),
              th.tensor(lstm_states[1], dtype=th.float32, device=model.policy.device))
    episode_starts = th.tensor(episode_start, dtype=th.float32, device=model.policy.device)
    obs = obs_as_tensor(state, model.policy.device)
    dis, _ = model.policy.get_distribution(obs, states, episode_starts)
    probs = dis.distribution.probs
    return probs.detach().cpu().numpy()

def add_bar_chart_to_image(img, probs):
    fig, ax = plt.subplots(figsize=(5, 2))
    actions = np.arange(len(probs))
    ax.bar(actions, probs, color='blue')
    ax.set_xlabel('Actions')
    ax.set_ylabel('Probability')
    ax.set_title('Action Probabilities')
    ax.set_ylim(0, 1)
    fig.tight_layout()

    fig.canvas.draw()
    chart_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    chart_img = chart_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    chart_img = Image.fromarray(chart_img)
    plt.close(fig)

    total_width = max(img.width, chart_img.width)
    total_height = img.height + chart_img.height
    combined_img = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    combined_img.paste(img, (0, 0))
    combined_img.paste(chart_img, (0, img.height))

    return combined_img

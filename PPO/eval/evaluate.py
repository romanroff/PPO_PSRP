import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .kpi_logger import log_kpi_metrics
from .utils import predict_proba, add_bar_chart_to_image

def evaluate_model(model, args):
    eval_env = model.get_env()
    obs = eval_env.reset()
    done = False
    lstm_states = None
    num_envs = 1
    episode_starts = np.ones((num_envs,), dtype=bool)
    visited_actions = set()
    set_completed = False
    total_distance_fixed = False

    # Логирование KPI
    steps = 0
    capacities_list, new_day, image_arrays = [], [], []
    kpi_data = {key: [] for key in ['capacity_rewards', 'distance_rewards', 'time_end_penalties', 
                                    'empty_load_penalties', 'restricted_station_penalties', 
                                    'revisit_penalties', 'dry_runs_penalties']}

    while not done:
    # while steps < 100:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        probs = predict_proba(model, obs, lstm_states=lstm_states, episode_start=episode_starts)
        obs, rewards, done, info = eval_env.step(action)
        render = eval_env.envs[0].render()
        img = add_bar_chart_to_image(render, probs[0])
        image_arrays.append(img)

        current_action = action.item()
        visited_actions.add(current_action)

        if visited_actions == set(range(args.n)) and not set_completed:
            set_completed = True

        elif set_completed and not total_distance_fixed:
            total_distance_fixed = True

        for key in kpi_data.keys():
            kpi_data[key].append(info[0][key])

        day_end = eval_env.envs[0].day_end
        new_day.append(day_end)
        capacities_list.append(eval_env.envs[0].init_capacities[:, 0].tolist())

        episode_starts = done
        steps += 1

        if done:
        # if steps == 100:

            total_rewards = np.sum([np.array(kpi_data[key]) for key in kpi_data.keys()], axis=0)

            kpi_data.update({'total_reward': total_rewards, 'new_day': new_day})
            df = pd.DataFrame(kpi_data)
            log_kpi_metrics(df, capacities_list, args)
    
    if not os.path.exists(f"results/{args.n}_{args.n_steps}_{args.veh}/steps"):
        os.makedirs(f"results/{args.n}_{args.n_steps}_{args.veh}/steps")

    for idx, img in enumerate(image_arrays):
        img.save(f'results/{args.n}_{args.n_steps}_{args.veh}/steps/env_frame_{idx}.png')

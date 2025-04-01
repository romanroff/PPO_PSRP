import argparse
import pickle
import gymnasium
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
import datetime

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch

from PPO.environment.irp_env_custom import IRPEnv_Custom
from PPO.logger import InfoLoggerCallback, RewardsCallback, TensorboardGradientCallback
from PPO.settings import KEYS_TO_LOG, REWARDS_TO_LOG, PARAMETERS_DICT
from PPO.GNN.GNN_Gat import GATFeatureExtractor
from stable_baselines3.common.utils import Schedule
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Parse command-line arguments
args = argparse.ArgumentParser()
args.add_argument("--n", type=int, required=True)
args.add_argument("--n_steps", type=int, required=True)
args.add_argument("--veh", type=int, required=True)
args.add_argument("--timesteps", type=int, required=True)
args.add_argument("--pre_train", type=bool, default=False)
args = args.parse_args()

# Update parameters
PARAMETERS_DICT['num_nodes'] = args.n
PARAMETERS_DICT['k_vehicles'] = args.veh

# Load environment data
pkl_path = f"data_pkl/nodes-{args.n}_steps-{500}_veh-{args.veh}.pkl"
with open(pkl_path, "rb") as f:
    model_for_nn = pickle.load(f)

# Create training environment
env = IRPEnv_Custom(model_for_nn, PARAMETERS_DICT)

def maskable_predict(model, obs, env):
    """
    Функция для предсказания действия с учётом маскирования.
    """
    action_masks = env.action_masks()
    action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
    return action

# Initialize model
if args.pre_train:
    model = MaskablePPO.load(f"models/Masked_GNN_nsteps-{args.n_steps}_nodes-{args.n}_veh-{args.veh}/best_model.zip", env=env)
else:
    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs={
            "features_extractor_class": GATFeatureExtractor,
            "features_extractor_kwargs": {"embedding_size": 256},
            "net_arch": [256, 256, 256],
        },
        n_steps = args.n_steps,
        n_epochs=10,
        batch_size=32,
        ent_coef=0.01,
        learning_rate=3e-5,
        clip_range=0.3,
        gae_lambda=0.95,
        gamma=0.999,
        vf_coef=0.5,
        normalize_advantage=True,
        device='cuda',
        tensorboard_log="ppo_tensorboard/",
    )

# Set up callbacks
log_callback = InfoLoggerCallback(KEYS_TO_LOG)
rewards_callback = RewardsCallback(REWARDS_TO_LOG)
gradient_callback = TensorboardGradientCallback()

# Create evaluation environment
eval_env = IRPEnv_Custom(model_for_nn, PARAMETERS_DICT)
eval_env = gymnasium.wrappers.TimeLimit(eval_env, max_episode_steps=50)
eval_env = Monitor(eval_env, allow_early_resets=True)

# Define experiment name
exp_name = f'Masked_GNN_nsteps-{args.n_steps}_nodes-{args.n}_veh-{args.veh}'

# Set up evaluation callback
eval_callback = MaskableEvalCallback(
    eval_env,
    best_model_save_path=f"models/{exp_name}/",
    log_path=f"models/{exp_name}/",
    eval_freq=1000,
    deterministic=True,
    render=False,
)

# Train the model
model.learn(
    total_timesteps=args.timesteps,
    progress_bar=True,
    tb_log_name=exp_name,
    callback=[eval_callback, log_callback, rewards_callback],
)

# Example inference loop
obs, _ = env.reset()
for _ in range(10):
    action = maskable_predict(model, obs, env)
    obs, reward, done, _, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}")
    if done:
        obs, _ = env.reset()

import argparse
import pickle
import gymnasium
from sb3_contrib import TRPO, RecurrentPPO
import datetime

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch

from PPO.environment.irp_env_custom import IRPEnv_Custom    
from PPO.logger import InfoLoggerCallback, RewardsCallback, AdvantageLoggingCallback
from PPO.settings import KEYS_TO_LOG, REWARDS_TO_LOG, PARAMETERS_DICT
from PPO.GNN.GNN_Gat import GATFeatureExtractor
from stable_baselines3.common.utils import Schedule
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Парсинг аргументов
args = argparse.ArgumentParser()
args.add_argument("--n", type=int, required=True)
args.add_argument("--n_steps", type=int, required=True)
args.add_argument("--veh", type=int, required=True)
args.add_argument("--timesteps", type=int, required=True)
args.add_argument("--pre_train", type=bool, default=False)
args = args.parse_args()

# Обновление параметров
PARAMETERS_DICT['num_nodes'] = args.n
PARAMETERS_DICT['k_vehicles'] = args.veh

# Загрузка данных
pkl_path = f"data_pkl/nodes-{args.n}_steps-{500}_veh-{args.veh}.pkl"
with open(pkl_path, "rb") as f:
    model_for_nn = pickle.load(f)

# Создание среды
env = IRPEnv_Custom(model_for_nn, PARAMETERS_DICT)
env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=50)

if args.pre_train:
    # Загрузка предобученной модели TRPO
    model = TRPO.load(f"models_NN/radomized_GNN_TRPO_batch-{args.n_steps}_nodes-{args.n}_veh-{args.veh}/best_model.zip", env=env)
else:
    # Инициализация TRPO
    model = TRPO(
        policy="MultiInputPolicy",  
        env=env,
        policy_kwargs={
            "features_extractor_class": GATFeatureExtractor,
            "features_extractor_kwargs": {"embedding_size": 128},
            "net_arch": [512, 256, 128, 128, 128],  
            # "optimizer_kwargs": {"weight_decay": 1e-4},
        },
        n_steps=args.n_steps,  # Количество шагов для сбора данных
        learning_rate=3e-4,    # Скорость обучения
        batch_size=4048,         # Размер батча
        gamma=0.999,           # Дисконт-фактор
        gae_lambda=0.95,       # GAE-лямбда
        cg_damping=0.1,        # Дэмпинг для conjugate gradient (специфично для TRPO)
        cg_max_steps=20,       # Максимальное число итераций conjugate gradient
        target_kl=0.01,
        device='cuda',
        tensorboard_log="tppo_tensorboard/",  # Логирование в TensorBoard
    )

# Коллбэки для логирования
log_callback = InfoLoggerCallback(KEYS_TO_LOG)
rewards_callback = RewardsCallback(REWARDS_TO_LOG)
advantage_callback = AdvantageLoggingCallback()

# Среда для оценки
eval_env = IRPEnv_Custom(model_for_nn, PARAMETERS_DICT)
eval_env = gymnasium.wrappers.TimeLimit(eval_env, max_episode_steps=50)
eval_env = Monitor(eval_env, allow_early_resets=True)

# Название эксперимента
exp_name = f'radomized_GNN_TRPO_batch-{args.n_steps}_nodes-{args.n}_veh-{args.veh}'
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"models_NN/{exp_name}/",
    log_path=f"models_NN/{exp_name}/",
        eval_freq=1024*10,
    n_eval_episodes=1,
    deterministic=True,
    render=False,
)

# Обучение модели
model.learn(
    total_timesteps=args.timesteps,
    progress_bar=True,
    tb_log_name=exp_name,
    callback=[eval_callback, log_callback, rewards_callback, advantage_callback],
)
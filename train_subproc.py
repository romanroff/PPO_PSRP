import argparse
import pickle
import gymnasium
from sb3_contrib import TRPO
import datetime
import multiprocessing

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import torch

from PPO.environment.irp_env_custom import IRPEnv_Custom
from PPO.logger import InfoLoggerCallback, RewardsCallback, AdvantageLoggingCallback
from PPO.settings import KEYS_TO_LOG, REWARDS_TO_LOG, PARAMETERS_DICT
from PPO.GNN.GNN_Gat import GATFeatureExtractor
from stable_baselines3.common.utils import Schedule

def make_env(model_for_nn, parameters_dict, max_episode_steps=50):
    """
    Функция для создания среды (необходимо для векторизации)
    """
    def _init():
        env = IRPEnv_Custom(model_for_nn, parameters_dict)
        env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        
        return env
    return _init

def main():
    # Парсинг аргументов
    args = argparse.ArgumentParser()
    args.add_argument("--n", type=int, required=True)
    args.add_argument("--n_steps", type=int, required=True)
    args.add_argument("--veh", type=int, required=True)
    args.add_argument("--timesteps", type=int, required=True)
    args.add_argument("--pre_train", type=bool, default=False)
    args.add_argument("--n_envs", type=int, default=4, help="Количество параллельных сред")
    args.add_argument("--use_subproc", action="store_true", help="Использовать SubprocVecEnv вместо DummyVecEnv")
    args = args.parse_args()

    # Обновление параметров
    PARAMETERS_DICT['num_nodes'] = args.n
    PARAMETERS_DICT['k_vehicles'] = args.veh

    # Загрузка данных
    pkl_path = r'C:\Users\rkozl\Documents\PythonProjects\PPO_PSRP\data_pkl\vrp_data.pkl'
    with open(pkl_path, "rb") as f:
        model_for_nn = pickle.load(f)

    # Создание векторизованной среды
    print(f"Создание {args.n_envs} параллельных сред...")

    if args.use_subproc:
        # SubprocVecEnv - истинная параллелизация в отдельных процессах
        env = SubprocVecEnv([make_env(model_for_nn, PARAMETERS_DICT) for _ in range(args.n_envs)])
    else:
        # DummyVecEnv - псевдо-параллелизация в одном процессе (быстрее для простых сред)
        env = DummyVecEnv([make_env(model_for_nn, PARAMETERS_DICT) for _ in range(args.n_envs)])

    # Опционально: нормализация наблюдений и наград
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    if args.pre_train:
        # Загрузка предобученной модели TRPO
        model = TRPO.load(f"models/single_overfill_trpo_GNN_nsteps-{args.n_steps}_nodes-{args.n}_veh-{args.veh}_best/best_model.zip", env=env)
    else:
        # Инициализация TRPO с векторизованной средой
        model = TRPO(
            policy="MultiInputPolicy",  
            env=env,
            policy_kwargs={
                "features_extractor_class": GATFeatureExtractor,
                "features_extractor_kwargs": {"embedding_size": 128},
                "net_arch": [128, 128, 128, 128, 128],  
                # "optimizer_kwargs": {"weight_decay": 1e-4},
            },
            n_steps=args.n_steps,  # Количество шагов для сбора данных (на каждую среду)
            learning_rate=3e-4,    # Скорость обучения
            batch_size=256,         # Размер батча
            gamma=0.999,           # Дисконт-фактор
            gae_lambda=0.95,       # GAE-лямбда
            cg_damping=0.1,        # Дэмпинг для conjugate gradient (специфично для TRPO)
            cg_max_steps=20,       # Максимальное число итераций conjugate gradient
            device='cpu',
            tensorboard_log="tppo_tensorboard/",  # Логирование в TensorBoard
        )

    # Коллбэки для логирования
    log_callback = InfoLoggerCallback(KEYS_TO_LOG)
    rewards_callback = RewardsCallback(REWARDS_TO_LOG)
    advantage_callback = AdvantageLoggingCallback()

    # Среда для оценки (одиночная)
    eval_env = IRPEnv_Custom(model_for_nn, PARAMETERS_DICT)
    eval_env = gymnasium.wrappers.TimeLimit(eval_env, max_episode_steps=50)
    eval_env = Monitor(eval_env, allow_early_resets=True)

    # Название эксперимента
    exp_name = f'randomized_single_overfill_trpo_GNN_nsteps-{args.n_steps}_nodes-{args.n}_veh-{args.veh}_envs-{args.n_envs}'
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/{exp_name}/",
        log_path=f"models/{exp_name}/",
        eval_freq=1024*10,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
    )

    print(f"Начинаем обучение с {args.n_envs} параллельными средами...")
    print(f"Общее количество шагов на итерацию: {args.n_steps * args.n_envs}")

    # Обучение модели
    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=True,
        tb_log_name=exp_name,
        callback=[eval_callback, log_callback, rewards_callback, advantage_callback],
    )

    # Сохранение финальной модели
    model.save(f"models/{exp_name}/final_model")

    # Если использовалась нормализация, сохраняем её параметры
    # if isinstance(env, VecNormalize):
    #     env.save(f"models/{exp_name}/vec_normalize.pkl")

    print("Обучение завершено!")

if __name__ == "__main__":
    # Необходимо для Windows при использовании multiprocessing
    multiprocessing.freeze_support()
    main()
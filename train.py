import argparse
import pickle
import gymnasium
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Импортирование собственных модулей
from PPO.environment.irp_env_custom import IRPEnv_Custom
from PPO.logger import InfoLoggerCallback, RewardsCallback, TensorboardGradientCallback
from PPO.settings import KEYS_TO_LOG, REWARDS_TO_LOG, PARAMETERS_DICT

# Функция для обработки аргументов командной строки
def parse_args():
    parser = argparse.ArgumentParser(description="Параметры для запуска модели.")
    parser.add_argument("--n", type=int, required=True, help="Количество узлов.")
    parser.add_argument("--n_steps", type=int, required=True, help="Количество шагов.")
    parser.add_argument("--veh", type=int, required=True, help="Количество транспортных средств.")
    parser.add_argument("--timesteps", type=int, required=True, help="Количество шагов обучения.")
    parser.add_argument("--pre_train", type=bool, default=False, help="Загрузка предварительно обученной модели.")
    return parser.parse_args()

# Получение аргументов
args = parse_args()

# Настройка PARAMETERS_DICT на основе аргументов
PARAMETERS_DICT['num_nodes'] = args.n
PARAMETERS_DICT['k_vehicles'] = args.veh

# Загрузка model_for_nn из файла .pkl
pkl_path = f"data_pkl/nodes-{args.n}_steps-{args.n_steps}_veh-{args.veh}.pkl"
try:
    with open(pkl_path, "rb") as f:
        model_for_nn = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Файл {pkl_path} не найден. Проверьте путь и наличие файла.")

# Создание основной среды
env = IRPEnv_Custom(model_for_nn, PARAMETERS_DICT)

# Инициализация модели RecurrentPPO (либо загрузка, либо новая модель)
if args.pre_train:
    model = RecurrentPPO.load(rf"models/nsteps-{args.n_steps}_nodes-{args.n}_veh-{args.veh}/best_model.zip", env=env)
else:
    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        seed=32,
        verbose=1,
        n_steps=args.n_steps,
        device='cpu',
        tensorboard_log="ppo_tensorboard/",
    )

# Настройка eval среды и callbacks
eval_env = IRPEnv_Custom(model_for_nn, PARAMETERS_DICT)
eval_env = gymnasium.wrappers.TimeLimit(eval_env, max_episode_steps=300)
eval_env = Monitor(eval_env, allow_early_resets=True)

exp_name = f'nsteps-{args.n_steps}_nodes-{args.n}_veh-{args.veh}'
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"models/{exp_name}/",
    log_path=f"models/{exp_name}/",
    eval_freq=10000,
    deterministic=True,
    render=False,
)

log_callback = InfoLoggerCallback(KEYS_TO_LOG)
rewards_callback = RewardsCallback(REWARDS_TO_LOG)
gradient_callback = TensorboardGradientCallback()

# Запуск обучения модели
model.learn(
    total_timesteps=args.timesteps,
    progress_bar=True,
    tb_log_name=exp_name,
    callback=[eval_callback, log_callback, rewards_callback],
)
import pickle
from sb3_contrib import RecurrentPPO
from PPO.settings import PARAMETERS_DICT
from PPO.environment.irp_env_custom import IRPEnv_Custom
from PPO.eval.arguments import parse_args
from PPO.eval.evaluate import evaluate_model
from stable_baselines3.common.vec_env import DummyVecEnv

import warnings
warnings.filterwarnings("ignore")

def main():
    # Получение аргументов
    args = parse_args()

    # Настройка PARAMETERS_DICT на основе аргументов
    PARAMETERS_DICT['num_nodes'] = args.n
    PARAMETERS_DICT['k_vehicles'] = args.veh

    # Загрузка model_for_nn из файла .pkl
    pkl_path = f"data_pkl/nodes-{args.n}_steps-{500}_veh-{args.veh}.pkl"
    try:
        with open(pkl_path, "rb") as f:
            model_for_nn = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {pkl_path} не найден. Проверьте путь и наличие файла.")

    # Создание и обертка среды
    env = IRPEnv_Custom(model_for_nn, PARAMETERS_DICT)
    env = DummyVecEnv([lambda: env])  # Оборачиваем в DummyVecEnv

    # Загрузка модели
    model = RecurrentPPO.load(rf"models/GNN_nsteps-{args.n_steps}_nodes-{args.n}_veh-{args.veh}/best_model.zip", env=env)
    
    # Проверка совместимости пространства действий
    if not isinstance(env.action_space, model.policy.action_space.__class__):
        raise ValueError("Загруженная модель несовместима с текущим пространством действий среды.")

    # Выполнение цикла оценки и логирование метрик
    evaluate_model(model, args)

if __name__ == "__main__":
    main()
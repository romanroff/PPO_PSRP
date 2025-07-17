import pickle
from sb3_contrib import TRPO
from PPO.settings import PARAMETERS_DICT
from PPO.environment.irp_env_custom import IRPEnv_Custom
from PPO.eval.arguments import parse_args
from PPO.eval.evaluate import evaluate_model_trpo
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings

warnings.filterwarnings("ignore")

def main():
    """
    Основная функция для оценки модели TRPO на указанном файле с возвратом последовательности действий в формате [[idx_station, p1, p2, dayend],].
    Returns:
        dict: Словарь с путями к файлам и соответствующими последовательностями действий.
    """
    # Получение аргументов
    args = parse_args()

    # Настройка PARAMETERS_DICT на основе аргументов
    PARAMETERS_DICT['num_nodes'] = args.n
    PARAMETERS_DICT['k_vehicles'] = args.veh

    # Список файлов .pkl
    pkl_files = [f"data_pkl/nodes-{args.n}_steps-500_veh-{args.veh}.pkl"]
    # pkl_files = [rf'C:\Users\rkozl\Documents\PythonProjects\PPO_PSRP\data_pkl\garmisch.pkl']

    # pkl_files = [rf'C:\Users\rkozl\Documents\PythonProjects\PPO_PSRP\data_pkl\bordeux_{i}.pkl' for i in range(0,7)]

    action_sequences = {}

    for idx, pkl_path in enumerate(pkl_files):
        print(f"Обработка файла: {pkl_path}")
        try:
            # Загрузка model_for_nn из файла .pkl
            with open(pkl_path, "rb") as f:
                model_for_nn = pickle.load(f)
                print(model_for_nn)

            print('coords ',model_for_nn[0])
            print('weight matrix',model_for_nn[1])
            print('min', model_for_nn[-4])
            print('max', model_for_nn[-3])
            
        except FileNotFoundError:
            print(f"Файл {pkl_path} не найден. Пропускаем.")
            continue

        # Создание и обертка среды
        env = IRPEnv_Custom(model_for_nn, PARAMETERS_DICT)
        env = DummyVecEnv([lambda: env])  # Оборачиваем в DummyVecEnv

        # Загрузка модели
        model_path = f"models_NN/radomized_GNN_TRPO_batch-{args.n_steps}_nodes-{args.n}_veh-{args.veh}/best_model.zip"
        try:
            model = TRPO.load(model_path, env=env)
        except FileNotFoundError:
            print(f"Модель {model_path} не найдена. Пропускаем файл {pkl_path}.")
            continue

        # Проверка совместимости пространства действий
        if not isinstance(env.action_space, model.policy.action_space.__class__):
            print(f"Модель несовместима с пространством действий для файла {pkl_path}. Пропускаем.")
            continue

        # Выполнение цикла оценки и получение последовательности действий
        action_sequence = evaluate_model_trpo(model, args)
        action_sequences[idx] = action_sequence

    return action_sequences

if __name__ == "__main__":
    action_sequences = main()
    print("\nИтоговые последовательности действий:")
    print('{')
    for idx, seq in action_sequences.items():
       
        print(f"{idx}: {seq},")
    print('}')
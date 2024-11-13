import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Параметры для запуска модели.")
    parser.add_argument("--n", type=int, required=True, help="Количество узлов.")
    parser.add_argument("--n_steps", type=int, required=True, help="Количество шагов.")
    parser.add_argument("--veh", type=int, required=True, help="Количество транспортных средств.")
    return parser.parse_args()

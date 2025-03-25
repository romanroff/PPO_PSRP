from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

class TensorboardGradientCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardGradientCallback, self).__init__(verbose)
        self.writer = None

    def _on_training_start(self):
        # Инициализация TensorBoard
        self.writer = SummaryWriter(log_dir='./ppo_gradients/')

    def _on_step(self) -> bool:
        # Получаем параметры модели
        model_params = self.model.policy.parameters()

        for idx, param in enumerate(model_params):
            layer_name = f"weigts/layer_{idx}"
            self.writer.add_histogram(layer_name, param, self.num_timesteps)
            if param.grad is not None:
                layer_name = f"gradients/layer_{idx}"
                if param.name:
                    layer_name = f"gradients/{param.name}"
                self.writer.add_histogram(layer_name, param.grad, self.num_timesteps)

        return True

    def _on_training_end(self):
        # Закрываем TensorBoard
        self.writer.close()


class InfoLoggerCallback(BaseCallback):
    def __init__(self, keys_to_log, verbose=0):
        super(InfoLoggerCallback, self).__init__(verbose)
        self.keys_to_log = keys_to_log
        self.episode_infos = {key: [] for key in self.keys_to_log}  # Словарь для накопления значений

    def _on_step(self) -> bool:
        # Собираем значения из infos для текущего шага
        info = self.locals['infos'][0]  # Предполагаем одну среду, берем первый элемент
        for key in self.keys_to_log:
            if key in info:
                self.episode_infos[key].append(info[key])  # Накапливаем значения

        # Проверяем, завершился ли эпизод
        if self.locals['dones'][0]:  # Если эпизод закончился
            for key in self.keys_to_log:
                if self.episode_infos[key]:  # Если есть накопленные значения
                    mean_value = sum(self.episode_infos[key]) / len(self.episode_infos[key])
                    self.logger.record(f'episode_info/{key}', mean_value)
                self.episode_infos[key] = []  # Сбрасываем для нового эпизода
        return True



class RewardsCallback(BaseCallback):
    def __init__(self, keys_to_log, verbose=0):
        super(RewardsCallback, self).__init__(verbose)
        self.keys_to_log = keys_to_log
        self.episode_rewards = {key: 0.0 for key in keys_to_log}  # Словарь для накопления наград

    def _on_step(self) -> bool:
        # Собираем награды из infos для текущего шага
        info = self.locals['infos'][0]  # Предполагаем одну среду, берем первый элемент
        for key in self.keys_to_log:
            if key in info:
                self.episode_rewards[key] += info[key]  # Накапливаем награды

        # Проверяем, завершился ли эпизод
        if self.locals['dones'][0]:  # Если эпизод закончился
            for key in self.keys_to_log:
                # Логируем сумму наград за эпизод
                self.logger.record(f'episode/{key}', self.episode_rewards[key])
                self.episode_rewards[key] = 0.0  # Сбрасываем для нового эпизода
        return True

class ActionProbabilityCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ActionProbabilityCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        print("Action probabilities:", self.locals['log_probs'])
        return True
    


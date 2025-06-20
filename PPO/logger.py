from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

import numpy as np



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


    

class AdvantageLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_rollout_end(self):
        advantages = self.model.rollout_buffer.advantages
        mean_adv = np.mean(advantages)
        std_adv = np.std(advantages)
        self.logger.record(f'advantage/advantage_mean', mean_adv)
        self.logger.record(f'advantage/advantage_std', std_adv)

    def _on_step(self) -> bool:
        return True



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

    def _on_step(self) -> bool:
        info_values = {key: [] for key in self.keys_to_log}

        for info in self.locals['infos']:
            for key in self.keys_to_log:
                if key in info:
                    info_values[key].append(info[key])

        for key, values in info_values.items():
            if values:
                mean_value = sum(values) / len(values)
                self.logger.record(f'env_info/{key}', mean_value)

        return True

class RewardsCallback(BaseCallback):
    def __init__(self, keys_to_log, verbose=0):
        super(RewardsCallback, self).__init__(verbose)
        self.keys_to_log = keys_to_log

    def _on_step(self) -> bool:
        info_values = {key: [] for key in self.keys_to_log}

        for info in self.locals['infos']:
            for key in self.keys_to_log:
                if key in info:
                    info_values[key].append(info[key])

        for key, values in info_values.items():
            if values:
                mean_value = sum(values) / len(values)
                self.logger.record(f'rewards/{key}', mean_value)

        return True

class ActionProbabilityCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ActionProbabilityCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        print("Action probabilities:", self.locals['log_probs'])
        return True

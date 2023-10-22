
import hydra
from utils.model import get_model_parameters

class InitHydraOptimizers:
    '''Stores and runs multiple optimizers. Allows for different learning rates.'''
    def __init__(self, model, optim_cfgs, param_cfgs):
        self._optimizers = []
        for p, cfg in zip(param_cfgs, optim_cfgs):
            params = get_model_parameters(model, keys=p)
            self._optimizers.append(
                cfg.type(**cfg.args, params=params)
            )
    def zero_grad(self):
        for optim in self._optimizers: optim.zero_grad()
    def step(self):
        for optim in self._optimizers: optim.step()

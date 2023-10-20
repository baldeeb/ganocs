
class MultiOptimizer:
    '''Stores and runs multiple optimizers. Allows for different learning rates.'''
    def __init__(self, *optimizers):
        self.optimizers = optimizers

    def zero_grad(self):
        for optim in self.optimizers: optim.zero_grad()
    def step(self):
        for optim in self.optimizers: optim.step()

        
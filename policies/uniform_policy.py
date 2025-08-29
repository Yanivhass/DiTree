import numpy as np

class UniformSampler:
    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, *args, **kwargs):
        return np.expand_dims(self.action_space.sample(), axis=0)
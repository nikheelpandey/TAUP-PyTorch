# import torch
# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
from tensorboardX import SummaryWriter

from torch import Tensor
from collections import OrderedDict
import os
import matplotlib
matplotlib.use('Agg') #https://stackoverflow.com/questions/49921721/runtimeerror-main-thread-is-not-in-main-loop-with-matplotlib-and-flask
import matplotlib.pyplot as plt

class Plotter(object):
    def __init__(self):
        self.logger = OrderedDict()
    def update(self, ordered_dict):
        for key, value in ordered_dict.items():
            if isinstance(value, Tensor):
                ordered_dict[key] = value.item()
            if self.logger.get(key) is None:
                self.logger[key] = [value]
            else:
                self.logger[key].append(value)

    def save(self, file, **kwargs):
        fig, axes = plt.subplots(nrows=len(self.logger), ncols=1, figsize=(8,2*len(self.logger)))
        fig.tight_layout()
        for ax, (key, value) in zip(axes, self.logger.items()):
            ax.plot(value)
            ax.set_title(key)

        plt.savefig(file, **kwargs)
        plt.close()




class Logger(object):
    def __init__(self, log_dir, tensorboard=True, matplotlib=True):

        self.reset(log_dir, tensorboard, matplotlib)

    def reset(self, log_dir=None, tensorboard=True, matplotlib=True):

        if log_dir is not None: self.log_dir=log_dir 
        self.writer = SummaryWriter(log_dir=self.log_dir) if tensorboard else None
        self.plotter = Plotter() if matplotlib else None
        self.counter = OrderedDict()

    def update_scalers(self, ordered_dict):

        for key, value in ordered_dict.items():
            if isinstance(value, Tensor):
                ordered_dict[key] = value.item()
            if self.counter.get(key) is None:
                self.counter[key] = 1
            else:
                self.counter[key] += 1

            if self.writer:
                self.writer.add_scalar(key, value, self.counter[key])

        if self.plotter: 
            self.plotter.update(ordered_dict)
            self.plotter.save(os.path.join(self.log_dir, 'plotter.svg'))
            





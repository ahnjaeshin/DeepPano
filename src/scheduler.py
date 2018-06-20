from torch.optim.lr_scheduler import _LRScheduler
import math

"""
modified from original pytorch implementation
ADDED warm restarts

Returns:
    [type] -- [description]
"""

class CosineAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, T_max, T_e, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.T_e = T_e
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        if self.last_epoch == self.T_max:
            print('restart')
            self.last_epoch = 0
            self.T_max = self.T_max * self.T_e
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupLR(_LRScheduler):
    def __init__(self, optimizer, epochs, niter_per_ep, base_lr=None, 
                 final_lr=1e-6, warmup_epochs=0, start_warmup_lr=0.0, last_epoch=-1):
        self.epochs = epochs
        self.niter_per_ep = niter_per_ep
        self.base_lr = base_lr if base_lr is not None else optimizer.param_groups[0]['lr']
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.start_warmup_lr = start_warmup_lr
        
        self.lr_schedule = self._cosine_scheduler(
            self.base_lr, self.final_lr, epochs, niter_per_ep, 
            warmup_epochs, start_warmup_lr
        )
        
        super(CosineAnnealingWarmupLR, self).__init__(optimizer, last_epoch)
    
    def _cosine_scheduler(self, base_value, final_value, epochs, niter_per_ep, 
                         warmup_epochs=0, start_warmup_value=0.0):
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / len(iters))
        )

        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * niter_per_ep
        return schedule
    
    def get_lr(self):
        if self.last_epoch >= len(self.lr_schedule):
            return [self.final_lr for _ in self.optimizer.param_groups]
        
        current_lr = self.lr_schedule[self.last_epoch]
        return [current_lr for _ in self.optimizer.param_groups]
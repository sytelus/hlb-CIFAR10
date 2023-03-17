import torch

class OptSched:
    def __init__(self, batchsize, net, total_train_steps, num_low_lr_steps_for_ema) -> None:
        self.opt = torch.optim.AdamW(net.parameters(), lr=0.005, weight_decay=0.2)

    def lr_step(self):
        pass

    def step(self):
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad()

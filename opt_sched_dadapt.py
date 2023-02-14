import torch

import dadaptation

class OptSched:
    def __init__(self, batchsize, net, total_train_steps, num_low_lr_steps_for_ema) -> None:
        # One optimizer for the regular network, and one for the biases. This allows us to use the superconvergence onecycle training policy for our networks....
        #self.opt = dadaptation.DAdaptAdam(net.parameters(), lr=0.02)
        self.opt = torch.optim.AdamW(net.parameters(), lr=0.0002)

    def lr_step(self):
        # We only want to step the lr_schedulers while we have training steps to consume. Otherwise we get a not-so-friendly error from PyTorch
        #self.lr_sched.step()
        pass

    def step(self):
        self.opt.step()

    def zero_grad(self):
        # Using 'set_to_none' I believe is slightly faster (albeit riskier w/ funky gradient update workflows) than under the default 'set to zero' method
        self.opt.zero_grad()

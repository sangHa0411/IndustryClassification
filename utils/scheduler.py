

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def get_noam(
    optimizer: Optimizer, num_warmup_steps: int, d_model : int, last_epoch: int = -1, 
):

    def lr_lambda(current_step):
        max_value = (num_warmup_steps**-0.5) * (d_model**-0.5)
        current_step += 1
        return ((d_model**-0.5) * min(current_step**-0.5, current_step*(num_warmup_steps**-1.5))) / max_value 

    return LambdaLR(optimizer, lr_lambda, last_epoch)


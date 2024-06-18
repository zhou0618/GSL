import math
import torch
from torch.optim.optimizer import Optimizer

class GSL(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, lr_period=2000, lr_factor=0.8, interp_factor=0.1, max_grad_norm=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        lr_period=lr_period, lr_factor=lr_factor, interp_factor=interp_factor, max_grad_norm=max_grad_norm)
        super(GSL, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            max_grad_norm = group['max_grad_norm']
            total_norm = 0.0

            # Compute the norm of the gradients for clipping
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                total_norm += grad.norm().item() ** 2
            total_norm = math.sqrt(total_norm)

            # Clip gradients if total norm exceeds max_grad_norm
            clip_coef = max_grad_norm / (total_norm + 1e-6)
            if clip_coef < 1:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    p.grad.data.mul_(clip_coef)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['prev_p_data'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # Periodic learning rate adjustment
                lr = group['lr'] * (1 + group['lr_factor'] * math.sin(2 * math.pi * state['step'] / group['lr_period']))

                # Step size
                step_size = lr / bias_correction1

                # Linear interpolation with previous parameter values
                interp_factor = group['interp_factor']
                prev_p_data = state['prev_p_data']
                new_p_data = p.data.addcdiv(-step_size, exp_avg, denom)
                p.data = (1 - interp_factor) * new_p_data + interp_factor * prev_p_data
                state['prev_p_data'] = p.data.clone()

        return loss
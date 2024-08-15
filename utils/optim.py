# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from typing import List
class Adan(Optimizer):
    """
    Implements a pytorch variant of Adan
    Adan was proposed in
    Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models[J]. arXiv preprint arXiv:2208.06677, 2022.
    https://arxiv.org/abs/2208.06677
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float, flot], optional): coefficients used for computing 
            running averages of gradient and its norm. (default: (0.98, 0.92, 0.99))
        eps (float, optional): term added to the denominator to improve 
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): decoupled weight decay (L2 penalty) (default: 0)
        max_grad_norm (float, optional): value used to clip 
            global grad norm (default: 0.0 no clip)
        no_prox (bool): how to perform the decoupled weight decay (default: False)
        foreach (bool): if True would use torch._foreach implementation. It's faster but uses
            slightly more memory. (default: True)
    """

    def __init__(self, params, lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-8,
                 weight_decay=0.0, max_grad_norm=0.0, no_prox=False, foreach: bool=True):
        if not 0.0 <= max_grad_norm:
            raise ValueError("Invalid Max grad norm: {}".format(max_grad_norm))
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm, no_prox=no_prox, foreach=foreach)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super(Adan, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('no_prox', False)

    @torch.no_grad()
    def restart_opt(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # State initialization

                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    # Exponential moving average of gradient difference
                    state['exp_avg_diff'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        """
            Performs a single optimization step.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.defaults['max_grad_norm'] > 0:
            device = self.param_groups[0]['params'][0].device
            global_grad_norm = torch.zeros(1, device=device)

            max_grad_norm = torch.tensor(self.defaults['max_grad_norm'], device=device)
            for group in self.param_groups:

                for p in group['params']:
                    if p.grad is not None:
                        grad = p.grad
                        global_grad_norm.add_(grad.pow(2).sum())

            global_grad_norm = torch.sqrt(global_grad_norm)

            clip_global_grad_norm = torch.clamp(max_grad_norm / (global_grad_norm + group['eps']), max=1.0)
        else:
            clip_global_grad_norm = 1.0

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avg_diffs = []
            pre_grads = []
            
            beta1, beta2, beta3 = group['betas']
            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            bias_correction1 = 1.0 - beta1 ** group['step']
            bias_correction2 = 1.0 - beta2 ** group['step']
            bias_correction3 = 1.0 - beta3 ** group['step']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_diff'] = torch.zeros_like(p)

                if 'pre_grad' not in state or group['step'] == 1:
                    # at first step grad wouldn't be clipped by `clip_global_grad_norm`
                    # this is only to simplify implementation
                    state['pre_grad'] = p.grad

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                exp_avg_diffs.append(state['exp_avg_diff'])
                pre_grads.append(state['pre_grad'])

            kwargs = dict(
                params=params_with_grad,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                exp_avg_diffs=exp_avg_diffs,
                pre_grads=pre_grads,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                bias_correction1=bias_correction1,
                bias_correction2=bias_correction2,
                bias_correction3_sqrt=math.sqrt(bias_correction3),
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                no_prox=group['no_prox'],
                clip_global_grad_norm=clip_global_grad_norm,
            )
            if group["foreach"]:
                copy_grads = _multi_tensor_adan(**kwargs)
            else:
                copy_grads = _single_tensor_adan(**kwargs)

            for p, copy_grad in zip(params_with_grad, copy_grads):
                self.state[p]['pre_grad'] = copy_grad
        
        return loss
        
def _single_tensor_adan(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    exp_avg_diffs: List[Tensor],
    pre_grads: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3_sqrt: float,
    lr: float,
    weight_decay: float,
    eps: float,
    no_prox: bool,
    clip_global_grad_norm: Tensor,
):
    copy_grads = []
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_diff = exp_avg_diffs[i]
        pre_grad = pre_grads[i]

        grad = grad.mul_(clip_global_grad_norm)
        copy_grads.append(grad.clone())
        
        diff = grad - pre_grad
        update = grad + beta2 * diff
        
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
        exp_avg_diff.mul_(beta2).add_(diff, alpha=1 - beta2)  # diff_t
        exp_avg_sq.mul_(beta3).addcmul_(update, update, value=1 - beta3)  # n_t

        denom = ((exp_avg_sq).sqrt() / bias_correction3_sqrt).add_(eps)
        update = ((exp_avg / bias_correction1 + beta2 * exp_avg_diff / bias_correction2)).div_(denom)

        if no_prox:
            param.mul_(1 - lr * weight_decay)
            param.add_(update, alpha=-lr)
        else:
            param.add_(update, alpha=-lr)
            param.div_(1 + lr * weight_decay)
    return copy_grads

def _multi_tensor_adan(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    exp_avg_diffs: List[Tensor],
    pre_grads: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3_sqrt: float,
    lr: float,
    weight_decay: float,
    eps: float,
    no_prox: bool,
    clip_global_grad_norm: Tensor,
):
    if clip_global_grad_norm<1.0:
        torch._foreach_mul_(grads, clip_global_grad_norm.item())
    copy_grads = [g.clone() for g in grads]

    diff = torch._foreach_sub(grads, pre_grads)
    # NOTE: line below while looking identical gives different result, due to float precision errors.
    # using mul+add produces identical results to single-tensor, using add+alpha doesn't
    # On cuda this difference doesn't matter due to its' own precision non-determinism
    # update = torch._foreach_add(grads, torch._foreach_mul(diff, beta2))
    update = torch._foreach_add(grads, diff, alpha=beta2)

    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1) # m_t

    torch._foreach_mul_(exp_avg_diffs, beta2)
    torch._foreach_add_(exp_avg_diffs, diff, alpha=1 - beta2) # diff_t

    torch._foreach_mul_(exp_avg_sqs, beta3)
    torch._foreach_addcmul_(exp_avg_sqs, update, update, value=1 - beta3) # n_t

    denom = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denom, bias_correction3_sqrt)
    torch._foreach_add_(denom, eps)

    update = torch._foreach_div(exp_avgs, bias_correction1)
    # NOTE: same issue as above. beta2 * diff / bias_correction2 != diff * (beta2 / bias_correction2)
    # using faster version by default. 
    # torch._foreach_add_(update, torch._foreach_div(torch._foreach_mul(exp_avg_diffs, beta2), bias_correction2))
    torch._foreach_add_(update, torch._foreach_mul(exp_avg_diffs, beta2 / bias_correction2))
    torch._foreach_div_(update, denom)

    if no_prox:
        torch._foreach_mul_(params, 1 - lr * weight_decay)
        torch._foreach_add_(params, update, alpha=-lr)
    else:
        torch._foreach_add_(params, update, alpha=-lr)
        torch._foreach_div_(params, 1 + lr * weight_decay)
    return copy_grads





class gassopti(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.9), threshold=2, eps=1e-8,
                 weight_decay=0.0, max_grad_norm=0.0, no_prox=False, foreach: bool=False):
        if not 0.0 <= max_grad_norm:
            raise ValueError("Invalid Max grad norm: {}".format(max_grad_norm))
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, threshold=threshold,
                        max_grad_norm=max_grad_norm, no_prox=no_prox, foreach=foreach)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super(gassopti, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('no_prox', False)

    @torch.no_grad()
    def restart_opt(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # State initialization

                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        """
            Performs a single optimization step.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.defaults['max_grad_norm'] > 0:
            device = self.param_groups[0]['params'][0].device
            global_grad_norm = torch.zeros(1, device=device)

            max_grad_norm = torch.tensor(self.defaults['max_grad_norm'], device=device)
            for group in self.param_groups:

                for p in group['params']:
                    if p.grad is not None:
                        grad = p.grad
                        global_grad_norm.add_(grad.pow(2).sum())

            global_grad_norm = torch.sqrt(global_grad_norm)

            clip_global_grad_norm = torch.clamp(max_grad_norm / (global_grad_norm + group['eps']), max=1.0)
        else:
            clip_global_grad_norm = 1.0

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avg_diffs = []
            pre_grads = []
            
            beta1, beta2 = group['betas']
            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            bias_correction1 = 1.0 - beta1 ** group['step']
            bias_correction2 = 1.0 - beta2 ** group['step']
            # bias_correction3 = 1.0 - beta3 ** group['step']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                if 'pre_grad' not in state or group['step'] == 1:
                    # at first step grad wouldn't be clipped by `clip_global_grad_norm`
                    # this is only to simplify implementation
                    state['pre_grad'] = p.grad

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                # exp_avg_diffs.append(state['exp_avg_diff'])
                pre_grads.append(state['pre_grad'])

            kwargs = dict(
                params=params_with_grad,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                exp_avg_diffs=exp_avg_diffs,
                pre_grads=pre_grads,
                beta1=beta1,
                beta2=beta2,
                beta3=0,
                bias_correction1=bias_correction1,
                bias_correction2=bias_correction2,
                bias_correction3_sqrt=math.sqrt(0),
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                no_prox=group['no_prox'],
                threshold=group['threshold'],
                clip_global_grad_norm=clip_global_grad_norm,
            )
            if group["foreach"]:
                copy_grads = _multi_tensor_sgopti(**kwargs)
            else:
                copy_grads = _single_tensor_sgopti(**kwargs)

            for p, copy_grad in zip(params_with_grad, copy_grads):
                self.state[p]['pre_grad'] = copy_grad
        
        return loss
        
def _single_tensor_sgopti(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    exp_avg_diffs: List[Tensor],
    pre_grads: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3_sqrt: float,
    lr: float,
    weight_decay: float,
    eps: float,
    no_prox: bool,
    threshold: float,
    clip_global_grad_norm: Tensor,
):
    copy_grads = []
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        # exp_avg_diff = exp_avg_diffs[i]
        # pre_grad = pre_grads[i]

        grad = grad.mul_(clip_global_grad_norm)
        copy_grads.append(grad.clone())
        
        # diff = grad - pre_grad
        # update = grad + beta2 * diff
        
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
        # exp_avg_diff.mul_(beta2).add_(diff, alpha=1 - beta2)  # diff_t
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # denom = ((exp_avg_sq).sqrt() / bias_correction3_sqrt).add_(eps)
        # update = ((exp_avg / bias_correction1 + beta2 * exp_avg_diff / bias_correction2)).div_(denom)
        denom = exp_avg/bias_correction1 / (torch.abs(exp_avg_sq/bias_correction2-(exp_avg/bias_correction1)**2).sqrt()).add_(eps)
        update = torch.clamp(denom, -threshold, threshold)

        # update = exp_avg / (torch.abs_(exp_avg).add_(eps))

        if no_prox:
            param.mul_(1 - lr * weight_decay)
            param.add_(update, alpha=-lr)
        else:
            param.add_(update, alpha=-lr)
            param.div_(1 + lr * weight_decay)
    return copy_grads

def _multi_tensor_sgopti(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    exp_avg_diffs: List[Tensor],
    pre_grads: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3_sqrt: float,
    lr: float,
    weight_decay: float,
    eps: float,
    no_prox: bool,
    threshold: float,
    clip_global_grad_norm: Tensor,
):
    if clip_global_grad_norm<1.0:
        torch._foreach_mul_(grads, clip_global_grad_norm.item())
    copy_grads = [g.clone() for g in grads]

    diff = torch._foreach_sub(grads, pre_grads)
    # NOTE: line below while looking identical gives different result, due to float precision errors.
    # using mul+add produces identical results to single-tensor, using add+alpha doesn't
    # On cuda this difference doesn't matter due to its' own precision non-determinism
    # update = torch._foreach_add(grads, torch._foreach_mul(diff, beta2))
    # update = torch._foreach_add(grads, diff, alpha=beta2)

    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1) # m_t

    torch._foreach_mul_(exp_avg_diffs, beta2)
    torch._foreach_add_(exp_avg_diffs, diff, alpha=1 - beta2) # diff_t

    torch._foreach_mul_(exp_avg_sqs, beta3)
    torch._foreach_addcmul_(exp_avg_sqs, update, update, value=1 - beta3) # n_t

    denom = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denom, bias_correction3_sqrt)
    torch._foreach_add_(denom, eps)

    update = torch._foreach_div(exp_avgs, bias_correction1)
    # NOTE: same issue as above. beta2 * diff / bias_correction2 != diff * (beta2 / bias_correction2)
    # using faster version by default. 
    # torch._foreach_add_(update, torch._foreach_div(torch._foreach_mul(exp_avg_diffs, beta2), bias_correction2))
    torch._foreach_add_(update, torch._foreach_mul(exp_avg_diffs, beta2 / bias_correction2))
    torch._foreach_div_(update, denom)

    if no_prox:
        torch._foreach_mul_(params, 1 - lr * weight_decay)
        torch._foreach_add_(params, update, alpha=-lr)
    else:
        torch._foreach_add_(params, update, alpha=-lr)
        torch._foreach_div_(params, 1 + lr * weight_decay)
    return copy_grads



class gasspidopti(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.5, 0.5), threshold=2, eps=1e-8,
                 weight_decay=0.0, max_grad_norm=0.0, no_prox=False, foreach: bool=False):
        if not 0.0 <= max_grad_norm:
            raise ValueError("Invalid Max grad norm: {}".format(max_grad_norm))
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[2]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, threshold=threshold,
                        max_grad_norm=max_grad_norm, no_prox=no_prox, foreach=foreach)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super(gasspidopti, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('no_prox', False)

    @torch.no_grad()
    def restart_opt(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # State initialization

                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_diff'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        """
            Performs a single optimization step.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.defaults['max_grad_norm'] > 0:
            device = self.param_groups[0]['params'][0].device
            global_grad_norm = torch.zeros(1, device=device)

            max_grad_norm = torch.tensor(self.defaults['max_grad_norm'], device=device)
            for group in self.param_groups:

                for p in group['params']:
                    if p.grad is not None:
                        grad = p.grad
                        global_grad_norm.add_(grad.pow(2).sum())

            global_grad_norm = torch.sqrt(global_grad_norm)

            clip_global_grad_norm = torch.clamp(max_grad_norm / (global_grad_norm + group['eps']), max=1.0)
        else:
            clip_global_grad_norm = 1.0

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avg_diffs = []
            pre_grads = []
            
            beta1, beta2, beta3 = group['betas']
            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            bias_correction1 = 1.0 - beta1 ** group['step']
            bias_correction2 = 1.0 - beta2 ** group['step']
            # bias_correction3 = 1.0 - beta3 ** group['step']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_diff'] = torch.zeros_like(p)

                if 'pre_grad' not in state or group['step'] == 1:
                    # at first step grad wouldn't be clipped by `clip_global_grad_norm`
                    # this is only to simplify implementation
                    state['pre_grad'] = p.grad

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                exp_avg_diffs.append(state['exp_avg_diff'])
                # exp_avg_diffs.append(state['exp_avg_diff'])
                pre_grads.append(state['pre_grad'])

            kwargs = dict(
                params=params_with_grad,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                exp_avg_diffs=exp_avg_diffs,
                pre_grads=pre_grads,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                bias_correction1=bias_correction1,
                bias_correction2=bias_correction2,
                bias_correction3_sqrt=math.sqrt(0),
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                no_prox=group['no_prox'],
                threshold=group['threshold'],
                clip_global_grad_norm=clip_global_grad_norm,
            )
            if group["foreach"]:
                copy_grads = _multi_tensor_gasspidopti(**kwargs)
            else:
                copy_grads = _single_tensor_gasspidopti(**kwargs)

            for p, copy_grad in zip(params_with_grad, copy_grads):
                self.state[p]['pre_grad'] = copy_grad
        
        return loss
        
def _single_tensor_gasspidopti(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    exp_avg_diffs: List[Tensor],
    pre_grads: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3_sqrt: float,
    lr: float,
    weight_decay: float,
    eps: float,
    no_prox: bool,
    threshold: float,
    clip_global_grad_norm: Tensor,
):
    copy_grads = []
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_diff = exp_avg_diffs[i]
        pre_grad = pre_grads[i]

        grad = grad.mul_(clip_global_grad_norm)
        copy_grads.append(grad.clone())
        
        diff = grad - pre_grad
        # update = grad + beta2 * diff
        
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
        exp_avg_diff.mul_(beta2).add_(diff, alpha=1 - beta2)  # diff_t
        exp_avg_sq.mul_(beta1).addcmul_(grad, grad, value=1 - beta1)

        # denom = ((exp_avg_sq).sqrt() / bias_correction3_sqrt).add_(eps)
        # update = ((exp_avg / bias_correction1 + beta2 * exp_avg_diff / bias_correction2)).div_(denom)
        denom = (exp_avg / bias_correction1 + beta3 * exp_avg_diff / bias_correction2) / (torch.abs(exp_avg_sq/bias_correction1-(exp_avg/bias_correction1)**2).sqrt()).add_(eps)
        update = torch.clamp(denom, -threshold, threshold)

        # update = exp_avg / (torch.abs_(exp_avg).add_(eps))

        if no_prox:
            param.mul_(1 - lr * weight_decay)
            param.add_(update, alpha=-lr)
        else:
            param.add_(update, alpha=-lr)
            param.div_(1 + lr * weight_decay)
    return copy_grads

def _multi_tensor_gasspidopti(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    exp_avg_diffs: List[Tensor],
    pre_grads: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3_sqrt: float,
    lr: float,
    weight_decay: float,
    eps: float,
    no_prox: bool,
    threshold: float,
    clip_global_grad_norm: Tensor,
):
    if clip_global_grad_norm<1.0:
        torch._foreach_mul_(grads, clip_global_grad_norm.item())
    copy_grads = [g.clone() for g in grads]

    diff = torch._foreach_sub(grads, pre_grads)
    # NOTE: line below while looking identical gives different result, due to float precision errors.
    # using mul+add produces identical results to single-tensor, using add+alpha doesn't
    # On cuda this difference doesn't matter due to its' own precision non-determinism
    # update = torch._foreach_add(grads, torch._foreach_mul(diff, beta2))
    # update = torch._foreach_add(grads, diff, alpha=beta2)

    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1) # m_t

    torch._foreach_mul_(exp_avg_diffs, beta2)
    torch._foreach_add_(exp_avg_diffs, diff, alpha=1 - beta2) # diff_t

    torch._foreach_mul_(exp_avg_sqs, beta3)
    torch._foreach_addcmul_(exp_avg_sqs, update, update, value=1 - beta3) # n_t

    denom = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denom, bias_correction3_sqrt)
    torch._foreach_add_(denom, eps)

    update = torch._foreach_div(exp_avgs, bias_correction1)
    # NOTE: same issue as above. beta2 * diff / bias_correction2 != diff * (beta2 / bias_correction2)
    # using faster version by default. 
    # torch._foreach_add_(update, torch._foreach_div(torch._foreach_mul(exp_avg_diffs, beta2), bias_correction2))
    torch._foreach_add_(update, torch._foreach_mul(exp_avg_diffs, beta2 / bias_correction2))
    torch._foreach_div_(update, denom)

    if no_prox:
        torch._foreach_mul_(params, 1 - lr * weight_decay)
        torch._foreach_add_(params, update, alpha=-lr)
    else:
        torch._foreach_add_(params, update, alpha=-lr)
        torch._foreach_div_(params, 1 + lr * weight_decay)
    return copy_grads





class Agassopti(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.9), threshold=2, eps=1e-8, down_rate=0.9999,
                 up_rate=1.001, weight_decay=0.0, max_grad_norm=0.0, no_prox=False, foreach: bool=False):
        if not 0.0 <= max_grad_norm:
            raise ValueError("Invalid Max grad norm: {}".format(max_grad_norm))
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, up_rate=up_rate, down_rate=down_rate,
                        weight_decay=weight_decay, threshold=threshold,
                        max_grad_norm=max_grad_norm, no_prox=no_prox, foreach=foreach)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super(Agassopti, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('no_prox', False)

    @torch.no_grad()
    def restart_opt(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # State initialization

                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['scale'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        """
            Performs a single optimization step.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.defaults['max_grad_norm'] > 0:
            device = self.param_groups[0]['params'][0].device
            global_grad_norm = torch.zeros(1, device=device)

            max_grad_norm = torch.tensor(self.defaults['max_grad_norm'], device=device)
            for group in self.param_groups:

                for p in group['params']:
                    if p.grad is not None:
                        grad = p.grad
                        global_grad_norm.add_(grad.pow(2).sum())

            global_grad_norm = torch.sqrt(global_grad_norm)

            clip_global_grad_norm = torch.clamp(max_grad_norm / (global_grad_norm + group['eps']), max=1.0)
        else:
            clip_global_grad_norm = 1.0

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avg_diffs = []
            pre_grads = []
            scales = []
            
            beta1, beta2 = group['betas']
            up_rate = group['up_rate']
            down_rate = group['down_rate']
            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            bias_correction1 = 1.0 - beta1 ** group['step']
            bias_correction2 = 1.0 - beta2 ** group['step']
            # bias_correction3 = 1.0 - beta3 ** group['step']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['scale'] = torch.ones_like(p)

                if 'pre_grad' not in state or group['step'] == 1:
                    # at first step grad wouldn't be clipped by `clip_global_grad_norm`
                    # this is only to simplify implementation
                    state['pre_grad'] = p.grad

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                # exp_avg_diffs.append(state['exp_avg_diff'])
                pre_grads.append(state['pre_grad'])
                scales.append(state['scale'])

            kwargs = dict(
                params=params_with_grad,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                exp_avg_diffs=exp_avg_diffs,
                pre_grads=pre_grads,
                scales=scales,
                up_rate=up_rate,
                down_rate=down_rate,
                beta1=beta1,
                beta2=beta2,
                beta3=0,
                bias_correction1=bias_correction1,
                bias_correction2=bias_correction2,
                bias_correction3_sqrt=math.sqrt(0),
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                no_prox=group['no_prox'],
                threshold=group['threshold'],
                clip_global_grad_norm=clip_global_grad_norm,
            )
            if group["foreach"]:
                copy_grads = _multi_tensor_Asgopti(**kwargs)
            else:
                copy_grads = _single_tensor_Asgopti(**kwargs)

            for p, copy_grad in zip(params_with_grad, copy_grads):
                self.state[p]['pre_grad'] = copy_grad
        
        return loss
        
def _single_tensor_Asgopti(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    exp_avg_diffs: List[Tensor],
    pre_grads: List[Tensor],
    scales: List[Tensor],
    up_rate: float,
    down_rate: float,
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3_sqrt: float,
    lr: float,
    weight_decay: float,
    eps: float,
    no_prox: bool,
    threshold: float,
    clip_global_grad_norm: Tensor,
):
    copy_grads = []
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        # exp_avg_diff = exp_avg_diffs[i]
        pre_grad = pre_grads[i]
        scale = scales[i]

        grad = grad.mul_(clip_global_grad_norm)
        copy_grads.append(grad.clone())
        
        # diff = grad - pre_grad
        # update = grad + beta2 * diff
        last_exp_avg = exp_avg.clone()
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t

        up_data = (torch.abs(exp_avg - last_exp_avg) >= torch.abs(exp_avg + last_exp_avg)).type(grad.type())
        down_data = (torch.abs(exp_avg - last_exp_avg) <= torch.abs(exp_avg + last_exp_avg)).type(grad.type())

        # up_data = (torch.abs(grad - pre_grad) >= torch.abs(grad + pre_grad)).type(grad.type())
        # down_data = (torch.abs(grad - pre_grad) <= torch.abs(grad + pre_grad)).type(grad.type())

        scale = scale.mul_(up_data*up_rate + down_data*down_rate)

        # exp_avg_diff.mul_(beta2).add_(diff, alpha=1 - beta2)  # diff_t
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # denom = ((exp_avg_sq).sqrt() / bias_correction3_sqrt).add_(eps)
        # update = ((exp_avg / bias_correction1 + beta2 * exp_avg_diff / bias_correction2)).div_(denom)
        # denom = exp_avg/bias_correction1 / (torch.abs(exp_avg_sq/bias_correction2-(exp_avg/bias_correction1)**2).sqrt()).mul_(scale).add_(eps)
        denom = exp_avg/bias_correction1 / (torch.abs(exp_avg_sq/bias_correction2).sqrt()).mul_(scale).add_(eps)
        update = torch.clamp(denom, -threshold, threshold)
        # update = denom

        # update = exp_avg / (torch.abs_(exp_avg).add_(eps))

        if no_prox:
            param.mul_(1 - lr * weight_decay)
            param.add_(update, alpha=-lr)
        else:
            param.add_(update, alpha=-lr)
            param.div_(1 + lr * weight_decay)
    return copy_grads

def _multi_tensor_Asgopti(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    exp_avg_diffs: List[Tensor],
    pre_grads: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    bias_correction1: float,
    bias_correction2: float,
    bias_correction3_sqrt: float,
    lr: float,
    weight_decay: float,
    eps: float,
    no_prox: bool,
    threshold: float,
    clip_global_grad_norm: Tensor,
):
    if clip_global_grad_norm<1.0:
        torch._foreach_mul_(grads, clip_global_grad_norm.item())
    copy_grads = [g.clone() for g in grads]

    diff = torch._foreach_sub(grads, pre_grads)
    # NOTE: line below while looking identical gives different result, due to float precision errors.
    # using mul+add produces identical results to single-tensor, using add+alpha doesn't
    # On cuda this difference doesn't matter due to its' own precision non-determinism
    # update = torch._foreach_add(grads, torch._foreach_mul(diff, beta2))
    # update = torch._foreach_add(grads, diff, alpha=beta2)

    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1) # m_t

    torch._foreach_mul_(exp_avg_diffs, beta2)
    torch._foreach_add_(exp_avg_diffs, diff, alpha=1 - beta2) # diff_t

    torch._foreach_mul_(exp_avg_sqs, beta3)
    torch._foreach_addcmul_(exp_avg_sqs, update, update, value=1 - beta3) # n_t

    denom = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denom, bias_correction3_sqrt)
    torch._foreach_add_(denom, eps)

    update = torch._foreach_div(exp_avgs, bias_correction1)
    # NOTE: same issue as above. beta2 * diff / bias_correction2 != diff * (beta2 / bias_correction2)
    # using faster version by default. 
    # torch._foreach_add_(update, torch._foreach_div(torch._foreach_mul(exp_avg_diffs, beta2), bias_correction2))
    torch._foreach_add_(update, torch._foreach_mul(exp_avg_diffs, beta2 / bias_correction2))
    torch._foreach_div_(update, denom)

    if no_prox:
        torch._foreach_mul_(params, 1 - lr * weight_decay)
        torch._foreach_add_(params, update, alpha=-lr)
    else:
        torch._foreach_add_(params, update, alpha=-lr)
        torch._foreach_div_(params, 1 + lr * weight_decay)
    return copy_grads


class Lion(Optimizer):
  r"""Implements Lion algorithm."""

  def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
    """Initialize the hyperparameters.
    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
      lr (float, optional): learning rate (default: 1e-4)
      betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.99))
      weight_decay (float, optional): weight decay coefficient (default: 0)
    """

    if not 0.0 <= lr:
      raise ValueError('Invalid learning rate: {}'.format(lr))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
    defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
    super().__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.
    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    Returns:
      the loss.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        # Perform stepweight decay
        p.data.mul_(1 - group['lr'] * group['weight_decay'])

        grad = p.grad
        state = self.state[p]
        # State initialization
        if len(state) == 0:
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(p)

        exp_avg = state['exp_avg']
        beta1, beta2 = group['betas']

        # Weight update
        update = exp_avg * beta1 + grad * (1 - beta1)
        p.add_(torch.sign(update), alpha=-group['lr'])
        # Decay the momentum running average coefficient
        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

    return loss
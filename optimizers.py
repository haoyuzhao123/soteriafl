try:
    import cupy as xp
except ModuleNotFoundError:
    import numpy as xp

import numpy as np

from nda import log
from nda.optimizers import Optimizer   
from nda.optimizers import compressor

def unbiased_random(x, a):
    dim = x.shape[0]
    return (dim / a) * compressor.random(x, a)

class FederatedOptimizer(Optimizer):
    def __init__(self, p, batch_size=1, eta=0.1, compression='identity', compression_param=None, perturbation_variance=None, G=None, **kwargs):
        super().__init__(p, **kwargs)
        self.eta = eta
        self.compression_param = compression_param
        self.compression = compression
        self.perturbation_variance = perturbation_variance
        self.batch_size = batch_size
        self.G = G

        if self.x.ndim == 2:
            self.x = self.x.mean(axis=1)
        self.x = self.x.reshape(-1, 1).repeat(self.p.n_agent, axis=1)

    def init(self):
        if self.compression == 'natural_compression':
            self.compression_operator = compressor.natural_compression
        elif self.compression == 'unbiased_random_sparsification':
            self.compression_operator = lambda x: unbiased_random(x, self.compression_param)
        elif self.compression == 'random_sparsification':
            self.compression_operator = lambda x: compressor.random(x, self.compression_param)
        elif self.compression == 'random_quantization':
            self.compression_operator = lambda x: compressor.random_quantization(x, self.compression_param)
        elif self.compression == 'identity':
            self.compression_operator = compressor.identity
        else:
            raise NotImplemented(f'Compression {self.compression} is not implemented')

class CDP_SGD(FederatedOptimizer):

    def update(self):
        self.comm_rounds += 1

        sample_list = xp.random.randint(0, self.p.m, (self.p.n_agent, self.batch_size))
        grad = self.grad(self.x, j=sample_list)
        # Gradient clipping
        grad *= xp.clip(self.G / xp.linalg.norm(grad, axis=0), 0, 1)

        if self.perturbation_variance is not None:
            grad += self.perturbation_variance * xp.random.randn(*self.x.shape)

        grad = self.compression_operator(grad)
        self.x -= self.eta * grad.mean(axis=1, keepdims=True)


class SoteriaFL(FederatedOptimizer):
    def __init__(self, p, gamma=0.1, local_update_method='sgd', p_update_snapshot=None, **kwargs):
        super().__init__(p, **kwargs)
        self.gamma = gamma
        self.local_update_method = local_update_method
        self.p_update_snapshot = p_update_snapshot

        if local_update_method == 'sgd':
            self.local_update = self.local_update_sgd
        elif local_update_method == 'svrg':
            self.local_update = self.local_update_svrg
            
        self.name += f'-{local_update_method.upper()}'

    def init(self):
        super().init()
        self.s_i = xp.zeros_like(self.x)
        self.s = self.s_i.mean(axis=1, keepdims=True)
        if self.local_update_method == 'svrg':
            self.w = self.x.copy()
            self.grad_w = self.grad(self.w)
            self.should_update_snapshot = xp.random.binomial(1, self.p_update_snapshot, self.n_iters + 1)

    def local_update_sgd(self):
        sample_list = xp.random.randint(0, self.p.m, (self.p.n_agent, self.batch_size))
        return self.grad(self.x, j=sample_list)

    def local_update_svrg(self):
        sample_list = xp.random.randint(0, self.p.m, (self.p.n_agent, self.batch_size))
        g_i = self.grad(self.x, j=sample_list) - self.grad(self.w, j=sample_list) + self.grad_w
        if self.should_update_snapshot[self.t]:
            self.w = self.x.copy()
            self.grad_w = self.grad(self.w)  # added snapshot gradient update
        return g_i

    def update(self):
        self.comm_rounds += 1

        # Agents
        g_i = self.local_update()
        # Gradient clipping
        g_i *= xp.clip(self.G / xp.linalg.norm(g_i, axis=0), 0, 1)

        if self.perturbation_variance is not None:
            g_i += self.perturbation_variance * xp.random.randn(*self.x.shape)
        v_i = self.compression_operator(g_i - self.s_i)
        self.s_i += self.gamma * v_i

        # Server
        v = v_i.mean(axis=1, keepdims=True)
        self.x -= self.eta * (self.s + v) # This will broadcast 1-d vector (self.s + v) to 2-d vector self.x
        self.s += self.gamma * v


class Q_DPSGD_1(FederatedOptimizer):
    '''Differentially Private and Communication Efficient Collaborative Learning'''

    def __init__(self, p, epsilon=0.9, **kwargs):
        super().__init__(p, **kwargs)
        self.epsilon = epsilon

    def update(self):
        self.comm_rounds += 1
        
        z = self.compression_operator(self.x)
        sample_list = xp.random.randint(0, self.p.m, (self.p.n_agent, self.batch_size))
        grad = self.grad(self.x, j=sample_list)
        # Gradient clipping
        grad *= xp.clip(self.G / xp.linalg.norm(grad, axis=0), 0, 1)

        if self.perturbation_variance is not None:
            grad += self.perturbation_variance * xp.random.randn(*self.x.shape)

        self.x = (1 - self.epsilon + self.epsilon / self.p.n_agent) * self.x + self.epsilon * z.mean(axis=1, keepdims=True) - self.epsilon / self.p.n_agent * z - self.epsilon * self.eta * grad


class LDP_SGD(FederatedOptimizer):
    '''Private Non-Convex Federated Learning Without a Trusted Server'''

    def update(self):
        self.comm_rounds += 1

        sample_list = xp.random.randint(0, self.p.m, (self.p.n_agent, self.batch_size))
        grad = self.grad(self.x, j=sample_list)
        
        # Gradient clipping
        grad *= xp.clip(self.G / xp.linalg.norm(grad, axis=0), 0, 1)

        if self.perturbation_variance is not None:
            grad += self.perturbation_variance * xp.random.randn(*self.x.shape)

        self.x -= self.eta * grad.mean(axis=1, keepdims=True)

        
class LDP_SVRG(FederatedOptimizer):
    '''Private Non-Convex Federated Learning Without a Trusted Server'''

    def __init__(self, p, p_update_snapshot=None, **kwargs):
        super().__init__(p, **kwargs)
        self.p_update_snapshot = p_update_snapshot
    
    def init(self):
        super().init()
        self.w = self.x.copy()
        self.grad_w = self.grad(self.w)
        self.should_update_snapshot = xp.random.binomial(1, self.p_update_snapshot, self.n_iters + 1)

    def local_update(self):
        sample_list = xp.random.randint(0, self.p.m, (self.p.n_agent, self.batch_size))
        g_i = self.grad(self.x, j=sample_list) - self.grad(self.w, j=sample_list) + self.grad_w
        if self.should_update_snapshot[self.t]:
            self.w = self.x.copy()
            self.grad_w = self.grad(self.w)  # added snapshot gradient update
        return g_i

    def update(self):
        self.comm_rounds += 1

        # Agents
        g_i = self.local_update()
        
        # Gradient clipping
        g_i *= xp.clip(self.G / xp.linalg.norm(g_i, axis=0), 0, 1)

        if self.perturbation_variance is not None:
            g_i += self.perturbation_variance * xp.random.randn(*self.x.shape)

        # Server
        g = g_i.mean(axis=1, keepdims=True)
        self.x -= self.eta * g # This will broadcast 1-d vector (self.s + v) to 2-d vector self.x
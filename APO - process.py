

import numpy as np
from scipy.stats import multivariate_normal as normal
import math as ma

class Process(object):

    def __init__(self, process_config):
        self.dim = process_config.dim
        self.total_time = process_config.total_time
        self.num_time_interval = process_config.num_time_interval
        self.x_init_random = process_config.init_random
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)


    def sample(self, num_sample):
        """Process Sample - Euler-Maruyama"""
        raise NotImplementedError


class GBM1DMAR(Process):

    def __init__(self, process_config):
        super(GBM1DMAR, self).__init__(process_config)
        self.x_init = process_config.init
        self.miu = process_config.miu
        self.sigma = process_config.sigma

    def sample(self, num_sample):

        dw_sample = normal.rvs(size=[num_sample, self.num_time_interval+1])

        x_sample = np.zeros([num_sample, self.num_time_interval+1])
        if self.x_init_random == "False":

            x_sample[:, 0] = np.ones(num_sample) * self.x_init

        else:
            x_sample[:, 0] = self.x_init + dw_sample[:, 0]

        for i in range(self.num_time_interval):
            x_sample[:, i + 1] = x_sample[:, i] + self.miu*x_sample[:, i]*self.delta_t +\
                                 self.sigma*x_sample[:, i] * dw_sample[:, i] * self.sqrt_delta_t

        return x_sample


class GBM1DSOL(Process):

    def __init__(self, process_config):
        super(GBM1DSOL, self).__init__(process_config)
        self.x_init = process_config.init
        self.miu = process_config.miu
        self.sigma = process_config.sigma

    def sample(self, num_sample):

        dw_sample = normal.rvs(size=[num_sample, self.num_time_interval+1])

        x_sample = np.zeros([num_sample, self.num_time_interval+1])
        if self.x_init_random == "False":
            x_sample[:, 0] = np.ones(num_sample) * self.x_init
        else:

            x_sample[:, 0] = self.x_init + dw_sample[:, 0]

        for i in range(self.num_time_interval):

          x_sample[:, i+1] = x_sample[:, i] * np.exp((self.miu - 0.5 * self.sigma ** 2) * self.delta_t +
                                                     self.sigma * dw_sample[:, i] * self.sqrt_delta_t)

        return x_sample


class GBM1DMARNEU(Process):

    def __init__(self, process_config):
        super(GBM1DMARNEU, self).__init__(process_config)
        self.x_init = process_config.init
        self.miu = process_config.miu
        self.sigma = process_config.sigma
        self.r = process_config.interest_rate
        self.divyield = process_config.dividend_yield

    def sample(self, num_sample):

        dw_sample = normal.rvs(size=[num_sample, self.num_time_interval+1])

        x_sample = np.zeros([num_sample, self.num_time_interval+1])
        if self.x_init_random == "False":

            x_sample[:, 0] = np.ones(num_sample) * self.x_init

        else:
            x_sample[:, 0] = self.x_init + dw_sample[:, 0]

        for i in range(self.num_time_interval):
            x_sample[:, i + 1] = x_sample[:, i] + (self.r - self.divyield)*x_sample[:, i]*self.delta_t +\
                                 self.sigma*x_sample[:, i] * dw_sample[:, i] * self.sqrt_delta_t

        return x_sample

class GBM1DSOLNEU(Process):

    def __init__(self, process_config):
        super(GBM1DSOLNEU, self).__init__(process_config)
        self.x_init = process_config.init
        self.miu = process_config.miu
        self.sigma = process_config.sigma
        self.r = process_config.interest_rate
        self.divyield = process_config.dividend_yield

    def sample(self, num_sample):

        dw_sample = normal.rvs(size=[num_sample, self.num_time_interval+1])

        x_sample = np.zeros([num_sample, self.num_time_interval+1])
        if self.x_init_random == "False":
            x_sample[:, 0] = np.ones(num_sample) * self.x_init
        else:

            x_sample[:, 0] = self.x_init + dw_sample[:, 0]

        for i in range(self.num_time_interval):

          x_sample[:, i+1] = x_sample[:, i] * np.exp((self.r - self.divyield - 0.5 * self.sigma ** 2) * self.delta_t +
                                                     self.sigma * dw_sample[:, i] * self.sqrt_delta_t)

        return x_sample

class GBM1DSOLNEUMUL(Process):

        def __init__(self, process_config):
            super(GBM1DSOLNEUMUL, self).__init__(process_config)
            self.x_init = process_config.init
            self.miu = process_config.miu
            self.sigma = process_config.sigma
            self.r = process_config.interest_rate
            self.divyield = process_config.dividend_yield
            self.init_range = process_config.init_range

        def sample(self, num_sample):

            dw_sample = normal.rvs(size=[num_sample, self.num_time_interval + 1])

            x_sample = np.zeros([num_sample, self.num_time_interval + 1])

            if self.x_init_random == "False":
                x_sample[:, 0] = np.floor(np.random.uniform(self.init_range[0], self.init_range[1], num_sample))
            else:
                x_sample[:, 0] = np.random.uniform(self.init_range[0], self.init_range[1], num_sample) + dw_sample[:, 0]

            for i in range(self.num_time_interval):
                x_sample[:, i + 1] = x_sample[:, i] * np.exp(
                    (self.r - self.divyield - 0.5 * self.sigma ** 2) * self.delta_t +
                    self.sigma * dw_sample[:, i] * self.sqrt_delta_t)

            return x_sample

class ORNUHL1DMAR(Process):

    def __init__(self, process_config):
        super(ORNUHL1DMAR, self).__init__(process_config)
        self.x_init = process_config.init
        self.miu = process_config.miu
        self.sigma = process_config.sigma

    def sample(self, num_sample):

        dw_sample = normal.rvs(size=[num_sample, self.num_time_interval+1])

        x_sample = np.zeros([num_sample, self.num_time_interval+1])
        if self.x_init_random == "False":

            x_sample[:, 0] = np.ones(num_sample) * self.x_init

        else:
            x_sample[:, 0] = self.x_init + dw_sample[:, 0]

        for i in range(self.num_time_interval):
            x_sample[:, i + 1] = x_sample[:, i] - x_sample[:, i]*self.delta_t +\
                                 self.sigma*x_sample[:, i] * dw_sample[:, i] * self.sqrt_delta_t

        return x_sample

class WIENNER1D(Process):

    def __init__(self, process_config):
        super(WIENNER1D, self).__init__(process_config)
        self.x_init = process_config.init

    def sample(self, num_sample):

        dw_sample = normal.rvs(size=[num_sample, self.num_time_interval+1])

        x_sample = np.zeros([num_sample, self.num_time_interval+1])
        if self.x_init_random == "False":

            x_sample[:, 0] = np.ones(num_sample) * self.x_init

        else:

            x_sample[:, 0] = self.x_init + dw_sample[:, 0]

        for i in range(self.num_time_interval):

            x_sample[:, i + 1] = x_sample[:, i] + dw_sample[:, i] * self.sqrt_delta_t

        return x_sample

class SHIPRAND1D(Process):

    def __init__(self, process_config):
        super(SHIPRAND1D, self).__init__(process_config)
        self.x_init = process_config.init
        self.mu = process_config.miu
        self.sigma = process_config.sigma
        self.init_range = process_config.init_range
    def sample(self, num_sample):

        dw_sample = normal.rvs(size=[num_sample, self.num_time_interval + 1])

        x_sample = np.zeros([num_sample, self.num_time_interval + 1], dtype='float64')

        if self.x_init_random == "False":
            x_sample[:, 0] = np.round(np.random.uniform(self.init_range[0], self.init_range[1], num_sample), decimals=2)
        else:
            x_sample[:, 0] = np.round(np.random.uniform(self.init_range[0], self.init_range[1], num_sample)) + dw_sample[:, 0]

        for i in range(self.num_time_interval):
            for j in range(num_sample):
                x_sample[j, i + 1] = (self.mu/self.sigma)*x_sample[j, i] - \
                                 (self.mu/self.sigma)*np.square(x_sample[j, i])*dw_sample[j, i] * self.sqrt_delta_t

        return x_sample
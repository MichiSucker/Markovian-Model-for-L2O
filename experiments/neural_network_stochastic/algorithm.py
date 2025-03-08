import torch
import torch.nn as nn


class StochasticNnOptimizer(nn.Module):

    def __init__(self, batch_size: int):
        super(StochasticNnOptimizer, self).__init__()

        in_size = 3
        h_size = 15
        out_size = 2
        self.update_layer = nn.Sequential(
            nn.Conv2d(in_size, 1 * h_size, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1 * h_size, 1 * h_size, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1 * h_size, out_size, kernel_size=1),
        )

        in_size = 4
        h_size = 15
        out_size = 4
        self.weighting = nn.Sequential(
            nn.Linear(in_size, 1 * h_size),
            nn.ReLU(),
            nn.Linear(1 * h_size, 1 * h_size),
            nn.ReLU(),
            nn.Linear(1 * h_size, out_size),
        )

        # For stability
        self.eps = torch.tensor(1e-8).float()
        self.beta_1 = torch.tensor(0.9)
        self.beta_2 = torch.tensor(0.999)
        self.gamma = torch.tensor(0.001)
        self.batch_size = batch_size

    def forward(self, optimization_algorithm):

        # Compute gradient and normalize
        grad, loss = optimization_algorithm.loss_function.compute_stochastic_gradient(
            optimization_algorithm.current_state[-1], batch_size=self.batch_size, return_loss=True)
        t = optimization_algorithm.iteration_counter

        # Update momentum
        m_t_old = optimization_algorithm.current_state[0].detach().clone()
        v_t_old = optimization_algorithm.current_state[1].detach().clone()

        m_t = self.beta_1 * m_t_old + (1 - self.beta_1) * grad
        v_t = self.beta_2 * v_t_old + (1 - self.beta_2) * (grad ** 2)

        # Update state
        optimization_algorithm.current_state[0] = m_t
        optimization_algorithm.current_state[1] = v_t

        # Normalize
        m_t_hat = m_t / (1 - self.beta_1 ** t)
        v_t_hat = v_t / (1 - self.beta_2 ** t)

        m_t_hat_norm = torch.linalg.norm(m_t_hat)
        if m_t_hat_norm > 1e-16:
            m_t_hat = m_t_hat / m_t_hat_norm

        v_t_hat_norm = torch.linalg.norm(v_t_hat)
        if v_t_hat_norm > 1e-16:
            v_t_hat = v_t_hat / v_t_hat_norm

        w = self.weighting(torch.concat((
            torch.log(1 + m_t_hat_norm).reshape((1,)),
            torch.log(1 + v_t_hat_norm).reshape((1,)),
            torch.log(1 + torch.tensor(t)).reshape((1,)),
            torch.log(1 + loss).reshape((1,)),
        )))

        p = self.update_layer(torch.concat((
            w[0] * m_t_hat.reshape((1, 1, 1, -1)),
            w[1] * v_t_hat.reshape((1, 1, 1, -1)),
            w[2] * (m_t_hat * v_t_hat).reshape((1, 1, 1, -1)),
        ), dim=1)).reshape((2, -1))

        new = optimization_algorithm.current_state[-1] - w[-1] * self.gamma * p[0] * m_t_hat / (
                0.001 * torch.abs(p[1]) * v_t_hat ** 0.5 + self.eps)

        return new

    @staticmethod
    def update_state(opt_algo):
        opt_algo.current_state[0] = opt_algo.current_state[0].detach().clone()
        opt_algo.current_state[1] = opt_algo.current_state[1].detach().clone()
        opt_algo.current_state[2] = opt_algo.current_iterate.detach().clone()

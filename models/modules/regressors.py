from typing import Optional
import torchquantum as tq
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import numpy as np

class RidgeRegressor(nn.Module):
    def __init__(self, lambda_init: Optional[float] =0.):
        super().__init__()
        self._lambda = nn.Parameter(torch.as_tensor(lambda_init, dtype=torch.float))

    def forward(self, reprs: Tensor, x: Tensor, reg_coeff: Optional[float] = None) -> Tensor:
        if reg_coeff is None:
            reg_coeff = self.reg_coeff()
        w, b = self.get_weights(reprs, x, reg_coeff)
        return w, b

    def get_weights(self, X: Tensor, Y: Tensor, reg_coeff: float) -> Tensor:
        batch_size, n_samples, n_dim = X.shape
        ones = torch.ones(batch_size, n_samples, 1, device=X.device)
        X = torch.concat([X, ones], dim=-1)

        if n_samples >= n_dim:
            # standard
            A = torch.bmm(X.mT, X)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            B = torch.bmm(X.mT, Y)

            weights = torch.linalg.solve(A, B)

        else:
            # Woodbury
            A = torch.bmm(X, X.mT)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            weights = torch.bmm(X.mT, torch.linalg.solve(A, Y))

        return weights[:, :-1], weights[:, -1:]

    def reg_coeff(self) -> Tensor:
        return F.softplus(self._lambda)


class EchoStateNetworkv2(nn.Module):


    def __init__(self, input_size,  reservoir_size, spectral_radius=0.9, leaking_rate=0.3, input_scaling=1.0):
        super().__init__()

        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate
        self.input_scaling = input_scaling
        self.input_size = input_size

        self.ridgeregressor = RidgeRegressor()

        self.W_res = torch.rand( reservoir_size, reservoir_size).cuda() - 0.5
        self.W_res *= spectral_radius / torch.max(torch.abs(torch.linalg.eigvals(self.W_res)))
        self.W_in = torch.rand(input_size, reservoir_size).cuda() - 0.5
        self.W_in *= input_scaling


    def forward(self, input_data, target_data):
        reservoir_states = self.run_reservoir(input_data)
        w,b = self.ridgeregressor(reservoir_states, target_data)
        return w,b

    def predict(self, input_data):
        reservoir_states = self.run_reservoir(input_data)
        return reservoir_states

    def run_reservoir(self, input_data):

        windows_size = 8
        overlap_size = 2
        out_size = windows_size-2*overlap_size

        padding1 = torch.zeros((input_data.shape[0], overlap_size,input_data.shape[-1])).cuda()
        x = torch.cat([padding1, input_data],dim=1)
        x = torch.cat([x, padding1],dim=1)
        x = x.unfold(1, windows_size, out_size).transpose(2,3)
        reservoir_states = torch.zeros((x.shape[0], x.shape[1], windows_size-overlap_size, self.reservoir_size)).cuda()

        for t in range(0, windows_size-overlap_size):

            reservoir_states[:, :, t, :] = (1 - self.leaking_rate) * reservoir_states[:, :, t - 1, :] + \
                                     self.leaking_rate * torch.sin(
                torch.matmul(reservoir_states[:, :, t - 1, :],self.W_res) +
                torch.matmul(x[:,:,t,:],self.W_in),
            ).squeeze()

        reservoir_states = reservoir_states[:,:,overlap_size:,:].reshape(input_data.shape[0],input_data.shape[1],self.reservoir_size)
        return reservoir_states
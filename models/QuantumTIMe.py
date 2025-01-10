import gin
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange, repeat
from models.modules.qinr import QINR
from models.modules.regressors import EchoStateNetworkv2


@gin.configurable()
def quantumtime(in_feats: int,time_size: int, layers: int, layer_size: int, n_blocks: int):
    return QuantumTime(in_feats, time_size, layers, layer_size, n_blocks)

class QuantumTime(nn.Module):
    def __init__(self, in_feats: int, time_size: int, layers: int, layer_size: int, n_blocks: int):
        super().__init__()

        self.qinr = QINR(in_feats=in_feats+1, time_size= time_size, layers=layers, n_wires=int(layer_size/4), n_blocks=n_blocks)
        self.adaptive_weights = EchoStateNetworkv2(layer_size,256,0.9,0.3,0.5)


    def forward(self, x: Tensor, x_time: Tensor, y_time: Tensor) -> Tensor:
        tgt_horizon_len = y_time.shape[1]
        batch_size, lookback_len, _ = x.shape
        coords = self.get_coords(lookback_len, tgt_horizon_len).to(x.device)
        if y_time.shape[-1] != 0:
            time = torch.cat([x_time, y_time], dim=1)
            coords = repeat(coords, '1 t 1 -> b t 1', b=time.shape[0])
            coords = torch.cat([coords, time], dim=-1)

            time_reprs = self.qinr(coords)
        else:

            time_reprs = repeat(self.qinr(coords), '1 t d -> b t d', b=batch_size)


        lookback_reprs = time_reprs[:, :-tgt_horizon_len]
        horizon_reprs = time_reprs[:, -tgt_horizon_len:]
        w, b = self.adaptive_weights(lookback_reprs, x)
        h = self.adaptive_weights.predict(horizon_reprs)
        preds = self.forecast(h, w, b)

        return preds

    def forecast(self, inp: Tensor, w: Tensor, b: Tensor) -> Tensor:
        return torch.einsum('... d o, ... t d -> ... t o', [w, inp]) + b

    def get_coords(self, lookback_len: int, horizon_len: int) -> Tensor:
        coords = torch.linspace(0, 1, lookback_len + horizon_len)
        return rearrange(coords, 't -> 1 t 1')

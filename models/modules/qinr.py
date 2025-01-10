import torch
import torch.nn as nn
from torch import Tensor
import torchquantum as tq

class QINRLayer(nn.Module):
    def __init__(self, n_wires: int,  n_blocks: int, time_size: int):
        super().__init__()
        self.n_wires = n_wires
        self.n_blocks = n_blocks
        self.norm = nn.BatchNorm1d(time_size)

        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires, n_blocks):
                super().__init__()
                self.n_wires = n_wires
                self.n_blocks = n_blocks
                self.measurez = tq.MeasureAll(tq.PauliZ)

                encoder = []
                index = 0
                for i in range(n_wires):
                    encoder.append({'input_idx': [index], 'func': 'rx', 'wires': [i]})
                    index += 1
                    encoder.append({'input_idx': [index], 'func': 'ry', 'wires': [i]})
                    index += 1
                    encoder.append({'input_idx': [index], 'func': 'rz', 'wires': [i]})
                    index += 1
                    encoder.append({'input_idx': [index], 'func': 'rx', 'wires': [i]})
                    index += 1


                self.encoder = tq.GeneralEncoder(encoder)

                self.rz1_layers = tq.QuantumModuleList()
                self.ry1_layers = tq.QuantumModuleList()
                self.rz2_layers = tq.QuantumModuleList()
                self.rz3_layers = tq.QuantumModuleList()
                self.cnot_layers = tq.QuantumModuleList()


                for _ in range(n_blocks):
                    self.rz1_layers.append(
                        tq.Op1QAllLayer(
                            op=tq.RZ,
                            n_wires=n_wires,
                            has_params=True,
                            trainable=True,
                        )
                    )
                    self.ry1_layers.append(
                        tq.Op1QAllLayer(
                                op=tq.RY,
                            n_wires=n_wires,
                            has_params=True,
                            trainable=True,
                        )
                    )
                    self.rz2_layers.append(
                        tq.Op1QAllLayer(
                            op=tq.RZ,
                            n_wires=n_wires,
                            has_params=True,
                            trainable=False,
                        )
                    )
                    self.rz3_layers.append(
                        tq.Op1QAllLayer(
                            op=tq.RZ,
                            n_wires=n_wires,
                            has_params=True,
                            trainable=False,
                        )
                    )
                    self.cnot_layers.append(
                        tq.Op2QAllLayer(
                            op=tq.CNOT,
                            n_wires=n_wires,
                            has_params=False,
                            trainable=False,
                            circular=True,
                        )
                    )

            def forward(self, x):
                x = x.squeeze()
                qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)

                for k in range(self.n_blocks):
                    self.rz1_layers[k](qdev)
                    self.ry1_layers[k](qdev)
                    self.rz2_layers[k](qdev)
                    self.cnot_layers[k](qdev)

                    if k != self.n_blocks - 1:
                        self.encoder(qdev, x)


                out = [self.measurez(qdev).unsqueeze(0)]

                for i in range(self.n_wires):
                    qdev.h(wires=i)
                out.append(self.measurez(qdev).unsqueeze(0))

                for i in range(self.n_wires):
                    qdev.h(wires=i)
                    qdev.sx(wires=i)
                out.append(self.measurez(qdev).unsqueeze(0))

                self.rz3_layers[-1](qdev)
                out.append(self.measurez(qdev).unsqueeze(0))

                out = torch.cat(out, dim=-1)
                return out

        self.QNN = QLayer(self.n_wires, self.n_blocks)
        self.QLinear = nn.Linear(self.n_wires*4, self.n_wires*4)
        self.CLinear = nn.Linear(self.n_wires*4,self.n_wires*4)
        self.linear = nn.Linear(self.n_wires*8,self.n_wires*4)

    def forward(self, x):

        x1 = self._qlayer(x)
        x2 =  self._clayer(x)
        out = self.linear(torch.cat([x1, x2],dim=-1))
        return out
    def _qlayer(self, x: Tensor) -> Tensor:
        return self.QNN(self.norm(self.QLinear(x)))
    def _clayer(self, x: Tensor) -> Tensor:
        return torch.relu(self.CLinear(x))


class QINR(nn.Module):
    def __init__(self, in_feats:int, time_size: int, layers: int, n_wires: int, n_blocks: int):
        super().__init__()
        clayers = [QINRLayer(n_wires, n_blocks, time_size) for _ in range(layers)]
        self.layers = nn.Sequential(*clayers)
        self.features = nn.Linear(in_feats, n_wires*4)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.layers(x)




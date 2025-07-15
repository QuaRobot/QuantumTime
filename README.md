# QuantumTime

This repository provides the official implementation of our paper,

> **Quantum Time-index Models with Reservoir for Time Series Forecasting**  
> *To appear at the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2025)*

---

We propose **Quantum Time-index Models with Reservoir (QuantumTime)** — a **quantum-classical hybrid model** for time series forecasting.

- 🧠 **Quantum machine learning module**: We integrate **variational quantum circuits (VQCs)** into the architecture to directly model the high-frequency components of time series. This quantum module acts as an implicit neural representation, enabling compact and expressive feature learning with significantly fewer parameters.

- 🔁 **Classical reservoir computing**: To capture sequential dependencies and improve extrapolation, we embed a classical **reservoir** module that provides dynamical memory and rich nonlinearity.

Our model builds upon and extends the framework proposed in the ICML 2023 paper:

Gerald Woo, Chenghao Liu, Doyen Sahoo, Akshat Kumar, and Steven C. H. Hoi.  
*Learning Deep Time-index Models for Time Series Forecasting.*  
ICML 2023, Proceedings of Machine Learning Research, Vol. 202.  
[https://proceedings.mlr.press/v202/woo23b.html](https://proceedings.mlr.press/v202/woo23b.html)

We also draw inspiration from the recent work on quantum implicit neural representations:

Jiaming Zhao, Wenbo Qiao, Peng Zhang, and Hui Gao.  
*Quantum Implicit Neural Representations.*  
Forty-first International Conference on Machine Learning (ICML 2024).  
[https://openreview.net/forum?id=50vc4HBuKU](https://openreview.net/forum?id=50vc4HBuKU)

## ▶️ Run the Forecasting Model

You can train and evaluate the model using the following command:

```bash
python -m experiments.forecast --config_path=experiments/configs/'dataset_name'/'config_file' run

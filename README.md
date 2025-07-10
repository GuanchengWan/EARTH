# Epidemiology-Aware Neural ODE with Continuous Disease Transmission Graph

>Guancheng Wan, Zewen Liu, Xiaojun Shan, Max S.Y. Lau, B. Aditya Prakash, Wei Jin

## Abstract

Epidemiological forecasting is crucial for public health decision-making, yet existing approaches often struggle with the complex spatial-temporal dependencies inherent in disease transmission processes. Traditional compartmental models, while interpretable, lack the flexibility to capture real-world complexities, whereas deep learning methods often ignore the underlying epidemiological principles. To address these limitations, we propose EARTH (Epidemiology-Aware Neural ODE with Continuous Disease Transmission Graph), a novel framework that seamlessly integrates epidemiological domain knowledge with neural ordinary differential equations (ODEs). Our approach models disease transmission as a continuous dynamical system over a learnable graph structure, where nodes represent geographical regions and edges capture disease transmission relationships. The key innovation lies in our epidemiology-aware neural ODE formulation, which incorporates domain-specific constraints and inductive biases while maintaining the flexibility of deep learning. We introduce a continuous disease transmission graph that adaptively learns spatial dependencies and temporal dynamics simultaneously. Extensive experiments on real-world epidemiological datasets demonstrate that EARTH significantly outperforms state-of-the-art methods in both short-term and long-term forecasting tasks, while providing interpretable insights into disease transmission patterns.

## Citation

```latex
@inproceedings{Wan_EpiODE_ICML25,
  title={Epidemiology-Aware Neural ODE with Continuous Disease Transmission Graph},
  author={Wan, Guancheng and Liu, Zewen and Shan, Xiaojun and Lau, Max S.Y. and Prakash, B. Aditya and Jin, Wei},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```
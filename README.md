#  FCHEV-ACC-EMS: Integrated velocity optimization and energy management for FCHEV: An eco-driving approach based on deep reinforcement learning.
## Overview

The original implementation of **Integrated velocity optimization and energy management for FCHEV: An eco-driving approach based on deep reinforcement learning.**


## Abstract

Ecological driving (eco-driving) is a promising technology for transportation sector to save energy and reduce emission, which works by improving vehicle behaviors in traffic scenarios. Fuel cell hybrid electric vehicles (FCHEV) are receiving extensive attentions due to global fossil energy crisis, but whose implementations for eco-driving result in multiple objective collaborative optimization problems. In this paper, an eco-driving framework for FCHEV is proposed based on deep deterministic policy gradient (DDPG) algorithm. And it combines adaptive cruise control (ACC) and energy management strategy (EMS) into an integrated architecture. Firstly, in order to achieve excellent balance between driving behaviors and fuel economy, an appropriate weight coefficient value is determined after adequate explorations. Secondly, power-varying equivalent hydrogen conversion coefficient function is constructed to save fuel consumption by 8.97%. Thirdly, ablation experiments for health state of fuel cell system present 19.95% decrease in terms of health degradation. Then, comparison experiments indicate that the DDPG-based eco-driving strategy can reach 94.16% of that of dynamic programming with respect to equivalent hydrogen consumption, meanwhile with best ride comfortability. Moreover, simulation results under validation driving cycle manifest its excellent adaptability.


## Data

1. **Driving Cycles can be found [here](https://github.com/sicilyala/project-data/tree/main/standard_driving_cycles).**

2. **Power system data can be found [here](https://github.com/sicilyala/project-data/tree/main/FCHEV_data).**


## Citation
**BibTex**
```
@article{chen2023integrated,
  title={Integrated velocity optimization and energy management for FCHEV: An eco-driving approach based on deep reinforcement learning},
  author={Chen, Weiqi and Peng, Jiankun and Ren, Tinghui and Zhang, Hailong and He, Hongwen and Ma, Chunye},
  journal={Energy Conversion and Management},
  volume={296},
  pages={117685},
  year={2023},
  publisher={Elsevier},
  url={https://www.sciencedirect.com/science/article/abs/pii/S0196890423010312}
}
```

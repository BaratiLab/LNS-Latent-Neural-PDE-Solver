# Latent Neural PDE Solver

Code for *"Latent Neural PDE Solver: a reduced-order modelling framework for partial differential equations"*  [paper](https://www.sciencedirect.com/science/article/pii/S0021999124009537), [arxiv](https://arxiv.org/abs/2402.17853).

<div style style=”line-height: 20%” align="center">
<h3> Models prediction on different systems  </h3>
<h4> Shallow water equation  </h4>

<img src="https://github.com/BaratiLab/LNS-Latent-Neural-PDE-Solver/blob/main/assets/shallow_water.gif" width="600">

<h4> Two-phase flow (tank sloshing)  </h4>

<img src="https://github.com/BaratiLab/LNS-Latent-Neural-PDE-Solver/blob/main/assets/twophase_varying_freq.gif" width="600">

</div>


### Environment setup

We provide a yaml file for setting up conda environment:
```conda create env -f environment.yml```

### Datasets

The shallow water equation dataset is from [PDEArena](https://github.com/pdearena/pdearena), which can be downloaded here: https://huggingface.co/datasets/pdearena/ShallowWater-2D/tree/main.

The 2D Navier-Stokes / Two-phase flow data can be downloaded from Huggingface.

* NS2d: https://huggingface.co/datasets/JleeOfficial/Imcompressible_flow_2d/tree/main
* Two-phase flow (varying liquid height): https://huggingface.co/datasets/JleeOfficial/Tank_sloshing_varying_height/tree/main
* Two-phase flow (varying oscillation frequency): https://huggingface.co/datasets/JleeOfficial/Tank_sloshing_varying_freq

### Experiments

Please refer to the yaml file in each experiment for detailed hyperparameter settings.

#### 1. NS2d
Stage 1 autoencoder training:
  
```python train_stage1_ns2d.py --config configs/ns2d_stage1_ae.yml```

Stage 2 dynamics propagator training:

```python train_stage2_ns2d.py --config configs/ns2d_stage2_prop.yml```

#### 2. Shallow water
Stage 1:

```python train_stage1_SW.py --config configs/SW_stage1_ae.yml```

Stage 2:

```python train_stage2_SW.py --config configs/SW_stage2_prop.yml```

#### 3. Two-phase flow
Stage 1:

```python train_stage1_twophase.py --config configs/twophase_stage1_ae.yml```

Stage 2:

```python train_stage2_twophase.py --config configs/twophase_stage2_prop.yml```

Stage 2 (conditional prediction based on oscillation frequency):

```python train_stage2_twophase_conditional.py --config configs/twophase_stage2_cond_prop.yml```

### Citation

If you find this project helpful, please consider citing our work:
```
@article{LI2025LNS,
title = {Latent Neural PDE Solver: a reduced-order modelling framework for partial differential equations},
journal = {Journal of Computational Physics},
pages = {113705},
year = {2025},
issn = {0021-9991},
doi = {https://doi.org/10.1016/j.jcp.2024.113705},
url = {https://www.sciencedirect.com/science/article/pii/S0021999124009537},
author = {Zijie Li and Saurabh Patil and Francis Ogoke and Dule Shu and Wilson Zhen and Michael Schneier and John R. Buchanan and Amir Barati Farimani},
abstract = {Neural networks have shown promising potential in accelerating the numerical simulation of systems governed by partial differential equations (PDEs). Different from many existing neural network surrogates operating on high-dimensional discretized fields, we propose to learn the dynamics of the system in the latent space with much coarser discretizations. In our proposed framework - Latent Neural PDE Solver (LNS), a non-linear autoencoder is first trained to project the full-order representation of the system onto the mesh-reduced space, then a temporal model is trained to predict the future state in this mesh-reduced space. This reduction process simplifies the training of the temporal model by greatly reducing the computational cost accompanying a fine discretization and enables more efficient backprop-through-time training. We study the capability of the proposed framework and several other popular neural PDE solvers on various types of systems including single-phase and multi-phase flows along with varying system parameters. We showcase that it has competitive accuracy and efficiency compared to the neural PDE solver that operates on full-order space.}
}
```

### Acknowledgement

Modern UNet implementation and conditioning for FNO block: [PDEArena](https://github.com/pdearena/pdearena).




  

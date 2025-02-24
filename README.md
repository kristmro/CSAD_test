# CSAD Testing Repository

The **CSAD Testing Repository** is a versatile simulation platform designed for advanced control research on the **C/S Arctic Drillship (CSAD)** model from Marine Cybernetics, used at NTNU. Built on McSimPython, this repository provides a dynamic testing ground for developing and evaluating adaptive control algorithms(RL, MRAC, MPC, etc.). It also supports meta-training, enabling the creation of robust, generalizable control systems tailored for complex maritime environments.

- **Adaptive Controllers:**  
  The repository includes controllers designed for adaptive control of marine vessels. The meta-trained adaptive controller approach draws inspiration from the work in [Adaptive-Control-Oriented-Meta-Learning](https://github.com/StanfordASL/Adaptive-Control-Oriented-Meta-Learning).

- **Model-Based Reinforcement Learning (MBRL):**  
  Controllers that leverage model predictions and are inspired by approaches described in [this NTNU paper](https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2687805/frobt-07-00032+%284%29.pdf?sequence=2). The aim is to meta-train this controller to rapidly adapt under varying wave conditions.

A key aspect of the project is the development of two distinct simulation pipelines:

1. **CSADtesting Pipeline:**  
   Built on McSimPython and numpy, this pipeline provides a robust simulation environment for marine vessel dynamics. It includes features such as grid-based domains, dynamic goals, obstacles, real-time rendering via pygame, and post-simulation plotting with matplotlib.

2. **DiffSim Pipeline:**  
   A PyTorch-based reimplementation that mirrors CSADtesting but supports full differentiability. This enables gradient-based meta-learning by allowing backpropagation through the entire simulation process.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Simulation Pipelines](#simulation-pipelines)
  - [CSADtesting](#csadtesting)
  - [DiffSim](#diffsim)
- [Repository Structure](#repository-structure)
- [Key Modules and Components](#key-modules-and-components)
  - [Environment](#environment)
  - [Controllers](#controllers)
  - [Allocation](#allocation)
  - [Filters](#filters)
  - [Meta-Learning](#meta-learning)
  - [Simulator and Utilities](#simulator-and-utilities)
- [Offline Data and Online Adaptation](#offline-data-and-online-adaptation)
- [Contributions](#contributions)
- [License](#license)

---

## Project Overview

This repository is part of a master’s thesis project titled **Meta-Learning for Adaptive Motion Control of Marine Vessels**. It aims to create a versatile simulation environment that can be used to meta-train different control strategies for marine vessels, notably the **C/S Arctic Drillship**. The platform is designed to:

- **Meta-train Adaptive Controllers:**  
  Develop controllers that rapidly adapt to changing environmental conditions. The meta-training approach is inspired by the work in [Adaptive-Control-Oriented-Meta-Learning](https://github.com/StanfordASL/Adaptive-Control-Oriented-Meta-Learning).

- **Meta-train MBRL Controllers:**  
  Utilize model-based reinforcement learning techniques—drawing inspiration from established literature—to adapt to dynamic disturbances such as wave-induced forces.

The repository supports both traditional simulation (using McSimPython & numpy) and a differentiable simulation pipeline (using PyTorch) to enable end-to-end gradient-based optimization and meta-learning.

---

## Simulation Pipelines

### CSADtesting

- **Description:**  
  The CSADtesting pipeline uses McSimPython and numpy to simulate a 6DOF vessel. It includes a grid-based environment that handles wave loads, obstacles, dynamic goals, and specialized tasks (e.g., a four-corner test).  
- **Key Features:**  
  - **Real-time Rendering:**  
    Visualization using pygame.
  - **Trajectory Plotting:**  
    Post-simulation analysis with matplotlib.
  - **Control Implementations:**  
    Multiple controllers including baseline adaptive controllers (e.g., in `adaptive_seakeeping.py` and `adaptiveFScontroller.py`) and a prototype MBRL controller.
  - **Thruster Allocation:**  
    Implements a PseudoInverseAllocator to map global control actions to individual thruster commands.
  - **Reference Filtering:**  
    Uses a third-order reference filter (`reference_filter.py`) to generate smooth desired trajectories.

### DiffSim

- **Description:**  
  The DiffSim pipeline reimplements the core simulation functionality in PyTorch. It replicates the CSADtesting functionality but uses differentiable modules to allow backpropagation through the simulation—a key enabler for meta-learning.
- **Key Features:**  
  - **Differentiable Vessel Simulation:**  
    Implemented in `DiffCsad.py` and `DiffVessel.py` using torch.Tensors.
  - **Differentiable Wave Modeling:**  
    Modules such as `DiffWaveLoad.py` and `DiffWaveSpectra.py` calculate wave loads and spectra in a fully differentiable manner.
  - **Differentiable Controllers and Allocation:**  
    PyTorch-based controllers (e.g., `DiffModelController.py`) and allocation/dynamics modules (e.g., in the `Allocation` directory) ensure full compatibility with gradient-based optimization.
  - **Utility Functions:**  
    Provided in `DiffUtils.py` for operations like rotations and differentiable interpolation.

---

## Repository Structure

```
CSAD_test/
├── CSADtesting/         # Core simulator and control implementations (using McSimPython & numpy)
│   ├── Controller/      
│   │   ├── __init__.py
│   │   ├── adaptiveFScontroller.py      # Baseline adaptive controller implementation
│   │   ├── adaptive_seakeeping.py         # MRAC-based heading controller combined with surge PID
│   │   └── mbrl_controller.py            # Prototype MBRL controller
│   ├── Environment/
│   │   ├── __init__.py
│   │   └── GridBoatEnv.py                # Grid-based environment with wave loads and dynamic tasks
│   ├── Main_test/      
│   │   ├── __init__.py
│   │   ├── ExMcEKF.py                   # Example Extended Kalman Filter run (if applicable)
│   │   ├── GridMainAdaptive.py          # Simulation entry using adaptive controller with thruster allocation
│   │   └── Gridmain.py                  # Standard simulation entry using MRAC heading + surge PID
│   ├── allocation/      
│   │   ├── __init__.py
│   │   └── allocation.py                 # Thruster allocation (PseudoInverseAllocator)
│   ├── filters/
│   │   ├── __init__.py
│   │   └── reference_filter.py           # Third-order reference filter (ThrdOrderRefFilter)
│   ├── metalearning/     
│   │   ├── __init__.py
│   │   ├── FAMLE.py
│   │   └── Reptile.py
│   └── __init__.py
├── DiffSim/                # PyTorch-based reimplementation of simulator and controllers
│   ├── Allocation/
│   │   ├── __init__.py
│   │   ├── DiffAllocation.py
│   │   ├── DiffThruster.py
│   │   └── DiffThrusterDynamics.py
│   ├── Controller/
│   │   ├── __init__.py
│   │   └── DiffModelController.py       # Simple model-based PD controller for testing
│   ├── DataGen/
│   │   ├── __init__.py
│   │   └── DiffDataGen.py                # (Under development) Data generation for training
│   ├── Environment/
│   │   ├── __init__.py
│   │   └── DiffGridBoatEnv.py            # Differentiable grid-based environment (PyTorch)
│   ├── Filter/
│   │   ├── __init__.py
│   │   └── DiffRefFilter.py              # Differentiable reference filter
│   ├── Main/
│   │   ├── __init__.py
│   │   └── DiffGridMainPID.py            # Entry point using a PID controller in the differentiable environment
│   ├── Simulator/
│   │   ├── __init__.py
│   │   ├── DiffCsad.py                   # Differentiable 6DOF CSAD vessel model
│   │   ├── DiffVessel.py                 # Base differentiable vessel class
│   │   ├── DiffWaveLoad.py               # Differentiable wave load module
│   │   └── DiffWaveSpectra.py            # Differentiable wave spectra
│   ├── DiffUtils.py                      # Differentiable utility functions (e.g., rotations, interpolation)
│   └── __init__.py
├── Offline_data/           # Scripts for offline data generation and plotting
│   ├── __init__.py
│   ├── generate_data.py
│   ├── plott_generate_data.py
│   └── trajectory.py
├── Online_adap/            # Scripts for online adaptation experiments
│   ├── __init__.py
│   └── train_adaptive.py
├── .gitignore
├── LICENSE
└── README.md
```

---

## Key Modules and Components

### Environment

- **CSADtesting/Environment/GridBoatEnv.py:**  
  Defines a grid-based simulation environment for boat navigation. It integrates a 6DOF vessel simulator (via McSimPython), applies wave loads generated from a JONSWAP spectrum, and supports dynamic tasks (including a four-corner test). It provides real-time rendering (using pygame) and post-simulation trajectory plotting (with matplotlib), along with a third-order reference filter to generate desired trajectories.

- **DiffSim/Environment/DiffGridBoatEnv.py:**  
  A differentiable version of the grid-based environment implemented in PyTorch. It utilizes differentiable vessel dynamics, wave load models, and reference filtering—making it suitable for gradient-based meta-learning.

### Controllers

- **CSADtesting/Controller/adaptive_seakeeping.py:**  
  Combines a Model Reference Adaptive Controller (MRAC) for heading control with a surge PID controller for forward speed, yielding a 3DOF control action.

- **CSADtesting/Controller/adaptiveFScontroller.py:**  
  Implements one adaptive controller approach. It uses a Fourier series-based method to try to online find the load of the wave-induced forces and is a model-based backstepping controller. I have further modified it from [Brørby, 2022](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/3024626) to include some saturations and limitations.

- **CSADtesting/Controller/mbrl_controller.py:**  
  A prototype MBRL controller that leverages model predictions for adaptive control, inspired by established literature.

- **DiffSim/Controller/DiffModelController.py:**  
  A simple, differentiable PD controller was implemented in PyTorch for testing within the DiffSim pipeline.

### Allocation

- **CSADtesting/allocation/allocation.py:**  
  Provides functions for thruster allocation using a pseudo-inverse method to map global control commands to individual thruster actions based on vessel geometry.

- **DiffSim/Allocation:**  
  Contains differentiable implementations of thruster allocation and thruster dynamics (e.g., `DiffAllocation.py`, `DiffThruster.py`, `DiffThrusterDynamics.py`) to support gradient-based learning.

### Filters

- **CSADtesting/filters/reference_filter.py:**  
  Implements a third-order reference filter (ThrdOrderRefFilter) to generate smooth desired trajectories (position, velocity, acceleration) for guidance tasks.

- **DiffSim/Filter/DiffRefFilter.py:**  
  A PyTorch-based differentiable reference filter that integrates seamlessly with the DiffSim pipeline.

### Meta-Learning

- **CSADtesting/metalearning:**  
  OBS! This is just as reference and have not actually been implemented correctly. Contains implementations of meta-learning algorithms such as FAMLE (place holder) and Reptile, which aim to quickly adapt control parameters across varying environmental conditions. Future extensions may include methods like MAML and FOMAML.

### Simulator and Utilities

- **DiffSim/Simulator:**  
  - **DiffCsad.py:**  
    Implements a fully differentiable 6DOF vessel model for CSAD using PyTorch.
  - **DiffVessel.py:**  
    Provides a base class for differentiable vessel simulation.
  - **DiffWaveLoad.py & DiffWaveSpectra.py:**  
    Compute wave loads and spectra in a fully differentiable manner.
- **DiffSim/DiffUtils.py:**  
  Offers differentiable utility functions for operations such as rotations, DOF conversions (`three2sixDOF`, `six2threeDOF`), and minimal 1D interpolation (`torch_lininterp_1d`).

### Offline Data and Online Adaptation

- **Offline_data:**  
  Contains scripts for generating training data, plotting trajectories, and analyzing simulation outputs.
- **Online_adap:**  
  Includes scripts for online adaptation experiments (e.g., training adaptive controllers in real time).

---

## Contributions

Contributions, bug reports, and feature suggestions are welcome!  
If you wish to contribute, please open an issue or submit a pull request with your ideas or fixes.

---

## License

This repository is distributed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

*Note: Detailed “How to Run” instructions are omitted at this time, as the simulation environment is still under active development.*

---

This README provides a comprehensive overview of the CSAD Testing Repository, its simulation pipelines, file structure, and key components. As the project evolves, additional details and instructions will be added.

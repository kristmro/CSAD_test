# CSAD Testing Repository

OBS OBS only Gridmain.py-> GridBoatEnv.py is fully tested and correct. LS.py implements an adaptive control strategy with parameter estimation for a nonlinear pendulum, combining swing-up control and stabilizing LQR control to bring the pendulum to an upright position while estimating its dynamic parameters over time.

This repository serves as a testing ground for developing a reinforcement learning (RL) simulation platform tailored to the **C/S Arctic Drillship** at NTNU. The ultimate goal is to create a flexible platform that supports various adaptive motion control strategies for marine vessels, with a focus on meta-learned model-based reinforcement learning (MBRL). 

**Note:** This is not the final version but a prototype to test and refine the platform's design and functionality.

## Project Context

The project is part of a master's thesis aimed at achieving **Meta-Learning for Adaptive Motion Control of Marine Vessels**. The adaptive control strategies explored include:
- Model-Based Reinforcement Learning (MBRL)
- Adaptive Model Reference Control (AMRC)
- Adaptive Model Predictive Control (AMPC)
- Adaptive Linear Quadratic Regulators (ALQR)

The platform is designed to be flexible to accommodate these approaches, with the primary focus on implementing meta-learned MBRL.

### Meta-Learning Methods
The meta-learning methods being explored include:
- **FAMLE**: A method focusing on fast and adaptive meta-learning for control.
- **Reptile**: A simple and computationally efficient approach to meta-learning.
- **MAML**: Model-Agnostic Meta-Learning.
- **FOMAML**: A first-order approximation of MAML.

The initial focus is on implementing and testing **Reptile** and **FAMLE**, with plans to expand to additional methods.

---

## File Structure
```
CSADtesting

├── __init__.py
│
├── Controller
│   ├── __init__.py
│   ├── Random_shooting.py
│   ├── mbrl_controller.py
│
├── Environment
│   ├── __init__.py
│   ├── GridBoatEnv.py
│   ├── GridBoatGymEnv.py
│   ├── Ingen_controller_csad.py
│
├── Main_test
│   ├── __init__.py
│   ├── Gridmain.py
│
├── adaptiveAlgo
│   ├── __init__.py
│   ├── LS.py
│
├── metalearning
│   ├── __init__.py
│   ├── FAMLE.py
│   ├── Reptile.py
```

---

## File and Directory Descriptions

### `Controller/`
Contains scripts for implementing various controllers:
- **`Random_shooting.py`**: Implements a random shooting method for MBRL.
- **`mbrl_controller.py`**: A controller leveraging model-based reinforcement learning techniques.

### `Environment/`
Defines simulation environments:
- **`GridBoatEnv.py`**: A grid-based environment for boat navigation simulations.
- **`GridBoatGymEnv.py`**: A Gym-compatible version of the grid-based environment.
- **`Ingen_controller_csad.py`**: Environment-specific logic for control and dynamics.

### `Main_test/`
Scripts for testing and running simulations:
- **`Gridmain.py`**: Entry point for running simulations in the grid-based environment.

### `adaptiveAlgo/`
Contains adaptive algorithms used in RL training:
- **`LS.py`**: Likely implements a least-squares-based method for learning or optimization.

### `metalearning/`
Houses meta-learning algorithms:
- **`FAMLE.py`**: Implements the FAMLE meta-learning algorithm.
- **`Reptile.py`**: Implements the Reptile meta-learning algorithm.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/kristmro/CSAD_test.git
   cd CSAD_test/CSADtesting
   ```

2. Install necessary dependencies:
   ```bash
   pip install numpy matplotlib pygame MCSimPython
   ```

3. Run a simulation using `Gridmain.py`:
   ```bash
   python Main_test/Gridmain.py
   ```

---

## Future Plans
- Extend the platform to support the implementation of meta-learned MBRL for the **C/S Arctic Drillship**.
- Incorporate additional adaptive control algorithms (AMRC, AMPC, ALQR).
- Implement and benchmark additional meta-learning methods (MAML, FOMAML, etc.).
- Improve documentation and provide detailed examples.

---

## Contributions
Contributions are welcome! If you have suggestions, bug fixes, or additional features to propose, please open an issue or create a pull request.

---

## License
This repository is distributed under the MIT License. See `LICENSE` for details.





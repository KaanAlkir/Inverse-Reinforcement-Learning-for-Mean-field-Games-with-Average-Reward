# Linear inverse reinforcement learning for mean-field games

This repository contains a numerical illustration of the finite-dimensional
linear inverse reinforcement learning formulation for stationary mean-field games
under an average-reward criterion.

The implementation is based on a two-state malware spread model. It first
computes an expert stationary mean-field equilibrium, then forms the associated
expert feature expectation vector, solves the dual inverse problem, and finally
recovers a policy consistent with the expert behaviour.

## Repository structure

- `src/linear_malware.py`: implementation of the forward and inverse solvers
- `notebooks/`: notebook version with explanations and plots
- `assets/`: reference material, including the implemented paper

## Main idea

The notebook implements the finite-dimensional linear theory developed in the
paper included in the assets folder, specialized to a malware spread model
adapted from the stationary mean-field game literature.

## Requirements

The code uses:
- `numpy`
- `matplotlib`

If you want to run the notebook, you will also need:
- `jupyter`

## References

[1] Ş. K. Alkır and N. Saldı, “Inverse Reinforcement Learning for Mean-field Games with Average Reward Criterion,” *2025 IEEE 64th Conference on Decision and Control (CDC)*, pp. 7272–7277, 2025.

[2] J. Subramanian and A. Mahajan, “Reinforcement learning in stationary mean-field games,” *Proceedings of the 18th International Conference on Autonomous Agents and MultiAgent Systems (AAMAS)*, 2019.
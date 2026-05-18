# Inverse reinforcement learning for mean-field games

This repository contains numerical illustrations of inverse reinforcement learning
for stationary mean-field games under an average-reward criterion. The examples
are based on the finite-dimensional maximum entropy formulation developed in
[[1]](#ref1) and its numerical extensions.

The repository includes three examples:

1. a two-state linear malware spread model,
2. a ten-state linear malware spread model,
3. an RKHS-based inverse reinforcement learning example.

The linear examples implement the finite-dimensional maximum entropy inverse
reinforcement learning formulation. The RKHS example illustrates the kernel-based
extension, where the reward is represented through kernel features rather than a
fixed finite-dimensional feature vector.

In each example, the code first computes or fixes an expert stationary mean-field
equilibrium, forms the corresponding expert statistics, solves the inverse
problem, and compares the recovered policy with the expert policy.

## Repository structure

- `src/`: Python implementations of the forward and inverse solvers
- `notebooks/`: notebook versions with explanations, diagnostics, and plots
- `assets/`: reference material, figures, and paper-related files

The notebooks are organized as numerical demonstrations. They contain the main
explanations, parameter choices, plotted outputs, and diagnostic checks used to
interpret the experiments.

## Examples

### Two-state linear malware model

The two-state model is a minimal malware spread example inspired by the
stationary mean-field malware model in [[2]](#ref2). The states represent healthy and
infected agents, and the actions represent doing nothing or repairing the
system.

This example is useful because the full forward and inverse pipeline can be seen
in the simplest possible setting. The code computes an expert stationary
mean-field equilibrium, constructs the expert occupation measure and feature
expectation vector, solves the dual maximum entropy inverse problem, and recovers
a policy from the optimized occupation measure.

### Ten-state linear malware model

The ten-state model is a higher-dimensional malware spread example with state
space

$$
\mathcal X= \{ 0,0.1,\ldots,0.9 \} .
$$

Each state represents a malware severity level. The action space is
$\mathcal A=\{0,1\}$, where $0$ corresponds to doing nothing and $1$ corresponds
to repair.

The reward is linear in the feature vector

$$
\varphi(x,a,\mu)=\left(x, x\mu_{\mathrm{av}}, a\right),
$$

where $\mu_{\mathrm{av}}$ is the average malware severity in the population.
The notebook solves the forward mean-field game, computes the expert statistics,
solves the linear dual inverse problem, and compares the expert and recovered
repair probabilities across all severity levels.

### RKHS-based example

The RKHS example illustrates a kernel-based inverse reinforcement learning
procedure. Instead of using only a fixed finite-dimensional feature vector, the
reward is represented through kernel features.

The notebook constructs kernel features, computes expert statistics, runs the
RKHS-based inverse learning procedure, and reports the recovered policy together
with diagnostic quantities such as score values, gradient norms, policy errors,
and feature-moment errors.

## Main idea

The numerical experiments follow the same general pipeline.

First, an expert stationary mean-field equilibrium $(\pi_E,\mu_E)$ is obtained
from a forward mean-field game. The corresponding expert occupation measure is

$$
\nu_E(x,a)=\mu_E(x)\pi_E(a\mid x).
$$

Then the expert feature expectation is computed as

$$
\langle \varphi\rangle_E=\sum_{x,a}\nu_E(x,a)\varphi(x,a,\mu_E).
$$

The inverse problem uses these expert statistics to recover a policy that is
consistent with the expert behavior. In the linear examples, this is done by
solving the finite-dimensional dual maximum entropy inverse problem. In the
RKHS example, the same idea is implemented with kernel-based reward features.

## Requirements

The code uses:

- `numpy`
- `matplotlib`

To run the notebooks, you will also need:

- `jupyter`

## References

<a id="ref1"></a>
[1] Ş. K. Alkır and N. Saldı, “[Inverse Reinforcement Learning for Mean-field Games with Average Reward Criterion](https://doi.org/10.1109/CDC57313.2025.11312818),” *2025 IEEE 64th Conference on Decision and Control (CDC)*, pp. 7272–7277, 2025.

<a id="ref2"></a>
[2] J. Subramanian and A. Mahajan, “[Reinforcement learning in stationary mean-field games](https://www.ifaamas.org/Proceedings/aamas2019/pdfs/p251.pdf),” *Proceedings of the 18th International Conference on Autonomous Agents and MultiAgent Systems (AAMAS)*, 2019.

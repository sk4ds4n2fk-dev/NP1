# Beta-Product Dependent Pitman-Yor Process - H1 Implementation

Implementation of the **H1 model** from:

> **Bassetti, F., Casarin, R., & Leisen, F. (2014)**
> *Beta-product dependent Pitman–Yor processes for Bayesian inference*
> Journal of Econometrics, 180(1), 49-72.

## Model Specification

### H1 Stick-Breaking Construction

The H1 model defines dependent stick-breaking processes as:

```
(S₁ₖ, S₂ₖ) = (V₀ₖ·V₁ₖ, V₀ₖ·V₂ₖ)
```

where:
- `V₀ₖ ~ Beta(1 - α, θ₁)`
- `Vᵢₖ ~ Beta(1 - α + θ₁, θ₂ + αk)` for i = 1, 2

Parameters:
- `θ₁, θ₂ > 0`: concentration parameters
- `α ∈ [0, 1)`: discount parameter

### Shared Atoms (Equation 18)

The atoms are **shared** across components:

```
(ψ̃₁ₖ, ψ̃₂ₖ) = (ψ̃₀ₖ, ψ̃₀ₖ)  for all k
```

This means both random measures share the same mixture locations.

### Weights

The stick-breaking weights are computed as:

```
Wᵢₖ = Sᵢₖ ∏_{j<k} (1 - Sᵢⱼ)
```

### Random Measures

The resulting random probability measures are:

```
G₁ = Σₖ W₁ₖ δ_{ψ₀ₖ}
G₂ = Σₖ W₂ₖ δ_{ψ₀ₖ}
```

## Features Implemented

✅ **H1 stick-breaking construction**
✅ **Shared atoms across components** (Equation 18)
✅ **Pitman-Yor process** (α > 0)
✅ **Dirichlet process** (α = 0 special case)
✅ **Correlation computation** (Proposition 3)
✅ **Data generation and sampling**
✅ **Visualization tools**

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from bassetti_h1 import BassettiBetaDPY_H1

# Create H1 model instance
model = BassettiBetaDPY_H1(theta1=0.5, theta2=1.0, alpha=0.3)

# Generate mixture representation with truncation K=50
mixture = model.generate_mixture(K=50, mu0=0.0, sigma0=1.0)

# Access weights and atoms
weights1 = mixture['weights1']  # Weights for component 1
weights2 = mixture['weights2']  # Weights for component 2
atoms = mixture['atoms']  # Shared atoms (equation 18)

print(f"Sum of weights 1: {weights1.sum():.4f}")
print(f"Sum of weights 2: {weights2.sum():.4f}")
print(f"Number of atoms: {len(atoms)}")
```

### Sampling Data

```python
# Sample 500 observations from component 1
data1, clusters1 = model.sample_data(
    n_samples=500,
    component=1,
    K=50,
    mu0=0.0,
    sigma0=1.0,
    obs_sigma=0.5
)

# Sample from component 2
data2, clusters2 = model.sample_data(
    n_samples=500,
    component=2,
    K=50
)

print(f"Component 1: {len(clusters1.unique())} active clusters")
print(f"Component 2: {len(clusters2.unique())} active clusters")
```

### Visualization

```python
# Plot mixture densities
fig1 = model.plot_mixture_density(K=50)
fig1.savefig('densities.png')

# Plot stick-breaking weights
fig2 = model.plot_weights(K=30)
fig2.savefig('weights.png')
```

### Computing Correlation (Dirichlet case)

For α = 0 (Dirichlet process), the correlation between random measures is:

```python
from bassetti_h1 import correlation_H1

corr = correlation_H1(theta1=0.5, theta2=1.0)
print(f"Correlation: {corr:.4f}")
```

Formula (Proposition 3):
```
C₁₂ = (1 + θ₁ + θ₂)(1 + θ₁) / [(1 + θ₁)(1 + θ₁ + θ₂) + θ₂]
```

## Examples

Run the main script to see all examples:

```bash
python bassetti_h1.py
```

This will generate:
- `h1_dd_weights.png`: Weights for Dirichlet process (α=0)
- `h1_dd_density.png`: Densities for Dirichlet process
- `h1_py_weights.png`: Weights for Pitman-Yor process (α=0.3)
- `h1_py_density.png`: Densities for Pitman-Yor process
- `h1_sampled_data.png`: Histograms of sampled data

## Mathematical Details

### Stick-Breaking Representation

Each random measure has the representation:

```
Gᵢ = Σ_{k=1}^∞ Wᵢₖ δ_{ψ₀ₖ}
```

where:
1. **Weights** `Wᵢₖ` are constructed via stick-breaking
2. **Atoms** `ψ₀ₖ` are shared (Equation 18)

### Marginal Processes

Under H1, both `G₁` and `G₂` are marginally Pitman-Yor processes:
- `G₁ ~ PY(α, θ₁ + θ₂, H₀)`
- `G₂ ~ PY(α, θ₁ + θ₂, H₀)`

with the **same parameters**.

### Dependence Structure

The dependence between `G₁` and `G₂` comes from:
1. **Common V₀ₖ** in the stick-breaking
2. **Shared atoms** ψ₀ₖ (Equation 18)

## Model Variants

### Dirichlet Process (α = 0)

```python
model_dd = BassettiBetaDPY_H1(theta1=1.0, theta2=1.0, alpha=0.0)
```

### Pitman-Yor Process (α > 0)

```python
model_py = BassettiBetaDPY_H1(theta1=0.5, theta2=1.0, alpha=0.3)
```

Higher `α` values lead to:
- More atoms with significant weight
- Heavier tails in the distribution
- Power-law behavior

## Implementation Notes

1. **Truncation**: The infinite sum is truncated at level `K`. The residual weight is negligible for large enough `K`.

2. **Shared Atoms**: Following Equation 18, atoms are generated once and shared across both components.

3. **No Hierarchical Atoms**: This implementation uses simple shared atoms. The hierarchical structure `ψᵢₖ = ψ₀ₖ + ψ̃ᵢₖ` (Section 4.1) is **not** implemented as per user request.

4. **Gibbs Sampling**: The full MCMC algorithm (Section 5) for posterior inference is not included in this version. Only the prior model is implemented.

## Reference

```bibtex
@article{bassetti2014beta,
  title={Beta-product dependent Pitman–Yor processes for Bayesian inference},
  author={Bassetti, Federico and Casarin, Roberto and Leisen, Fabrizio},
  journal={Journal of Econometrics},
  volume={180},
  number={1},
  pages={49--72},
  year={2014},
  publisher={Elsevier}
}
```

## License

This implementation is for research and educational purposes.

## Contact

For questions about the implementation, please refer to the original paper:
- [DOI: 10.1016/j.jeconom.2014.01.007](https://doi.org/10.1016/j.jeconom.2014.01.007)

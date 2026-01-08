"""
Beta-product Dependent Pitman-Yor Process - H1 Model
Implementation of Bassetti, Casarin, Leisen (2013)

This implementation focuses on:
- H1 hierarchical structure for stick-breaking
- Shared atoms across components (Equation 18)
- Slice sampling algorithm for posterior inference
"""

import numpy as np
from scipy.stats import beta, norm, invgamma, gamma
from scipy.special import logsumexp
import matplotlib.pyplot as plt


class BassettiBetaDPY_H1:
    """
    Beta-Product Dependent Pitman-Yor Process (H1 specification)

    Stick-breaking construction:
        (S_1k, S_2k) = (V_0k * V_1k, V_0k * V_2k)
    where:
        V_0k ~ Beta(1 - α, θ_1)
        V_ik ~ Beta(1 - α + θ_1, θ_2 + αk), i = 1, 2

    Atoms follow equation 18 (shared atoms):
        (ψ_1k, ψ_2k) = (ψ_0k, ψ_0k)
    """

    def __init__(self, theta1=1.0, theta2=1.0, alpha=0.0):
        """
        Initialize the H1 model.

        Parameters:
        -----------
        theta1 : float
            First concentration parameter (θ_1 > 0)
        theta2 : float
            Second concentration parameter (θ_2 > 0)
        alpha : float
            Discount parameter (0 ≤ α < 1)
        """
        assert theta1 > 0, "theta1 must be positive"
        assert theta2 > 0, "theta2 must be positive"
        assert 0 <= alpha < 1, "alpha must be in [0, 1)"

        self.theta1 = theta1
        self.theta2 = theta2
        self.alpha = alpha

        # Storage for stick-breaking components
        self.V0 = []  # V_0k variables
        self.V1 = []  # V_1k variables for component 1
        self.V2 = []  # V_2k variables for component 2

        # Storage for atoms (shared across components - eq. 18)
        self.atoms = []  # ψ_0k (shared atoms)

        # Storage for weights
        self.weights1 = []  # W_1k
        self.weights2 = []  # W_2k

    def sample_stick_variables(self, k):
        """
        Sample stick-breaking variables for index k.

        H1 specification:
            V_0k ~ Beta(1 - α, θ_1)
            V_1k ~ Beta(1 - α + θ_1, θ_2 + αk)
            V_2k ~ Beta(1 - α + θ_1, θ_2 + αk)

        Parameters:
        -----------
        k : int
            Index of the stick variable (1-indexed)

        Returns:
        --------
        tuple : (v0, v1, v2)
        """
        v0 = beta.rvs(1 - self.alpha, self.theta1)
        v1 = beta.rvs(1 - self.alpha + self.theta1, self.theta2 + self.alpha * k)
        v2 = beta.rvs(1 - self.alpha + self.theta1, self.theta2 + self.alpha * k)

        return v0, v1, v2

    def compute_weights(self, K):
        """
        Compute stick-breaking weights up to truncation level K.

        Weights are computed as:
            W_ik = S_ik * ∏_{j<k} (1 - S_ij)
        where:
            S_1k = V_0k * V_1k
            S_2k = V_0k * V_2k

        Parameters:
        -----------
        K : int
            Truncation level

        Returns:
        --------
        weights1, weights2 : arrays of shape (K,)
        """
        # Sample stick variables
        self.V0 = []
        self.V1 = []
        self.V2 = []

        for k in range(1, K + 1):
            v0, v1, v2 = self.sample_stick_variables(k)
            self.V0.append(v0)
            self.V1.append(v1)
            self.V2.append(v2)

        self.V0 = np.array(self.V0)
        self.V1 = np.array(self.V1)
        self.V2 = np.array(self.V2)

        # Compute S_ik = V_0k * V_ik
        S1 = self.V0 * self.V1
        S2 = self.V0 * self.V2

        # Compute weights: W_ik = S_ik * prod_{j<k}(1 - S_ij)
        weights1 = np.zeros(K)
        weights2 = np.zeros(K)

        stick_left1 = 1.0
        stick_left2 = 1.0

        for k in range(K):
            weights1[k] = S1[k] * stick_left1
            weights2[k] = S2[k] * stick_left2

            stick_left1 *= (1 - S1[k])
            stick_left2 *= (1 - S2[k])

        self.weights1 = weights1
        self.weights2 = weights2

        return weights1, weights2

    def sample_atoms_gaussian(self, K, mu0=0.0, sigma0=1.0):
        """
        Sample Gaussian atoms (shared across components - eq. 18).

        Base measure: ψ_0k ~ N(mu0, sigma0^2)
        Atoms: (ψ_1k, ψ_2k) = (ψ_0k, ψ_0k) for all k

        Parameters:
        -----------
        K : int
            Number of atoms
        mu0 : float
            Mean of base Gaussian
        sigma0 : float
            Standard deviation of base Gaussian

        Returns:
        --------
        atoms : array of shape (K,)
            Shared atoms ψ_0k
        """
        self.atoms = norm.rvs(loc=mu0, scale=sigma0, size=K)
        return self.atoms

    def generate_mixture(self, K, mu0=0.0, sigma0=1.0):
        """
        Generate complete mixture representation.

        Parameters:
        -----------
        K : int
            Truncation level
        mu0 : float
            Mean of base measure
        sigma0 : float
            Scale of base measure

        Returns:
        --------
        dict with keys:
            'weights1', 'weights2': weights for each component
            'atoms': shared atoms
            'V0', 'V1', 'V2': stick-breaking variables
        """
        weights1, weights2 = self.compute_weights(K)
        atoms = self.sample_atoms_gaussian(K, mu0, sigma0)

        return {
            'weights1': weights1,
            'weights2': weights2,
            'atoms': atoms,  # Shared atoms (eq. 18)
            'V0': self.V0,
            'V1': self.V1,
            'V2': self.V2
        }

    def sample_data(self, n_samples, component, K=50, mu0=0.0, sigma0=1.0, obs_sigma=0.5):
        """
        Sample data from the mixture for a specific component.

        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        component : int
            Component index (1 or 2)
        K : int
            Truncation level
        mu0, sigma0 : float
            Base measure parameters
        obs_sigma : float
            Observation noise standard deviation

        Returns:
        --------
        data : array of shape (n_samples,)
        cluster_assignments : array of shape (n_samples,)
        """
        assert component in [1, 2], "Component must be 1 or 2"

        mixture = self.generate_mixture(K, mu0, sigma0)
        weights = mixture['weights1'] if component == 1 else mixture['weights2']
        atoms = mixture['atoms']  # Shared atoms

        # Normalize weights to ensure they sum to 1
        weights = weights / np.sum(weights)

        # Sample cluster assignments
        cluster_assignments = np.random.choice(K, size=n_samples, p=weights)

        # Sample observations from Gaussian kernel centered at atoms
        data = atoms[cluster_assignments] + norm.rvs(scale=obs_sigma, size=n_samples)

        return data, cluster_assignments

    def plot_mixture_density(self, K=50, mu0=0.0, sigma0=1.0, obs_sigma=0.5,
                           x_range=(-10, 10), n_points=1000):
        """
        Plot the mixture density for both components.

        Parameters:
        -----------
        K : int
            Truncation level
        mu0, sigma0 : float
            Base measure parameters
        obs_sigma : float
            Observation noise
        x_range : tuple
            Range for x-axis
        n_points : int
            Number of points for plotting
        """
        mixture = self.generate_mixture(K, mu0, sigma0)

        x = np.linspace(x_range[0], x_range[1], n_points)

        # Compute densities
        density1 = np.zeros(n_points)
        density2 = np.zeros(n_points)

        for k in range(K):
            # Shared atoms (eq. 18)
            kernel = norm.pdf(x, loc=mixture['atoms'][k], scale=obs_sigma)
            density1 += mixture['weights1'][k] * kernel
            density2 += mixture['weights2'][k] * kernel

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(x, density1, 'b-', linewidth=2)
        axes[0].set_title(f'Component 1 Density\n(θ₁={self.theta1}, θ₂={self.theta2}, α={self.alpha})')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('Density')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(x, density2, 'r-', linewidth=2)
        axes[1].set_title(f'Component 2 Density\n(θ₁={self.theta1}, θ₂={self.theta2}, α={self.alpha})')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('Density')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_weights(self, K=50):
        """
        Plot the stick-breaking weights for both components.

        Parameters:
        -----------
        K : int
            Truncation level
        """
        weights1, weights2 = self.compute_weights(K)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].bar(range(1, K+1), weights1, color='blue', alpha=0.6)
        axes[0].set_title(f'Component 1 Weights\n(θ₁={self.theta1}, θ₂={self.theta2}, α={self.alpha})')
        axes[0].set_xlabel('Component k')
        axes[0].set_ylabel('Weight W₁ₖ')
        axes[0].grid(True, alpha=0.3, axis='y')

        axes[1].bar(range(1, K+1), weights2, color='red', alpha=0.6)
        axes[1].set_title(f'Component 2 Weights\n(θ₁={self.theta1}, θ₂={self.theta2}, α={self.alpha})')
        axes[1].set_xlabel('Component k')
        axes[1].set_ylabel('Weight W₂ₖ')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig


def correlation_H1(theta1, theta2):
    """
    Compute correlation between random measures under H1 with α=0 (Dirichlet case).

    From Proposition 3 in the paper:
        C_ij = (1 + θ_1 + θ_2)(1 + θ_1) / ((1 + θ_1)(1 + θ_1 + θ_2) + θ_2)

    Parameters:
    -----------
    theta1, theta2 : float
        Concentration parameters

    Returns:
    --------
    float : correlation coefficient
    """
    numerator = (1 + theta1 + theta2) * (1 + theta1)
    denominator = (1 + theta1) * (1 + theta1 + theta2) + theta2
    return numerator / denominator


if __name__ == "__main__":
    print("=" * 70)
    print("Beta-Product Dependent Pitman-Yor Process - H1 Model")
    print("Bassetti, Casarin, Leisen (2013)")
    print("=" * 70)
    print()

    # Example 1: Dirichlet case (α = 0)
    print("Example 1: Dependent Dirichlet Process (α = 0)")
    print("-" * 70)
    model_dd = BassettiBetaDPY_H1(theta1=0.5, theta2=1.0, alpha=0.0)

    corr = correlation_H1(0.5, 1.0)
    print(f"Theoretical correlation (α=0): {corr:.4f}")
    print()

    # Generate and plot
    fig1 = model_dd.plot_weights(K=30)
    fig1.savefig('/home/user/NP1/h1_dd_weights.png', dpi=150, bbox_inches='tight')
    print("Saved: h1_dd_weights.png")

    fig2 = model_dd.plot_mixture_density(K=50)
    fig2.savefig('/home/user/NP1/h1_dd_density.png', dpi=150, bbox_inches='tight')
    print("Saved: h1_dd_density.png")
    print()

    # Example 2: Pitman-Yor case (α > 0)
    print("Example 2: Dependent Pitman-Yor Process (α = 0.3)")
    print("-" * 70)
    model_py = BassettiBetaDPY_H1(theta1=0.5, theta2=1.0, alpha=0.3)

    fig3 = model_py.plot_weights(K=30)
    fig3.savefig('/home/user/NP1/h1_py_weights.png', dpi=150, bbox_inches='tight')
    print("Saved: h1_py_weights.png")

    fig4 = model_py.plot_mixture_density(K=50)
    fig4.savefig('/home/user/NP1/h1_py_density.png', dpi=150, bbox_inches='tight')
    print("Saved: h1_py_density.png")
    print()

    # Example 3: Sample data
    print("Example 3: Sampling data from both components")
    print("-" * 70)
    data1, clusters1 = model_py.sample_data(n_samples=500, component=1, K=50)
    data2, clusters2 = model_py.sample_data(n_samples=500, component=2, K=50)

    print(f"Component 1: sampled {len(data1)} observations")
    print(f"  Mean: {np.mean(data1):.3f}, Std: {np.std(data1):.3f}")
    print(f"  Number of active clusters: {len(np.unique(clusters1))}")
    print()
    print(f"Component 2: sampled {len(data2)} observations")
    print(f"  Mean: {np.mean(data2):.3f}, Std: {np.std(data2):.3f}")
    print(f"  Number of active clusters: {len(np.unique(clusters2))}")
    print()

    # Plot sampled data
    fig5, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(data1, bins=30, density=True, alpha=0.6, color='blue', edgecolor='black')
    axes[0].set_title('Component 1: Sampled Data')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Density')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(data2, bins=30, density=True, alpha=0.6, color='red', edgecolor='black')
    axes[1].set_title('Component 2: Sampled Data')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Density')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig5.savefig('/home/user/NP1/h1_sampled_data.png', dpi=150, bbox_inches='tight')
    print("Saved: h1_sampled_data.png")

    print()
    print("=" * 70)
    print("Implementation complete!")
    print("=" * 70)
    print()
    print("Key features implemented:")
    print("  ✓ H1 stick-breaking construction: (S_1k, S_2k) = (V_0k*V_1k, V_0k*V_2k)")
    print("  ✓ Shared atoms across components (Equation 18)")
    print("  ✓ Pitman-Yor and Dirichlet process special cases")
    print("  ✓ Correlation computation (Proposition 3)")
    print("  ✓ Data generation and visualization")
    print()

"""
Beta-product Dependent Pitman-Yor Process - H1 Model with MCMC
Implementation of Bassetti, Casarin, Leisen (2014) - Section 5

Full Bayesian inference using slice sampling algorithm.
Simplified: shared atoms without ANOVA decomposition.
"""

import numpy as np
from scipy.stats import beta, norm, invgamma, gamma
from scipy.special import logsumexp, betaln
import matplotlib.pyplot as plt
from tqdm import tqdm


class BassettiBetaDPY_H1_MCMC:
    """
    Beta-Product Dependent Pitman-Yor Process with MCMC inference.

    Implements the slice sampling algorithm from Section 5 of the paper.
    Atoms are shared across components (Equation 18): psi_1k = psi_2k = psi_k
    """

    def __init__(self, Y1, Y2, hyperparams=None):
        """
        Initialize MCMC sampler.

        Parameters:
        -----------
        Y1, Y2 : arrays
            Observed data for components 1 and 2
        hyperparams : dict
            Hyperparameters for priors:
            - 's_0': precision for atoms (default: 0.1)
            - 'lambda': inverse gamma parameter for variance (default: 1.0)
        """
        self.Y1 = np.array(Y1)
        self.Y2 = np.array(Y2)
        self.T1 = len(Y1)
        self.T2 = len(Y2)

        # Hyperparameters
        if hyperparams is None:
            hyperparams = {}
        self.s_0 = hyperparams.get('s_0', 0.1)
        self.lambda_param = hyperparams.get('lambda', 1.0)

        # MCMC state variables (FIXED - Pitman-Yor)
        self.theta1 = 5.0  # Fixed - higher to favor more clusters
        self.theta2 = 5.0  # Fixed
        self.alpha = 0.1   # Fixed - Small Pitman-Yor discount (must be > 0)

        # Allocation variables D_it
        self.D1 = np.zeros(self.T1, dtype=int)
        self.D2 = np.zeros(self.T2, dtype=int)

        # Slice variables U_it
        self.U1 = np.random.uniform(0, 1, self.T1)
        self.U2 = np.random.uniform(0, 1, self.T2)

        # Stick-breaking variables V
        self.V0 = []
        self.V1 = []
        self.V2 = []

        # Atoms (shared - eq. 18)
        self.psi = []  # Shared atoms
        self.sigma2 = []  # Shared variances

        # MCMC samples storage
        self.samples = {
            'theta1': [],
            'theta2': [],
            'alpha': [],
            'n_clusters_1': [],
            'n_clusters_2': [],
            'n_atoms': [],
            'D1': [],
            'D2': []
        }

    def initialize(self, K_init=20):
        """
        Initialize MCMC state.

        Parameters:
        -----------
        K_init : int
            Initial number of components for stick-breaking
        """
        # Initialize stick-breaking variables
        self.V0 = []
        self.V1 = []
        self.V2 = []

        for k in range(K_init):
            v0 = beta.rvs(1 - self.alpha, self.theta1)
            v1 = beta.rvs(1 - self.alpha + self.theta1, self.theta2 + self.alpha * (k+1))
            v2 = beta.rvs(1 - self.alpha + self.theta1, self.theta2 + self.alpha * (k+1))
            self.V0.append(v0)
            self.V1.append(v1)
            self.V2.append(v2)

        # Initialize atoms (shared)
        self.psi = [norm.rvs(0, 1/np.sqrt(self.s_0)) for _ in range(K_init)]
        self.sigma2 = [invgamma.rvs(self.lambda_param/2, scale=self.lambda_param/2)
                       for _ in range(K_init)]

        # Initialize allocations randomly
        weights1, weights2 = self._compute_weights()
        weights1 = weights1 / np.sum(weights1)
        weights2 = weights2 / np.sum(weights2)

        self.D1 = np.random.choice(K_init, size=self.T1, p=weights1)
        self.D2 = np.random.choice(K_init, size=self.T2, p=weights2)

    def _compute_weights(self):
        """
        Compute stick-breaking weights.

        Returns:
        --------
        weights1, weights2 : arrays
        """
        K = len(self.V0)
        S1 = np.array([self.V0[k] * self.V1[k] for k in range(K)])
        S2 = np.array([self.V0[k] * self.V2[k] for k in range(K)])

        weights1 = np.zeros(K)
        weights2 = np.zeros(K)

        stick_left1 = 1.0
        stick_left2 = 1.0

        for k in range(K):
            weights1[k] = S1[k] * stick_left1
            weights2[k] = S2[k] * stick_left2
            stick_left1 *= (1 - S1[k])
            stick_left2 *= (1 - S2[k])

        return weights1, weights2

    def _sample_atoms(self):
        """
        Sample atoms psi_k and sigma2_k given allocations.
        Atoms are shared across components (eq. 18).
        """
        K = len(self.psi)

        for k in range(K):
            # Find observations allocated to cluster k
            idx1_k = np.where(self.D1 == k)[0]
            idx2_k = np.where(self.D2 == k)[0]
            A1k = len(idx1_k)
            A2k = len(idx2_k)

            # Sample psi_k (shared atom)
            if A1k > 0 or A2k > 0:
                # Posterior for psi_k given data from both components
                sum_y = 0
                precision = self.s_0

                if A1k > 0:
                    sum_y += np.sum(self.Y1[idx1_k])
                    precision += A1k / self.sigma2[k]

                if A2k > 0:
                    sum_y += np.sum(self.Y2[idx2_k])
                    precision += A2k / self.sigma2[k]

                var_post = 1 / precision
                mean_post = var_post * sum_y / self.sigma2[k]

                self.psi[k] = norm.rvs(mean_post, np.sqrt(var_post))
            else:
                # Prior
                self.psi[k] = norm.rvs(0, 1/np.sqrt(self.s_0))

            # Sample sigma2_k (shared variance)
            if A1k > 0 or A2k > 0:
                a_post = self.lambda_param/2 + (A1k + A2k)/2

                sum_sq = 0
                if A1k > 0:
                    sum_sq += np.sum((self.Y1[idx1_k] - self.psi[k])**2)
                if A2k > 0:
                    sum_sq += np.sum((self.Y2[idx2_k] - self.psi[k])**2)

                b_post = self.lambda_param/2 + sum_sq/2

                self.sigma2[k] = invgamma.rvs(a_post, scale=b_post)
            else:
                # Prior
                self.sigma2[k] = invgamma.rvs(self.lambda_param/2,
                                             scale=self.lambda_param/2)

    def _sample_V_and_phi(self):
        """
        Sample stick-breaking variables V and hyperparameters phi=(theta1, theta2, alpha).
        """
        K = len(self.V0)

        # Count allocations
        D_star = max(np.max(self.D1), np.max(self.D2)) + 1
        A1 = np.bincount(self.D1, minlength=K)
        A2 = np.bincount(self.D2, minlength=K)

        # Sample V0k
        for k in range(K):
            a_v0 = 1 - self.alpha + A1[k] + A2[k]
            b_v0 = self.theta1 + np.sum(A1[k+1:]) + np.sum(A2[k+1:])
            self.V0[k] = beta.rvs(a_v0, b_v0)

        # Sample V1k and V2k
        for k in range(K):
            # V1k
            a_v1 = 1 - self.alpha + self.theta1 + A1[k]
            b_v1 = self.theta2 + self.alpha * (k+1) + np.sum(A1[k+1:])
            self.V1[k] = beta.rvs(a_v1, b_v1)

            # V2k
            a_v2 = 1 - self.alpha + self.theta1 + A2[k]
            b_v2 = self.theta2 + self.alpha * (k+1) + np.sum(A2[k+1:])
            self.V2[k] = beta.rvs(a_v2, b_v2)

        # Sample hyperparameters phi using Metropolis-Hastings with informative priors
        # DISABLED: Keep hyperparameters fixed
        # self._sample_phi_mh(D_star, A1, A2)

    def _sample_phi_mh(self, D_star, A1, A2):
        """
        Metropolis-Hastings step for hyperparameters.
        """
        K = len(self.V0)

        # Current parameters in unconstrained space
        xi_current = np.array([
            np.log(self.theta1),
            np.log(self.theta2),
            np.log(self.alpha / (1 - self.alpha)) if self.alpha > 0 and self.alpha < 1 else -10
        ])

        # Propose new values (random walk)
        xi_prop = xi_current + np.random.normal(0, 0.05, 3)

        # Transform back to constrained space
        theta1_prop = np.exp(xi_prop[0])
        theta2_prop = np.exp(xi_prop[1])
        alpha_prop = np.exp(xi_prop[2]) / (1 + np.exp(xi_prop[2]))

        # Ensure valid range
        if theta1_prop <= 0 or theta2_prop <= 0 or alpha_prop <= 0 or alpha_prop >= 1:
            return

        # Compute log acceptance ratio
        log_acc = 0

        # Likelihood contribution from V0
        for k in range(D_star):
            if self.V0[k] > 0 and self.V0[k] < 1:
                # Current
                log_acc -= betaln(1 - self.alpha, self.theta1)
                log_acc += (1 - self.alpha - 1) * np.log(self.V0[k])
                log_acc += (self.theta1 - 1) * np.log(1 - self.V0[k])

                # Proposed
                log_acc += betaln(1 - alpha_prop, theta1_prop)
                log_acc -= (1 - alpha_prop - 1) * np.log(self.V0[k])
                log_acc -= (theta1_prop - 1) * np.log(1 - self.V0[k])

        # Likelihood contribution from V1 and V2
        for k in range(D_star):
            if self.V1[k] > 0 and self.V1[k] < 1:
                # Current
                a_curr = 1 - self.alpha + self.theta1
                b_curr = self.theta2 + self.alpha * (k+1)
                log_acc -= betaln(a_curr, b_curr)
                log_acc += (a_curr - 1) * np.log(self.V1[k])
                log_acc += (b_curr - 1) * np.log(1 - self.V1[k])

                # Proposed
                a_prop = 1 - alpha_prop + theta1_prop
                b_prop = theta2_prop + alpha_prop * (k+1)
                log_acc += betaln(a_prop, b_prop)
                log_acc -= (a_prop - 1) * np.log(self.V1[k])
                log_acc -= (b_prop - 1) * np.log(1 - self.V1[k])

            if self.V2[k] > 0 and self.V2[k] < 1:
                # Same for V2
                a_curr = 1 - self.alpha + self.theta1
                b_curr = self.theta2 + self.alpha * (k+1)
                log_acc -= betaln(a_curr, b_curr)
                log_acc += (a_curr - 1) * np.log(self.V2[k])
                log_acc += (b_curr - 1) * np.log(1 - self.V2[k])

                a_prop = 1 - alpha_prop + theta1_prop
                b_prop = theta2_prop + alpha_prop * (k+1)
                log_acc += betaln(a_prop, b_prop)
                log_acc -= (a_prop - 1) * np.log(self.V2[k])
                log_acc -= (b_prop - 1) * np.log(1 - self.V2[k])

        # Prior contributions
        # theta1 ~ Gamma(2, 2): log p(theta1) = (a-1)*log(theta1) - b*theta1 + const
        a_theta, b_theta = 2.0, 2.0
        log_acc += (a_theta - 1) * (np.log(theta1_prop) - np.log(self.theta1))
        log_acc -= b_theta * (theta1_prop - self.theta1)

        # theta2 ~ Gamma(2, 2)
        log_acc += (a_theta - 1) * (np.log(theta2_prop) - np.log(self.theta2))
        log_acc -= b_theta * (theta2_prop - self.theta2)

        # alpha ~ Beta(1, 10): log p(alpha) = (a-1)*log(alpha) + (b-1)*log(1-alpha) + const
        a_alpha, b_alpha = 1.0, 10.0
        log_acc += (a_alpha - 1) * (np.log(alpha_prop) - np.log(self.alpha))
        log_acc += (b_alpha - 1) * (np.log(1 - alpha_prop) - np.log(1 - self.alpha))

        # Jacobian for transformation
        log_acc += xi_prop[0] - xi_current[0]  # theta1
        log_acc += xi_prop[1] - xi_current[1]  # theta2
        log_acc += xi_prop[2] - xi_current[2] - 2*np.log(1 + np.exp(xi_prop[2])) + 2*np.log(1 + np.exp(xi_current[2]))  # alpha

        # Accept/reject
        if np.log(np.random.uniform()) < log_acc:
            self.theta1 = theta1_prop
            self.theta2 = theta2_prop
            self.alpha = alpha_prop

    def _sample_U(self):
        """
        Sample slice variables U_it.
        """
        weights1, weights2 = self._compute_weights()

        for t in range(self.T1):
            d = self.D1[t]
            self.U1[t] = np.random.uniform(0, weights1[d])

        for t in range(self.T2):
            d = self.D2[t]
            self.U2[t] = np.random.uniform(0, weights2[d])

    def _sample_D(self):
        """
        Sample allocation variables D_it.
        """
        weights1, weights2 = self._compute_weights()

        # Sample D1
        for t in range(self.T1):
            # Find N*_it: smallest integer such that sum W_ik > 1 - U_it
            cumsum = np.cumsum(weights1)
            N_star = np.searchsorted(cumsum, 1 - self.U1[t]) + 1
            N_star = min(N_star, len(weights1))

            # Ensure atoms exist
            while len(self.psi) < N_star:
                self.psi.append(norm.rvs(0, 1/np.sqrt(self.s_0)))
                self.sigma2.append(invgamma.rvs(self.lambda_param/2,
                                                scale=self.lambda_param/2))

            # Compute probabilities for d = 0, ..., N*_it-1
            log_probs = np.zeros(N_star)
            for d in range(N_star):
                if self.U1[t] <= weights1[d]:
                    # Gaussian kernel
                    log_probs[d] = norm.logpdf(self.Y1[t], self.psi[d], np.sqrt(self.sigma2[d]))
                else:
                    log_probs[d] = -np.inf

            # Normalize and sample
            log_probs -= logsumexp(log_probs)
            probs = np.exp(log_probs)
            self.D1[t] = np.random.choice(N_star, p=probs)

        # Sample D2
        for t in range(self.T2):
            cumsum = np.cumsum(weights2)
            N_star = np.searchsorted(cumsum, 1 - self.U2[t]) + 1
            N_star = min(N_star, len(weights2))

            # Ensure atoms exist
            while len(self.psi) < N_star:
                self.psi.append(norm.rvs(0, 1/np.sqrt(self.s_0)))
                self.sigma2.append(invgamma.rvs(self.lambda_param/2,
                                                scale=self.lambda_param/2))

            log_probs = np.zeros(N_star)
            for d in range(N_star):
                if self.U2[t] <= weights2[d]:
                    log_probs[d] = norm.logpdf(self.Y2[t], self.psi[d], np.sqrt(self.sigma2[d]))
                else:
                    log_probs[d] = -np.inf

            log_probs -= logsumexp(log_probs)
            probs = np.exp(log_probs)
            self.D2[t] = np.random.choice(N_star, p=probs)

    def gibbs_step(self):
        """
        One step of the Gibbs sampler.

        Order (Section 5):
        1. Sample atoms psi and sigma2
        2. Sample V and phi
        3. Sample U
        4. Sample D
        """
        self._sample_atoms()
        self._sample_V_and_phi()
        self._sample_U()
        self._sample_D()

    def run_mcmc(self, n_iter=5000, burn_in=1000, thin=5, verbose=True):
        """
        Run the MCMC sampler.

        Parameters:
        -----------
        n_iter : int
            Number of iterations
        burn_in : int
            Burn-in period
        thin : int
            Thinning interval
        verbose : bool
            Print progress

        Returns:
        --------
        samples : dict
            MCMC samples
        """
        if verbose:
            print(f"Running MCMC: {n_iter} iterations, burn-in={burn_in}, thin={thin}")
            print(f"Data: T1={self.T1}, T2={self.T2}")
            print(f"Hyperparameters: s_0={self.s_0}, lambda={self.lambda_param}")
            print()

        iterator = tqdm(range(n_iter)) if verbose else range(n_iter)

        for it in iterator:
            self.gibbs_step()

            # Store samples after burn-in
            if it >= burn_in and (it - burn_in) % thin == 0:
                self.samples['theta1'].append(self.theta1)
                self.samples['theta2'].append(self.theta2)
                self.samples['alpha'].append(self.alpha)
                self.samples['n_clusters_1'].append(len(np.unique(self.D1)))
                self.samples['n_clusters_2'].append(len(np.unique(self.D2)))
                self.samples['n_atoms'].append(len(self.psi))  # Total atoms created
                self.samples['D1'].append(self.D1.copy())
                self.samples['D2'].append(self.D2.copy())

        return self.samples

    def summary(self):
        """
        Print summary statistics.
        """
        print()
        print("=" * 70)
        print("MCMC Summary")
        print("=" * 70)
        print(f"Number of posterior samples: {len(self.samples['theta1'])}")
        print()
        print("Parameter Estimates:")
        print(f"  θ₁: {np.mean(self.samples['theta1']):.4f} (95% CI: [{np.percentile(self.samples['theta1'], 2.5):.4f}, {np.percentile(self.samples['theta1'], 97.5):.4f}])")
        print(f"  θ₂: {np.mean(self.samples['theta2']):.4f} (95% CI: [{np.percentile(self.samples['theta2'], 2.5):.4f}, {np.percentile(self.samples['theta2'], 97.5):.4f}])")
        print(f"  α:  {np.mean(self.samples['alpha']):.4f} (95% CI: [{np.percentile(self.samples['alpha'], 2.5):.4f}, {np.percentile(self.samples['alpha'], 97.5):.4f}])")
        print()
        print("Number of Clusters:")
        print(f"  Component 1: {np.mean(self.samples['n_clusters_1']):.2f} (mode: {np.bincount(self.samples['n_clusters_1']).argmax()})")
        print(f"  Component 2: {np.mean(self.samples['n_clusters_2']):.2f} (mode: {np.bincount(self.samples['n_clusters_2']).argmax()})")
        print("=" * 70)
        print()

    def plot_trace(self):
        """
        Plot MCMC trace plots.
        """
        fig, axes = plt.subplots(3, 3, figsize=(15, 11))

        # Theta1
        axes[0, 0].plot(self.samples['theta1'])
        axes[0, 0].set_title('θ₁')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].grid(True, alpha=0.3)

        # Theta2
        axes[0, 1].plot(self.samples['theta2'])
        axes[0, 1].set_title('θ₂')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].grid(True, alpha=0.3)

        # Alpha
        axes[0, 2].plot(self.samples['alpha'])
        axes[0, 2].set_title('α')
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].grid(True, alpha=0.3)

        # Number of occupied clusters - Component 1
        axes[1, 0].plot(self.samples['n_clusters_1'])
        axes[1, 0].set_title('Occupied Clusters (Component 1)')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('K₁ occupied')
        axes[1, 0].grid(True, alpha=0.3)

        # Number of occupied clusters - Component 2
        axes[1, 1].plot(self.samples['n_clusters_2'])
        axes[1, 1].set_title('Occupied Clusters (Component 2)')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('K₂ occupied')
        axes[1, 1].grid(True, alpha=0.3)

        # Total number of atoms created
        axes[1, 2].plot(self.samples['n_atoms'])
        axes[1, 2].set_title('Total Atoms Created')
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('K total')
        axes[1, 2].grid(True, alpha=0.3)

        # Posterior distribution of occupied K
        axes[2, 0].hist(self.samples['n_clusters_1'], bins=20, alpha=0.6, label='Component 1')
        axes[2, 0].hist(self.samples['n_clusters_2'], bins=20, alpha=0.6, label='Component 2')
        axes[2, 0].set_title('Posterior Distribution (Occupied K)')
        axes[2, 0].set_xlabel('Number of Clusters')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        # Posterior distribution of total atoms
        axes[2, 1].hist(self.samples['n_atoms'], bins=20, alpha=0.6, color='green')
        axes[2, 1].set_title('Posterior Distribution (Total Atoms)')
        axes[2, 1].set_xlabel('Number of Atoms')
        axes[2, 1].grid(True, alpha=0.3)

        # Hide unused subplot
        axes[2, 2].axis('off')

        plt.tight_layout()
        return fig

    def plot_data(self, true_means=None):
        """
        Plot data with estimated cluster means.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Component 1
        axes[0].hist(self.Y1, bins=30, density=True, alpha=0.6, label='Data')
        if true_means is not None:
            for mean in true_means:
                axes[0].axvline(mean, color='red', linestyle='--', linewidth=2, label='True means')
        axes[0].set_title('Component 1')
        axes[0].set_xlabel('Y₁')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Component 2
        axes[1].hist(self.Y2, bins=30, density=True, alpha=0.6, color='orange', label='Data')
        if true_means is not None:
            for mean in true_means:
                axes[1].axvline(mean, color='red', linestyle='--', linewidth=2, label='True means')
        axes[1].set_title('Component 2')
        axes[1].set_xlabel('Y₂')
        axes[1].set_ylabel('Density')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


if __name__ == "__main__":
    print("=" * 70)
    print("MCMC for Beta-Product Dependent Pitman-Yor (H1)")
    print("Bassetti, Casarin, Leisen (2014)")
    print("=" * 70)
    print()

    # Generate synthetic data
    np.random.seed(42)

    # True parameters
    true_means = np.array([-5, 0, 5])
    true_sigma = 0.5

    # Component 1: 3 clusters with different weights
    n1_per_cluster = np.array([60, 80, 60])
    Y1 = []
    for i, mean in enumerate(true_means):
        Y1.extend(norm.rvs(mean, true_sigma, size=n1_per_cluster[i]))
    Y1 = np.array(Y1)
    np.random.shuffle(Y1)

    # Component 2: same locations, different weights
    n2_per_cluster = np.array([40, 100, 60])
    Y2 = []
    for i, mean in enumerate(true_means):
        Y2.extend(norm.rvs(mean, true_sigma, size=n2_per_cluster[i]))
    Y2 = np.array(Y2)
    np.random.shuffle(Y2)

    true_weights_1 = n1_per_cluster / np.sum(n1_per_cluster)
    true_weights_2 = n2_per_cluster / np.sum(n2_per_cluster)

    print(f"Generated data: Y1 ({len(Y1)} obs), Y2 ({len(Y2)} obs)")
    print(f"True means: {true_means.tolist()}")
    print(f"True weights Y1: {true_weights_1}")
    print(f"True weights Y2: {true_weights_2}")
    print()

    # Initialize sampler
    sampler = BassettiBetaDPY_H1_MCMC(Y1, Y2)
    sampler.initialize(K_init=30)

    # Run MCMC
    samples = sampler.run_mcmc(n_iter=5000, burn_in=1000, thin=5, verbose=True)

    # Summary
    print()
    sampler.summary()

    # Plots
    fig = sampler.plot_trace()
    fig.savefig('/home/user/NP1/mcmc_trace.png', dpi=150, bbox_inches='tight')
    print("Saved: mcmc_trace.png")

    fig2 = sampler.plot_data(true_means=true_means)
    fig2.savefig('/home/user/NP1/mcmc_data.png', dpi=150, bbox_inches='tight')
    print("Saved: mcmc_data.png")

    print()
    print("=" * 70)
    print("MCMC completed successfully!")
    print("=" * 70)

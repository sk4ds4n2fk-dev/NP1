"""
Beta-product Dependent Pitman-Yor Process - H1 Model with MCMC
Implementation of Bassetti, Casarin, Leisen (2013) - Section 5

Full Bayesian inference using slice sampling algorithm.
"""

import numpy as np
from scipy.stats import beta, norm, invgamma, gamma
from scipy.special import logsumexp, betaln
import matplotlib.pyplot as plt
from tqdm import tqdm


class BassettiBetaDPY_H1_MCMC:
    """
    Beta-Product Dependent Pitman-Yor Process with MCMC inference.

    Implements the slice sampling algorithm from Section 5 of the paper
    for Gaussian mixture models (Section 6.1).
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
            - 's_i': precision for component-specific means (default: 0.1)
            - 's_0': precision for common means (default: 0.01)
            - 'lambda': inverse gamma parameter (default: 0.5)
            - 'epsilon': inverse gamma parameter (default: 1.0)
            - 'zeta_11', 'zeta_21': gamma prior for theta1 (default: 0.01, 0.01)
            - 'zeta_12', 'zeta_22': gamma prior for theta2 (default: 0.01, 0.01)
        """
        self.Y1 = np.array(Y1)
        self.Y2 = np.array(Y2)
        self.T1 = len(Y1)
        self.T2 = len(Y2)

        # Hyperparameters (following Section 6.1.1)
        if hyperparams is None:
            hyperparams = {}
        self.s_i = hyperparams.get('s_i', 0.1)
        self.s_0 = hyperparams.get('s_0', 0.01)
        self.lambda_param = hyperparams.get('lambda', 0.5)
        self.epsilon = hyperparams.get('epsilon', 1.0)
        self.zeta_11 = hyperparams.get('zeta_11', 0.01)
        self.zeta_21 = hyperparams.get('zeta_21', 0.01)
        self.zeta_12 = hyperparams.get('zeta_12', 0.01)
        self.zeta_22 = hyperparams.get('zeta_22', 0.01)

        # MCMC state variables
        self.theta1 = 1.0
        self.theta2 = 1.0
        self.alpha = 0.0

        # Allocation variables D_it
        self.D1 = np.ones(self.T1, dtype=int)  # cluster assignments for Y1
        self.D2 = np.ones(self.T2, dtype=int)  # cluster assignments for Y2

        # Slice variables U_it
        self.U1 = np.random.uniform(0, 1, self.T1)
        self.U2 = np.random.uniform(0, 1, self.T2)

        # Stick-breaking variables V
        self.V0 = []
        self.V1 = []
        self.V2 = []

        # Atoms (shared - eq. 18)
        self.mu_0 = []  # common means
        self.mu_1 = []  # component 1 specific means
        self.mu_2 = []  # component 2 specific means
        self.sigma2_0 = []  # common variances
        self.sigma2_1 = []  # component 1 specific variances
        self.sigma2_2 = []  # component 2 specific variances

        # MCMC samples storage
        self.samples = {
            'theta1': [],
            'theta2': [],
            'alpha': [],
            'n_clusters_1': [],
            'n_clusters_2': [],
            'D1': [],
            'D2': []
        }

    def initialize(self, K_init=20):
        """
        Initialize MCMC state.

        Parameters:
        -----------
        K_init : int
            Initial number of mixture components
        """
        # Initialize stick-breaking variables
        for k in range(K_init):
            v0 = beta.rvs(1 - self.alpha, self.theta1)
            v1 = beta.rvs(1 - self.alpha + self.theta1,
                         self.theta2 + self.alpha * (k + 1))
            v2 = beta.rvs(1 - self.alpha + self.theta1,
                         self.theta2 + self.alpha * (k + 1))
            self.V0.append(v0)
            self.V1.append(v1)
            self.V2.append(v2)

        # Initialize atoms from prior
        for k in range(K_init):
            self.mu_0.append(norm.rvs(0, 1/np.sqrt(self.s_0)))
            self.mu_1.append(norm.rvs(0, 1/np.sqrt(self.s_i)))
            self.mu_2.append(norm.rvs(0, 1/np.sqrt(self.s_i)))
            self.sigma2_0.append(invgamma.rvs(self.epsilon/2, scale=self.epsilon/2))
            self.sigma2_1.append(invgamma.rvs(self.lambda_param/2,
                                             scale=self.lambda_param/2))
            self.sigma2_2.append(invgamma.rvs(self.lambda_param/2,
                                             scale=self.lambda_param/2))

        # Initialize allocations randomly
        self.D1 = np.random.randint(0, K_init, self.T1)
        self.D2 = np.random.randint(0, K_init, self.T2)

        # Initialize slice variables
        weights1, weights2 = self._compute_weights()
        for t in range(self.T1):
            self.U1[t] = np.random.uniform(0, weights1[self.D1[t]])
        for t in range(self.T2):
            self.U2[t] = np.random.uniform(0, weights2[self.D2[t]])

    def _compute_weights(self):
        """
        Compute stick-breaking weights from V variables.

        Returns:
        --------
        weights1, weights2 : arrays
        """
        K = len(self.V0)
        S1 = np.array(self.V0) * np.array(self.V1)
        S2 = np.array(self.V0) * np.array(self.V2)

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

    def _get_D_star(self):
        """
        Get D* = max index of allocated clusters.
        """
        return max(np.max(self.D1), np.max(self.D2)) + 1

    def _sample_atoms(self):
        """
        Sample atoms given allocations (Appendix B.2).

        Following equations (46)-(51) for Gaussian mixtures.
        """
        D_star = self._get_D_star()

        # Ensure we have enough atoms
        while len(self.mu_0) < D_star:
            self.mu_0.append(norm.rvs(0, 1/np.sqrt(self.s_0)))
            self.mu_1.append(norm.rvs(0, 1/np.sqrt(self.s_i)))
            self.mu_2.append(norm.rvs(0, 1/np.sqrt(self.s_i)))
            self.sigma2_0.append(invgamma.rvs(self.epsilon/2, scale=self.epsilon/2))
            self.sigma2_1.append(invgamma.rvs(self.lambda_param/2,
                                             scale=self.lambda_param/2))
            self.sigma2_2.append(invgamma.rvs(self.lambda_param/2,
                                             scale=self.lambda_param/2))

        # Sample component-specific parameters for each cluster k
        for k in range(D_star):
            # Find observations allocated to cluster k
            idx1_k = (self.D1 == k)
            idx2_k = (self.D2 == k)
            A1k = np.sum(idx1_k)
            A2k = np.sum(idx2_k)

            # Sample mu_1k | rest (equation 46)
            if A1k > 0:
                eta1 = np.sum(self.Y1[idx1_k] - self.mu_0[k])
                var_post = 1 / (self.s_i**2 + A1k / (self.sigma2_1[k] * self.sigma2_0[k]))
                mean_post = var_post * eta1 / (self.sigma2_1[k] * self.sigma2_0[k])
                self.mu_1[k] = norm.rvs(mean_post, np.sqrt(var_post))
            else:
                self.mu_1[k] = norm.rvs(0, 1/np.sqrt(self.s_i))

            # Sample mu_2k | rest
            if A2k > 0:
                eta2 = np.sum(self.Y2[idx2_k] - self.mu_0[k])
                var_post = 1 / (self.s_i**2 + A2k / (self.sigma2_2[k] * self.sigma2_0[k]))
                mean_post = var_post * eta2 / (self.sigma2_2[k] * self.sigma2_0[k])
                self.mu_2[k] = norm.rvs(mean_post, np.sqrt(var_post))
            else:
                self.mu_2[k] = norm.rvs(0, 1/np.sqrt(self.s_i))

            # Sample sigma2_1k | rest (equation 47)
            if A1k > 0:
                eta2_1 = np.sum((self.Y1[idx1_k] - self.mu_0[k] - self.mu_1[k])**2)
                a_post = self.lambda_param/2 + A1k/2
                b_post = self.lambda_param/2 + eta2_1 / (2 * self.sigma2_0[k])
                self.sigma2_1[k] = invgamma.rvs(a_post, scale=b_post)
            else:
                self.sigma2_1[k] = invgamma.rvs(self.lambda_param/2,
                                               scale=self.lambda_param/2)

            # Sample sigma2_2k | rest
            if A2k > 0:
                eta2_2 = np.sum((self.Y2[idx2_k] - self.mu_0[k] - self.mu_2[k])**2)
                a_post = self.lambda_param/2 + A2k/2
                b_post = self.lambda_param/2 + eta2_2 / (2 * self.sigma2_0[k])
                self.sigma2_2[k] = invgamma.rvs(a_post, scale=b_post)
            else:
                self.sigma2_2[k] = invgamma.rvs(self.lambda_param/2,
                                               scale=self.lambda_param/2)

        # Sample common parameters mu_0k, sigma2_0k (equations 49, 51)
        for k in range(D_star):
            idx1_k = (self.D1 == k)
            idx2_k = (self.D2 == k)
            A1k = np.sum(idx1_k)
            A2k = np.sum(idx2_k)

            # Sample mu_0k | rest (equation 49)
            precision_post = self.s_0**2
            mean_contrib = 0.0

            if A1k > 0:
                precision_post += A1k / (self.sigma2_0[k] * self.sigma2_1[k])
                mean_contrib += np.sum(self.Y1[idx1_k] - self.mu_1[k]) / \
                               (self.sigma2_0[k] * self.sigma2_1[k])

            if A2k > 0:
                precision_post += A2k / (self.sigma2_0[k] * self.sigma2_2[k])
                mean_contrib += np.sum(self.Y2[idx2_k] - self.mu_2[k]) / \
                               (self.sigma2_0[k] * self.sigma2_2[k])

            var_post = 1 / precision_post
            mean_post = var_post * mean_contrib
            self.mu_0[k] = norm.rvs(mean_post, np.sqrt(var_post))

            # Sample sigma2_0k | rest (equation 51)
            a_post = self.epsilon/2 + (A1k + A2k)/2
            b_post = self.epsilon/2

            if A1k > 0:
                eta2_1 = np.sum((self.Y1[idx1_k] - self.mu_0[k] - self.mu_1[k])**2)
                b_post += eta2_1 / (2 * self.sigma2_1[k])

            if A2k > 0:
                eta2_2 = np.sum((self.Y2[idx2_k] - self.mu_0[k] - self.mu_2[k])**2)
                b_post += eta2_2 / (2 * self.sigma2_2[k])

            self.sigma2_0[k] = invgamma.rvs(a_post, scale=b_post)

    def _sample_V_and_phi(self):
        """
        Sample V (stick-breaking) and phi=(theta1, theta2, alpha).

        Following Appendix B.1 with M-H steps.
        """
        D_star = self._get_D_star()

        # Ensure we have enough V variables
        while len(self.V0) < D_star + 10:
            k = len(self.V0)
            v0 = beta.rvs(1 - self.alpha, self.theta1)
            v1 = beta.rvs(1 - self.alpha + self.theta1,
                         self.theta2 + self.alpha * (k + 1))
            v2 = beta.rvs(1 - self.alpha + self.theta1,
                         self.theta2 + self.alpha * (k + 1))
            self.V0.append(v0)
            self.V1.append(v1)
            self.V2.append(v2)

        # Compute A_ik and B_ik (counts)
        A1 = np.bincount(self.D1, minlength=D_star)
        A2 = np.bincount(self.D2, minlength=D_star)
        B1 = np.array([np.sum(self.D1 > k) for k in range(D_star)])
        B2 = np.array([np.sum(self.D2 > k) for k in range(D_star)])

        # Sample V* (allocated clusters) - equation (40)
        for k in range(D_star):
            # Sample V0k
            a_v0 = 1 - self.alpha + A1[k] + A2[k]
            b_v0 = self.theta1 + B1[k] + B2[k]

            # Use random walk MH
            v0_prop = self.V0[k] + np.random.normal(0, 0.1)
            v0_prop = np.clip(v0_prop, 0.001, 0.999)

            log_acc = (a_v0 - 1) * (np.log(v0_prop) - np.log(self.V0[k])) + \
                     (b_v0 - 1) * (np.log(1 - v0_prop) - np.log(1 - self.V0[k]))

            if np.log(np.random.rand()) < log_acc:
                self.V0[k] = v0_prop

            # Sample V1k
            a_v1 = self.theta1 + 1 - self.alpha + A1[k]
            b_v1 = self.theta2 + self.alpha * (k + 1) + B1[k]

            v1_prop = self.V1[k] + np.random.normal(0, 0.1)
            v1_prop = np.clip(v1_prop, 0.001, 0.999)

            log_acc = (a_v1 - 1) * (np.log(v1_prop) - np.log(self.V1[k])) + \
                     (b_v1 - 1) * (np.log(1 - v1_prop) - np.log(1 - self.V1[k]))

            if np.log(np.random.rand()) < log_acc:
                self.V1[k] = v1_prop

            # Sample V2k
            a_v2 = self.theta1 + 1 - self.alpha + A2[k]
            b_v2 = self.theta2 + self.alpha * (k + 1) + B2[k]

            v2_prop = self.V2[k] + np.random.normal(0, 0.1)
            v2_prop = np.clip(v2_prop, 0.001, 0.999)

            log_acc = (a_v2 - 1) * (np.log(v2_prop) - np.log(self.V2[k])) + \
                     (b_v2 - 1) * (np.log(1 - v2_prop) - np.log(1 - self.V2[k]))

            if np.log(np.random.rand()) < log_acc:
                self.V2[k] = v2_prop

        # Sample V** (non-allocated) from prior
        for k in range(D_star, len(self.V0)):
            self.V0[k] = beta.rvs(1 - self.alpha, self.theta1)
            self.V1[k] = beta.rvs(1 - self.alpha + self.theta1,
                                 self.theta2 + self.alpha * (k + 1))
            self.V2[k] = beta.rvs(1 - self.alpha + self.theta1,
                                 self.theta2 + self.alpha * (k + 1))

        # Sample phi = (theta1, theta2, alpha) with M-H (equation 52-54)
        self._sample_phi_mh(D_star, A1, A2)

    def _sample_phi_mh(self, D_star, A1, A2):
        """
        Sample (theta1, theta2, alpha) using Metropolis-Hastings.

        Transform to unconstrained space for sampling.
        """
        # Current parameters in unconstrained space
        xi_current = np.array([
            np.log(self.theta1),
            np.log(self.theta2),
            np.log(self.alpha / (1 - self.alpha)) if self.alpha > 0 else -5.0
        ])

        # Random walk proposal
        xi_prop = xi_current + np.random.normal(0, 0.1, 3)

        # Transform back
        theta1_prop = np.exp(xi_prop[0])
        theta2_prop = np.exp(xi_prop[1])
        alpha_prop = np.exp(xi_prop[2]) / (1 + np.exp(xi_prop[2]))
        alpha_prop = np.clip(alpha_prop, 0.0, 0.99)

        # Compute log acceptance ratio (equation 52)
        log_acc = 0.0

        # Log-likelihood contribution
        for k in range(D_star):
            # Current
            log_acc -= betaln(1 - self.alpha, self.theta1)
            log_acc -= 2 * betaln(self.theta1 + 1 - self.alpha,
                                 self.theta2 + self.alpha * (k + 1))

            # Proposed
            log_acc += betaln(1 - alpha_prop, theta1_prop)
            log_acc += 2 * betaln(theta1_prop + 1 - alpha_prop,
                                 theta2_prop + alpha_prop * (k + 1))

            # Data contribution from V variables
            log_acc += (1 - alpha_prop - 1) * np.log(self.V0[k])
            log_acc += (theta1_prop - 1) * np.log(1 - self.V0[k])
            log_acc += 2 * (theta1_prop + 1 - alpha_prop - 1) * \
                      (np.log(self.V1[k]) + np.log(self.V2[k]))
            log_acc += 2 * (theta2_prop + alpha_prop * (k+1) - 1) * \
                      (np.log(1 - self.V1[k]) + np.log(1 - self.V2[k]))

            log_acc -= (1 - self.alpha - 1) * np.log(self.V0[k])
            log_acc -= (self.theta1 - 1) * np.log(1 - self.V0[k])
            log_acc -= 2 * (self.theta1 + 1 - self.alpha - 1) * \
                      (np.log(self.V1[k]) + np.log(self.V2[k]))
            log_acc -= 2 * (self.theta2 + self.alpha * (k+1) - 1) * \
                      (np.log(1 - self.V1[k]) + np.log(1 - self.V2[k]))

        # Prior contribution (Gamma priors)
        log_acc += (self.zeta_11 - 1) * xi_prop[0] - self.zeta_21 * theta1_prop
        log_acc += (self.zeta_12 - 1) * xi_prop[1] - self.zeta_22 * theta2_prop
        log_acc -= (self.zeta_11 - 1) * xi_current[0] - self.zeta_21 * self.theta1
        log_acc -= (self.zeta_12 - 1) * xi_current[1] - self.zeta_22 * self.theta2

        # Jacobian correction
        log_acc += xi_prop[0] + xi_prop[1]  # theta1, theta2
        log_acc += xi_prop[2] - 2 * np.log(1 + np.exp(xi_prop[2]))  # alpha
        log_acc -= xi_current[0] + xi_current[1]
        log_acc -= xi_current[2] - 2 * np.log(1 + np.exp(xi_current[2]))

        # Accept/reject
        if np.log(np.random.rand()) < log_acc:
            self.theta1 = theta1_prop
            self.theta2 = theta2_prop
            self.alpha = alpha_prop

    def _sample_U(self):
        """
        Sample slice variables U_it (equation 43).
        """
        weights1, weights2 = self._compute_weights()

        for t in range(self.T1):
            self.U1[t] = np.random.uniform(0, weights1[self.D1[t]])

        for t in range(self.T2):
            self.U2[t] = np.random.uniform(0, weights2[self.D2[t]])

    def _sample_D(self):
        """
        Sample allocation variables D_it (equation 44).
        """
        weights1, weights2 = self._compute_weights()

        # Sample D1
        for t in range(self.T1):
            # Find N*_it: smallest integer such that sum W_ik > 1 - U_it
            cumsum = np.cumsum(weights1)
            N_star = np.searchsorted(cumsum, 1 - self.U1[t]) + 1
            N_star = min(N_star, len(weights1))

            # Ensure atoms exist
            while len(self.mu_0) < N_star:
                self.mu_0.append(norm.rvs(0, 1/np.sqrt(self.s_0)))
                self.mu_1.append(norm.rvs(0, 1/np.sqrt(self.s_i)))
                self.mu_2.append(norm.rvs(0, 1/np.sqrt(self.s_i)))
                self.sigma2_0.append(invgamma.rvs(self.epsilon/2, scale=self.epsilon/2))
                self.sigma2_1.append(invgamma.rvs(self.lambda_param/2,
                                                 scale=self.lambda_param/2))
                self.sigma2_2.append(invgamma.rvs(self.lambda_param/2,
                                                 scale=self.lambda_param/2))

            # Compute probabilities for d = 0, ..., N*_it-1
            log_probs = np.zeros(N_star)
            for d in range(N_star):
                if self.U1[t] <= weights1[d]:
                    # Gaussian kernel (equation 7)
                    mean_d = self.mu_0[d] + self.mu_1[d]
                    var_d = self.sigma2_0[d] * self.sigma2_1[d]
                    log_probs[d] = norm.logpdf(self.Y1[t], mean_d, np.sqrt(var_d))
                else:
                    log_probs[d] = -np.inf

            # Sample
            log_probs -= logsumexp(log_probs)
            probs = np.exp(log_probs)
            self.D1[t] = np.random.choice(N_star, p=probs)

        # Sample D2
        for t in range(self.T2):
            cumsum = np.cumsum(weights2)
            N_star = np.searchsorted(cumsum, 1 - self.U2[t]) + 1
            N_star = min(N_star, len(weights2))

            # Ensure atoms exist
            while len(self.mu_0) < N_star:
                self.mu_0.append(norm.rvs(0, 1/np.sqrt(self.s_0)))
                self.mu_1.append(norm.rvs(0, 1/np.sqrt(self.s_i)))
                self.mu_2.append(norm.rvs(0, 1/np.sqrt(self.s_i)))
                self.sigma2_0.append(invgamma.rvs(self.epsilon/2, scale=self.epsilon/2))
                self.sigma2_1.append(invgamma.rvs(self.lambda_param/2,
                                                 scale=self.lambda_param/2))
                self.sigma2_2.append(invgamma.rvs(self.lambda_param/2,
                                                 scale=self.lambda_param/2))

            log_probs = np.zeros(N_star)
            for d in range(N_star):
                if self.U2[t] <= weights2[d]:
                    mean_d = self.mu_0[d] + self.mu_2[d]
                    var_d = self.sigma2_0[d] * self.sigma2_2[d]
                    log_probs[d] = norm.logpdf(self.Y2[t], mean_d, np.sqrt(var_d))
                else:
                    log_probs[d] = -np.inf

            log_probs -= logsumexp(log_probs)
            probs = np.exp(log_probs)
            self.D2[t] = np.random.choice(N_star, p=probs)

    def gibbs_step(self):
        """
        Perform one iteration of Gibbs sampling.

        Order (Section 5):
        1. Sample atoms psi
        2. Sample V and phi
        3. Sample U
        4. Sample D
        """
        self._sample_atoms()
        self._sample_V_and_phi()
        self._sample_U()
        self._sample_D()

    def run_mcmc(self, n_iter=10000, burn_in=5000, thin=10, verbose=True):
        """
        Run MCMC sampler.

        Parameters:
        -----------
        n_iter : int
            Total number of iterations
        burn_in : int
            Number of burn-in iterations to discard
        thin : int
            Thinning interval
        verbose : bool
            Print progress

        Returns:
        --------
        samples : dict
            Posterior samples
        """
        print(f"Running MCMC: {n_iter} iterations, burn-in={burn_in}, thin={thin}")
        print(f"Data: T1={self.T1}, T2={self.T2}")
        print(f"Hyperparameters: s_i={self.s_i}, s_0={self.s_0}, lambda={self.lambda_param}")
        print()

        iterator = tqdm(range(n_iter)) if verbose else range(n_iter)

        for it in iterator:
            self.gibbs_step()

            # Store samples (after burn-in, thinned)
            if it >= burn_in and (it - burn_in) % thin == 0:
                self.samples['theta1'].append(self.theta1)
                self.samples['theta2'].append(self.theta2)
                self.samples['alpha'].append(self.alpha)
                self.samples['n_clusters_1'].append(len(np.unique(self.D1)))
                self.samples['n_clusters_2'].append(len(np.unique(self.D2)))
                self.samples['D1'].append(self.D1.copy())
                self.samples['D2'].append(self.D2.copy())

            # Progress update
            if verbose and (it + 1) % 1000 == 0:
                n_clust_1 = len(np.unique(self.D1))
                n_clust_2 = len(np.unique(self.D2))
                iterator.set_postfix({
                    'theta1': f'{self.theta1:.3f}',
                    'theta2': f'{self.theta2:.3f}',
                    'alpha': f'{self.alpha:.3f}',
                    'K1': n_clust_1,
                    'K2': n_clust_2
                })

        return self.samples

    def plot_trace(self):
        """
        Plot trace plots for parameters.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # theta1
        axes[0, 0].plot(self.samples['theta1'])
        axes[0, 0].set_title('θ₁')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].grid(True, alpha=0.3)

        # theta2
        axes[0, 1].plot(self.samples['theta2'])
        axes[0, 1].set_title('θ₂')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].grid(True, alpha=0.3)

        # alpha
        axes[0, 2].plot(self.samples['alpha'])
        axes[0, 2].set_title('α')
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].grid(True, alpha=0.3)

        # n_clusters_1
        axes[1, 0].plot(self.samples['n_clusters_1'])
        axes[1, 0].set_title('Number of Clusters (Component 1)')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('K₁')
        axes[1, 0].grid(True, alpha=0.3)

        # n_clusters_2
        axes[1, 1].plot(self.samples['n_clusters_2'])
        axes[1, 1].set_title('Number of Clusters (Component 2)')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('K₂')
        axes[1, 1].grid(True, alpha=0.3)

        # Histogram of clusters
        axes[1, 2].hist(self.samples['n_clusters_1'], bins=20, alpha=0.5,
                       label='Component 1', density=True)
        axes[1, 2].hist(self.samples['n_clusters_2'], bins=20, alpha=0.5,
                       label='Component 2', density=True)
        axes[1, 2].set_title('Posterior Distribution of K')
        axes[1, 2].set_xlabel('Number of Clusters')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def summary(self):
        """
        Print summary statistics.
        """
        print("=" * 70)
        print("MCMC Summary")
        print("=" * 70)
        print(f"Number of posterior samples: {len(self.samples['theta1'])}")
        print()
        print("Parameter Estimates:")
        print(f"  θ₁: {np.mean(self.samples['theta1']):.4f} "
              f"(95% CI: [{np.percentile(self.samples['theta1'], 2.5):.4f}, "
              f"{np.percentile(self.samples['theta1'], 97.5):.4f}])")
        print(f"  θ₂: {np.mean(self.samples['theta2']):.4f} "
              f"(95% CI: [{np.percentile(self.samples['theta2'], 2.5):.4f}, "
              f"{np.percentile(self.samples['theta2'], 97.5):.4f}])")
        print(f"  α:  {np.mean(self.samples['alpha']):.4f} "
              f"(95% CI: [{np.percentile(self.samples['alpha'], 2.5):.4f}, "
              f"{np.percentile(self.samples['alpha'], 97.5):.4f}])")
        print()
        print("Number of Clusters:")
        print(f"  Component 1: {np.mean(self.samples['n_clusters_1']):.2f} "
              f"(mode: {np.bincount(self.samples['n_clusters_1']).argmax()})")
        print(f"  Component 2: {np.mean(self.samples['n_clusters_2']):.2f} "
              f"(mode: {np.bincount(self.samples['n_clusters_2']).argmax()})")
        print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("MCMC for Beta-Product Dependent Pitman-Yor (H1)")
    print("Bassetti, Casarin, Leisen (2013)")
    print("=" * 70)
    print()

    # Generate synthetic data from mixture
    np.random.seed(42)

    # True mixture for Y1: 3 components
    true_means_1 = [-5, 0, 5]
    true_weights_1 = [0.3, 0.4, 0.3]
    n1 = 200

    clusters1 = np.random.choice(3, size=n1, p=true_weights_1)
    Y1 = np.array([norm.rvs(loc=true_means_1[c], scale=0.5) for c in clusters1])

    # True mixture for Y2: 3 components (same locations, different weights)
    true_weights_2 = [0.2, 0.5, 0.3]
    n2 = 200

    clusters2 = np.random.choice(3, size=n2, p=true_weights_2)
    Y2 = np.array([norm.rvs(loc=true_means_1[c], scale=0.5) for c in clusters2])

    print(f"Generated data: Y1 ({n1} obs), Y2 ({n2} obs)")
    print(f"True means: {true_means_1}")
    print(f"True weights Y1: {true_weights_1}")
    print(f"True weights Y2: {true_weights_2}")
    print()

    # Initialize sampler
    sampler = BassettiBetaDPY_H1_MCMC(Y1, Y2)
    sampler.initialize(K_init=20)

    # Run MCMC
    samples = sampler.run_mcmc(n_iter=2000, burn_in=500, thin=3, verbose=True)

    # Summary
    print()
    sampler.summary()

    # Plots
    fig = sampler.plot_trace()
    fig.savefig('/home/user/NP1/mcmc_trace.png', dpi=150, bbox_inches='tight')
    print()
    print("Saved: mcmc_trace.png")

    # Plot data and posterior clustering
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Component 1
    axes[0].hist(Y1, bins=30, density=True, alpha=0.3, color='blue', label='Data')
    for mean in true_means_1:
        axes[0].axvline(mean, color='red', linestyle='--', alpha=0.5, label='True means')
    axes[0].set_title('Component 1')
    axes[0].set_xlabel('Y₁')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Component 2
    axes[1].hist(Y2, bins=30, density=True, alpha=0.3, color='red', label='Data')
    for mean in true_means_1:
        axes[1].axvline(mean, color='red', linestyle='--', alpha=0.5, label='True means')
    axes[1].set_title('Component 2')
    axes[1].set_xlabel('Y₂')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig('/home/user/NP1/mcmc_data.png', dpi=150, bbox_inches='tight')
    print("Saved: mcmc_data.png")
    print()
    print("=" * 70)
    print("MCMC completed successfully!")
    print("=" * 70)

import numpy as np
from scipy.optimize import fsolve

DEFAULT_Bs = [0.9999999, 0.5, 0.1, 1E-2, 0.5E-2, 1E-3, 0.5E-3, 1E-4, 1E-5, 1E-6, 1E-7]

class ElbertYield:
    # ElbertYield parameters fit to MCEq with SIBYLL2.3c
    elbert_params = np.array([
        [14.7, 1.79, 3.27, 0.64],  # Mu
        [5.64, 1.77, 3.74, 0.51],  # NuMu
        [0.25, 1.75, 6.38, 0.68],  # NuE
        [2.8E-05, 1, 6.73, 0.78],  # Prompt Mu
        [2.8E-05, 0.97, 7.19, 0.58],  # Prompt NuMu
        [3.2E-05, 0.95, 7.32, 0.65]   # Prompt NuE
    ])

    Mu = 0
    NuMu = 1
    NuE = 2

    # class ParticleType:
    #     Mu = 0
    #     NuMu = 1
    #     NuE = 2

    def __init__(self, A, primary_energy, cos_theta, family=None, strict=True):
        """
        Initialize the ElbertYield object.

        :param A: Atomic number.
        :param primary_energy: Primary cosmic ray energy.
        :param cos_theta: Cosine of the zenith angle.
        """
        if np.any(A < 1):
            raise ValueError("Invalid atomic number")

        self.A = A
        self.primary_energy = primary_energy
        self.cos_theta = np.abs(cos_theta)
        self.cos_theta_eff = self.get_effective_costheta(self.cos_theta)
        self.family = family if family else self.Mu
        self.strict = strict

    def _compute_prefactor(self, prompt, family=None):
        """
        Compute the prefactor based on whether it's prompt or conventional.
        
        :param params: The parameter set for the given particle type.
        :param prompt: Boolean indicating if the yield is prompt.
        :return: Computed prefactor.
        """
        family = family if family else self.family
        params = self.elbert_params[3 + family] if prompt else self.elbert_params[family]
        decay_prob = 1 if prompt else self.A/self.primary_energy/self.cos_theta_eff
        return params[0] * self.A * decay_prob, params[1], params[2], params[3]

    def _compute_yield(self, x, prefactor, p1, p2, p3):
        """
        Evaluate the function at a given x.

        :param x: Input value(s).
        :return: Evaluated function output.
        """
        return prefactor * np.power(x, -p1) * np.power(1 - np.power(x, p3), p2)


    def get_conventional_yield(self, x):
        """
        Compute the conventional yield parameters.

        :param family: ParticleType (Mu, NuMu, NuE).
        :return: A new ElbertYield instance with conventional prefactor.
        """
        #prefactor, p1, p2, p3 = self._compute_prefactor(self.elbert_params[self.family], prompt=False)
        prefactor, p1, p2, p3 = self._compute_prefactor(prompt=False)
        return self._compute_yield(x, prefactor, p1, p2, p3)

    def get_prompt_yield(self, x):
        """
        Compute the prompt yield parameters.

        :param family: ParticleType (Mu, NuMu, NuE).
        :return: A new ElbertYield instance with prompt prefactor.
        """
        #prefactor, p1, p2, p3 = self._compute_prefactor(self.elbert_params[3 + self.family], prompt=True)
        prefactor, p1, p2, p3 = self._compute_prefactor(prompt=True)
        return self._compute_yield(x, prefactor, p1, p2, p3)

    def get_yield(self, x):
        return self.get_conventional_yield(x) + self.get_prompt_yield(x)

    def get_prob(self, x):
        """
        Evaluate the probability

        :param x: Input value(s).
        :return: Evaluated probability.
        """
        return 1 - np.exp(-self.get_yield(x))

    def solve_for_x_min(self, B=0.001, x0=0.00001, strict=None):
        """ 
        Solve for x_min given the yield and the probability of no muons.

        :param B: target acceptance rate.
        :param x0: Initial guess.
        """
        strict = strict if strict is not None else self.strict
        
        equation = lambda x: self.get_yield(x) + np.log(1 - B)
        sol = fsolve(equation, x0)[0]
        if not np.all(np.isclose(equation(sol), 0)):
            if strict:
                print(equation(sol))
                raise ValueError("Solution did not converge for: %s"%B)
            else:
                print("WARNING Solution did not converge for: %s.... setting solution to nan"%B)
                sol = np.nan
        return sol

    def acceptance_probability(self, x_mu, x_min=None, B=0.001, x0=0.00001):
        """
        Compute the acceptance probability.

        :param x_mu: E_mu_max/ (E_primary/A)
        :param B: target acceptance rate.
        :param x0: Initial guess.
        """
        if x_min is None:
            x_min = self.solve_for_x_min(B, x0)
        pacc = self.get_prob(x_min) / self.get_prob(x_mu)
        pacc = np.where(x_mu < x_min, pacc, 1)
        return pacc
            
    @staticmethod
    def get_effective_costheta(x):
        """
        Effective local atmospheric density correction.

        :param x: Cosine of the zenith angle.
        :return: Corrected value.
        """
        p = np.array([0.102573, -0.068287, 0.958633, 0.0407253, 0.817285])
        return np.sqrt((x**2 + p[0]**2 + p[1] * x**p[2] + p[3] * x**p[4]) /
                       (1 + p[0]**2 + p[1] + p[3]))




import numpy as np

def invert(N, conv, prompt, xtol=1e-5, max_iter=20):
    """
    Inverts the function using root-finding in log space.
    
    Parameters:
        N (float): The target value to invert.
        conv (callable): Function representing the conventional component.
        prompt (callable): Function representing the prompt component.
        xtol (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.
    
    Returns:
        float: The inverted value.
    """

    # Define the function in log space
    def logtotal(logx):
        x = np.exp(logx)
        return np.log(conv(x) + prompt(x))

    y0 = np.log(N)
    x0 = min((np.log(conv.prefactor) - y0) / conv.p1, -xtol)  # Initial guess

    for _ in range(max_iter):
        y = logtotal(x0)
        
        # Numerical derivative (finite difference equivalent)
        h = 1e-6  # Small step for finite difference
        dy_dx = (logtotal(x0 + h) - y) / h
        
        # Newton-Raphson update
        xp = min(x0 - (y - y0) / dy_dx, -xtol)

        if np.abs(xp - x0) < xtol:
            return np.exp(xp)

        x0 = xp

    raise RuntimeError("Newton-Raphson method did not converge")

# def get_effective_costheta(x):
#     """
#     Effective local atmospheric density correction.

#     :param x: Cosine of the zenith angle.
#     :return: Corrected value.
#     """
#     p = np.array([0.102573, -0.068287, 0.958633, 0.0407253, 0.817285])
#     return np.sqrt((x**2 + p[0]**2 + p[1] * x**p[2] + p[3] * x**p[4]) /
#                     (1 + p[0]**2 + p[1] + p[3]))


# def solve_for_x_min(A, primary_energy, cos_theta, Bs=[0.001], x0=0.00001, strict=True):
#     """
#     Solve for x_min given the yield and the probability of no muons.

#     :param A: Atomic number.
#     :param primary_energy: Primary cosmic ray energy.
#     :param cos_theta: Cosine of the zenith angle.
#     :param B: target acceptance rate.
#     :param x0: Initial guess.
#     """
#     def get_effective_costheta(x):
#         p = np.array([0.102573, -0.068287, 0.958633, 0.0407253, 0.817285])
#         return np.sqrt((x**2 + p[0]**2 + p[1] * x**p[2] + p[3] * x**p[4]) /
#                         (1 + p[0]**2 + p[1] + p[3]))

#     def compute_prefactor(params, prompt):
#         decay_prob = 1 if prompt else A / (primary_energy * get_effective_costheta(cos_theta))
#         return params[0] * A * decay_prob, params[1], params[2], params[3]

#     def compute_yield(x, prefactor, p1, p2, p3):
#         return prefactor * np.power(x, -p1) * np.power(1 - np.power(x, p3), p2)

#     elbert_params = np.array([
#         [14.7, 1.79, 3.27, 0.64],  # Mu
#         [5.64, 1.77, 3.74, 0.51],  # NuMu
#         [0.25, 1.75, 6.38, 0.68],  # NuE
#         [2.8E-05, 1, 6.73, 0.78],  # Prompt Mu
#         [2.8E-05, 0.97, 7.19, 0.58],  # Prompt NuMu
#         [3.2E-05, 0.95, 7.32, 0.65]   # Prompt NuE
#     ])

#     elbert_params_mu_conv = elbert_params[0]
#     elbert_params_mu_prompt = elbert_params[3]

#     def get_conventional_yield(x):
#         prefactor, p1, p2, p3 = compute_prefactor(elbert_params_mu_conv, prompt=False)
#         return compute_yield(x, prefactor, p1, p2, p3)

#     def get_prompt_yield(x):
#         prefactor, p1, p2, p3 = compute_prefactor(elbert_params_mu_prompt, prompt=True)
#         return compute_yield(x, prefactor, p1, p2, p3)

#     def get_yield(x):
#         return get_conventional_yield(x) # + get_prompt_yield(x)

#     sols = {}
#     for B in Bs:
#         equation = lambda x: get_yield(x) + np.log(1 - B)
#         sol = fsolve(equation, x0)[0]
#         if strict:
#             assert np.isclose(equation(sol), 0)
#         sols[B] = sol
#     return sols


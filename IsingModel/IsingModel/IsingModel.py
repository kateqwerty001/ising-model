import numpy as np
import networkx as nx


class IsingModel:
    def __init__(self, adjacency_matrix, name, T, interaction_type=None, steps=100_000, burn_in=20_000, sample_every=100, seed=None, external_field=None):
        """
        Model Isinga z próbnikem Gibbsa dla dowolnego grafu.

        Parametry:
        - adjacency_matrix: macierz sąsiedztwa grafu (numpy array)
        - name: nazwa grafu (do wyświetlania w wynikach)
        - T: temperatura (float)
        - interaction_type: 'ferro', 'antiferro', 'spin_glass'
        - steps: liczba kroków próbnika Gibbsa
        - burn_in: liczba kroków przed rozpoczęciem pomiarów
        - sample_every: co ile kroków zbierać magnetyzację
        - seed: losowe ziarno
        """
        np.random.seed(seed)
        self.adjacency_matrix = adjacency_matrix
        self.name = name
        self.N = adjacency_matrix.shape[0]
        self.steps = steps
        self.burn_in = burn_in
        self.sample_every = sample_every
        self.T = T
        self.beta = 1.0 / T
        self.interaction_type = interaction_type
        self.external_field = external_field 
        self.J, self.h = self._generate_ising_parameters()

    def _generate_ising_parameters(self):
        """
        Generuje interakcje J_ij oraz pola h_i w zależności od typu interakcji.
        """
        J = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.adjacency_matrix[i, j]:
                    if self.interaction_type == 'ferro':
                        p = np.random.uniform(0, 1)
                    elif self.interaction_type == 'antiferro':
                        p = np.random.uniform(-1, 0)
                    elif self.interaction_type == 'spin_glass' or self.interaction_type == None:
                        p = np.random.uniform(-1, 1)
                    else:
                        raise ValueError("Invalid interaction_type. Choose 'ferro', 'antiferro', or 'spin_glass'.")
                    J[i, j] = p
                    J[j, i] = p

        if self.external_field is None:
            h = np.random.uniform(-0.5, 0.5, size=self.N)
        else:
            h = np.full(self.N, self.external_field)
        return J, h

    def run_gibbs_sampler(self):
        np.random.seed(None)
        spins = np.ones(self.N) 
        magnetizations = []

        for step in range(self.steps):
            selected_spin = np.random.randint(self.N)
            H = np.sum(self.J[selected_spin] * spins) + self.h[selected_spin]
            p = 1 / (1 + np.exp(-2 * self.beta * H))
            spins[selected_spin] = 1 if np.random.rand() < p else -1

            if step >= self.burn_in and (step - self.burn_in) % self.sample_every == 0:
                m = np.sum(spins) / self.N
                magnetizations.append(m)

        return np.abs(np.mean(magnetizations)), np.std(magnetizations)

    def simulate(self):
        m_mean, m_std = self.run_gibbs_sampler()
        print(f"{self.name} | T = {self.T:.2f}, Type = {self.interaction_type}, Magnetization = {m_mean:.4f}")
        return self.T, m_mean, m_std

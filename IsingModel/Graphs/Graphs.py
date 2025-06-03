import numpy as np
import networkx as nx

class Graph:
    def get_ajacency_matrix(self):
        return self.A
    
    def _generate_ajacency_matrix(self):
        raise NotImplementedError("Subclasses must implement this method")

class LatticeGraph(Graph):
    # Graf regularnej dwuwymiarowej kraty L×L przy użyciu NetworkX.
    # Każdy wierzchołek ma do 4 sąsiadów (góra, dół, lewo, prawo).
    def __init__(self, L):
        self.L = L
        self.A = self._generate_ajacency_matrix()

    def _generate_ajacency_matrix(self):
        G = nx.grid_2d_graph(self.L, self.L)  # Graf 2D
        mapping = {(i, j): i * self.L + j for i in range(self.L) for j in range(self.L)}
        G = nx.relabel_nodes(G, mapping)  # Zamiana (i,j) -> indeks całkowity
        A = nx.to_numpy_array(G, dtype=int)
        return A


class CompleteGraph(Graph):
    # Graf pełny o N wierzchołkach – każdy wierzchołek połączony z każdym innym.
    # Modeluje sytuację, w której oddziaływują wszystkie spiny ze sobą.
    def __init__(self, N):
        self.N = N
        self.A = self._generate_ajacency_matrix()

    def _generate_ajacency_matrix(self):
        N = self.N
        A = np.ones((N, N), dtype=int) - np.eye(N, dtype=int)
        return A

class ErdosRenyiGraph(Graph):
    # Losowy graf G(n, p), gdzie każda para wierzchołków jest połączona z prawdopodobieństwem p.
    # Modeluje nieuporządkowane interakcje – sieci z przypadkowymi połączeniami.    
    def __init__(self, N, p):
        self.N = N
        self.p = p
        self.A = self._generate_ajacency_matrix()

    def _generate_ajacency_matrix(self):
        G = nx.erdos_renyi_graph(self.N, self.p)
        A = nx.to_numpy_array(G, dtype=int)
        return A

class WattsStrogatzGraph(Graph):
    # Graf małego świata – zaczyna się od regularnego pierścienia (każdy z k sąsiadami),
    # a następnie część krawędzi jest losowo przetasowana z prawdopodobieństwem p.
    # Reprezentuje systemy z lokalną strukturą i nielicznymi dalekimi połączeniami.
    def __init__(self, N, k, p):
        self.N = N
        self.k = k
        self.p = p
        self.A = self._generate_ajacency_matrix()

    def _generate_ajacency_matrix(self):
        G = nx.watts_strogatz_graph(self.N, self.k, self.p)
        A = nx.to_numpy_array(G, dtype=int)
        return A

class BarabasiAlbertGraph(Graph):
    # Graf ze wzrostem preferencyjnym – nowe wierzchołki dołączają do istniejących
    # z prawdopodobieństwem proporcjonalnym do ich stopnia.
    # Tworzy sieci ze strukturą skali – kilka wysoko połączonych węzłów (huby).
    def __init__(self, N, m):
        self.N = N
        self.m = m
        self.A = self._generate_ajacency_matrix()

    def _generate_ajacency_matrix(self):
        G = nx.barabasi_albert_graph(self.N, self.m)
        A = nx.to_numpy_array(G, dtype=int)
        return A
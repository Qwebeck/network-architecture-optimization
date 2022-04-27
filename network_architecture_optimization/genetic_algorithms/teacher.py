import numpy as np
from sklearn.model_selection import train_test_split
from network_architecture_optimization.mappers.network_builder import NetworkBuilder


class Teacher:

    def __init__(self, X: np.ndarray, y: np.ndarray, n_evaluations, network_builder: NetworkBuilder):
        self._input_shape = X[0].shape
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y)
        self._n_evaluations = n_evaluations
        self.network_builder = network_builder

    def map_chromosome_to_network_architecture(self, chromosome: np.ndarray):
        data = {'x_train': self.x_train, 'x_test': self.x_test, 'y_train': self.y_train, 'y_test': self.y_test}
        return self.network_builder.map_chromosome(self._input_shape, chromosome, data)

    def fitness_function(self, solution: np.ndarray, solution_idx: int):
        results = [self._evaluate_chromosome(solution) for _ in range(self._n_evaluations)]
        return np.array(results).mean()

    def _evaluate_chromosome(self, chromosome):
        model = self.map_chromosome_to_network_architecture(chromosome)
        model.fit(self.x_train, self.y_train, epochs=10)
        _, acc = model.evaluate(self.x_test,  self.y_test, verbose=2)
        return acc

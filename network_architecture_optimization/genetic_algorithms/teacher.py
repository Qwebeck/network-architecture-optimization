import numpy as np
from sklearn.model_selection import train_test_split
from network_architecture_optimization.mappers.architecture_mapper import AchitectureMapper


class Teacher:

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self._input_shape = X[0].shape
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y)
        self.architecture_mapper = AchitectureMapper()

    def create_initial_population(self) -> np.ndarray:
        return np.array([
            [1, 0, 1, 0, 2, 0, 3, 1],
            [3, 2, 1, 1, 1, 2, 1, 0],
            [1, 3, 1, 1, 0, 3, 1, 1],
            [1, 1, 3, 1, 1, 0, 3, 1]
        ], dtype=int)

    def map_chromosome_to_network_architecture(self, chromosome: np.ndarray):
        return self.architecture_mapper.map_chromosome(self._input_shape, chromosome)

    def fitness_function(self, solution: np.ndarray, solution_idx: int):
        model = self.map_chromosome_to_network_architecture(solution)
        print(model.summary())
        model.fit(self.x_train, self.y_train, epochs=50)
        _, acc = model.evaluate(self.x_test,  self.y_test, verbose=2)
        return acc

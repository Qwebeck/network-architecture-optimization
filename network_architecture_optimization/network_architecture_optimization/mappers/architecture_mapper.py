import tensorflow as tf
from typing import List
import numpy as np
from network_architecture_optimization.mappers.base_gene_mapper import GeneMapper
from network_architecture_optimization.mappers.gene_mappers import CoreLayerMapper
from network_architecture_optimization.ray_tuning.hyperparameter_optimization import HyperparameterOptimizer


class AchitectureMapper:
    def __init__(self):
        self.initial_mapper: GeneMapper = CoreLayerMapper()

    def map_chromosome(self, input_shape, chromosome: np.ndarray, data):

        inputs, outputs = self._build_layers(input_shape, chromosome)
        return self._build_model(inputs, outputs, data)

    def _build_layers(self, input_shape, chromosome):
        inputs = tf.keras.Input(shape=input_shape)
        body = tf.keras.layers.Flatten()(inputs)
        mapper = self.initial_mapper
        for gene in chromosome:
            body = mapper.map_gene_to_layer(body, gene)
            mapper = mapper.next_mapper(gene)
        outputs = tf.keras.layers.Dense(10)(body)
        outputs = tf.keras.layers.Softmax()(outputs)
        return inputs, outputs

    def _build_model(self, inputs, outputs, data):
        model = HyperparameterOptimizer(inputs, outputs, data).get_model()
        
        return model

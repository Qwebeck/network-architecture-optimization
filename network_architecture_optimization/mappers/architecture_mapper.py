import tensorflow as tf
from typing import List
import numpy as np
from network_architecture_optimization.mappers.base_gene_mapper import GeneMapper
from network_architecture_optimization.mappers.gene_mappers import CoreLayerMapper


class AchitectureMapper:
    def __init__(self):
        self.initial_mapper: GeneMapper = CoreLayerMapper()

    def map_chromosome(self, input_shape, chromosome: np.ndarray):

        inputs, outputs = self._build_layers(input_shape, chromosome)
        return self._build_model(inputs, outputs)

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

    def _build_model(self, inputs, outputs):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam',
                      loss=loss_fn,
                      metrics=['accuracy'])
        return model

from network_architecture_optimization.mappers.gene_mapper import GeneMapper
from network_architecture_optimization.mappers.layer_description import LayerDescription
import tensorflow as tf
from typing import List
import numpy as np
# from network_architecture_optimization.ray_tuning.hyperparameter_optimization import HyperparameterOptimizer
from dataclasses import fields


class NetworkBuilder:
    def __init__(self):
        self.mapper = GeneMapper()

    def map_chromosome(self, input_shape, chromosome: np.ndarray, data):
        inputs, outputs = self._build_layers(input_shape, chromosome)
        return self._build_model(inputs, outputs)

    def _build_layers(self, input_shape, chromosome):
        layer_description_size = len(fields(LayerDescription))
        assert len(chromosome) % layer_description_size == 0, "chromosome should be dividable by gene size, but it is not."
        inputs = tf.keras.Input(shape=input_shape)
        body = tf.keras.layers.Flatten()(inputs)
        for i in range(0, len(chromosome), layer_description_size):
            layer_description = LayerDescription(*chromosome[i: i+layer_description_size])
            body = self.mapper.map_layer(body, layer_description)
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

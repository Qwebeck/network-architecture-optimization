from network_architecture_optimization.mappers.layer_description import LayerDescription
from dataclasses import asdict, field
from typing import Callable
import tensorflow as tf


class GeneMapper:

    kernel_initializer = tf.keras.initializers.GlorotUniform(seed=None)
    drop_rate = 0.5

    def map_layer(self, model, layer_description: LayerDescription):
        """
        Maps gene on layer base on existing model
        :param model: already built model
        :param gene: gene to be mapped
        """

        for layer_type, gene_value in asdict(layer_description).items():
            mapper = self.get_gene_mapper(layer_type)
            layer = mapper(gene_value)
            if layer is None:
                continue
            model = layer(model)
        return model

    def get_gene_mapper(self, gene_name: str) -> Callable[[int], tf.keras.layers.Layer]:
        mappers = {
            'weight_initialization_gene': self.map_weight_init_gene,
            'dense_layer_gene': self.map_dense_layer_gene,
            'activation_gene': self.map_activation_gene,
            'dropout_rate_gene': self.map_dropout_rate_gene,
            'dropout_gene': self.map_dropout_gene,
            'normalization_gene': self.map_normalization_gene,
        }
        expected_gene_names = mappers.keys()
        assert gene_name in expected_gene_names, f"{gene_name} is an unkown gene. Genes that could be mapped are: f{expected_gene_names}"
        return mappers[gene_name]

    def map_weight_init_gene(self, gene: int):

        available_initializations = [
            tf.keras.initializers.GlorotUniform(seed=None),
            tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
            tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
            tf.keras.initializers.Zeros()
        ]

        self.kernel_initializer = available_initializations[gene % len(available_initializations)]

    def map_dense_layer_gene(self, gene: int) -> tf.keras.layers.Layer:

        available_layers = [
            tf.keras.layers.Dense(64, kernel_initializer=self.kernel_initializer),
            tf.keras.layers.Dense(32, kernel_initializer=self.kernel_initializer),
            tf.keras.layers.Dense(16, kernel_initializer=self.kernel_initializer),
            tf.keras.layers.Dense(8, kernel_initializer=self.kernel_initializer)
        ]
        return available_layers[gene % len(available_layers)]

    def map_activation_gene(self, gene: int) -> tf.keras.layers.Layer:

        available_layers = [
            tf.keras.layers.Activation(tf.keras.activations.relu),
            tf.keras.layers.Activation(tf.keras.activations.sigmoid),
            tf.keras.layers.Activation(tf.keras.activations.tanh),
            tf.keras.layers.Activation(tf.keras.activations.linear)
        ]
        return available_layers[gene % len(available_layers)]

    def map_dropout_rate_gene(self, gene: int) -> tf.keras.layers.Layer:

        available_values = [0.1, 0.3, 0.5, 0.7]
        self.drop_rate = available_values[gene % len(available_values)]

    def map_dropout_gene(self, gene: int) -> tf.keras.layers.Layer:
        if gene == 0:
            return None

        available_layers = [
            tf.keras.layers.Dropout(rate=self.drop_rate),
            tf.keras.layers.GaussianDropout(rate=self.drop_rate),
            tf.keras.layers.AlphaDropout(rate=self.drop_rate)
        ]
        return available_layers[(gene + 1) % len(available_layers)]

    def map_normalization_gene(self, gene: int) -> tf.keras.layers.Layer:
        if gene == 0:
            return None

        available_layers = [
            tf.keras.layers.experimental.SyncBatchNormalization(),
            tf.keras.layers.BatchNormalization()
        ]

        return available_layers[(gene + 1) % len(available_layers)]

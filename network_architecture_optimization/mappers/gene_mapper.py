from abc import ABC, abstractmethod
from typing import Callable

from network_architecture_optimization.mappers.layer_description import LayerDescription
from dataclasses import asdict
import tensorflow as tf


class GeneMapper:

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
            'dense_layer_gene': self.map_dense_layer_gene,
            'activation_gene': self.map_activation_gene,
            'dropout_gene': self.map_dropout_gene,
            'pooling_gene': self.map_pooling_gene,
            'normalization_gene': self.map_normalization_gene,
        }
        expected_gene_names = mappers.keys()
        assert gene_name in expected_gene_names, f"{gene_name} is an unkown gene. Genes that could be mapped are: f{expected_gene_names}"
        return mappers[gene_name]

    def map_dense_layer_gene(self, gene: int) -> tf.keras.layers.Layer:
        if gene == 0:
            return None
        available_layers = [
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(16),
            tf.keras.layers.Dense(8)
        ]
        return available_layers[(gene + 1) % len(available_layers)]

    def map_activation_gene(self, gene: int) -> tf.keras.layers.Layer:
        if gene == 0:
            return None
        avaialbe_layers = [
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.GaussianDropout(rate=0.5),
            tf.keras.layers.AlphaDropout(rate=0.5)
        ]
        return avaialbe_layers[(gene + 1) % len(avaialbe_layers)]

    def map_dropout_gene(self, gene: int) -> tf.keras.layers.Layer:
        ...

    def map_pooling_gene(self, gene: int) -> tf.keras.layers.Layer:
        ...

    def map_normalization_gene(self, gene: int) -> tf.keras.layers.Layer:
        ...

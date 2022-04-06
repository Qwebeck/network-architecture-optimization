from network_architecture_optimization.mappers.base_gene_mapper import GeneMapper
import tensorflow as tf


class RegularizationLayerMapper(GeneMapper):
    _available_layers = [
        tf.keras.layers.Dropout,
        tf.keras.layers.GaussianDropout,
        tf.keras.layers.AlphaDropout
    ]

    def map_gene_to_layer(self, model, gene):
        if gene == 0:
            return model
        layer = self._available_layers[gene % len(self._available_layers)](rate=0.5)
        return layer(model)

    def next_mapper(self, gene):
        return CoreLayerMapper()


class CoreLayerMapper(GeneMapper):

    def map_gene_to_layer(self, model, gene):
        if gene == 0:
            return model
        dense = [
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(16),
            tf.keras.layers.Dense(8)
        ][gene % 4]
        return dense(model)

    def next_mapper(self, gene):
        return RegularizationLayerMapper() if gene != 0 else self

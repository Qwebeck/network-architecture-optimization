from dataclasses import dataclass


@dataclass
class LayerDescription:
    dense_layer_gene: int
    activation_gene: int
    dropoutRate_gene: int
    dropout_gene: int
    weightInit_gene: int
    normalization_gene: int

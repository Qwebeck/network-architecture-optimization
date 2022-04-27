from dataclasses import dataclass, field
import tensorflow as tf


@dataclass
class LayerDescription:
    dense_layer_gene: int
    activation_gene: int
    dropout_rate_gene: int
    dropout_gene: int
    weight_initialization_gene: int
    normalization_gene: int



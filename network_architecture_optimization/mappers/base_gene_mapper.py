from abc import ABC, abstractmethod


class GeneMapper(ABC):
    @abstractmethod
    def map_gene_to_layer(self, model, gene):
        """
        Maps gene on layer base on existing model
        :param model: already built model
        :param gene: gene to be mapped
        """
        ...

    @abstractmethod
    def next_mapper(self, gene) -> 'GeneMapper':
        """
        Returns mappers for layers that could be used after this layer
        """
        ...

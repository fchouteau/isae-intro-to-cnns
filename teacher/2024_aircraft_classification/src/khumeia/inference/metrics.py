from typing import Iterator

from khumeia.roi.tile import PredictionTile


class Metric:
    """"""

    def compute(self, tiles):
        """

        Args:
            tiles([PredictionTile]):

        Returns:

        """
        raise NotImplementedError

    def __call__(self, tiles):
        """

        Args:
            tiles(Iterator[PredictionTile]):

        Returns:

        """
        return self.compute(tiles)


class Precision(Metric):
    def compute(self, tiles):
        """

        Args:
            tiles(Iterator[PredictionTile]):

        Returns:

        """
        tp = sum(map(lambda tile: 1 if tile.is_true_positive else 0, tiles))
        fp = sum(map(lambda tile: 1 if tile.is_false_positive else 0, tiles))
        fn = sum(map(lambda tile: 1 if tile.is_false_negative else 0, tiles))

        return 0.0 if tp == 0 else tp / (tp + fp)


class Recall(Metric):
    def compute(self, tiles):
        """

        Args:
            tiles(Iterator[PredictionTile]):

        Returns:

        """
        tp = sum(map(lambda tile: 1 if tile.is_true_positive else 0, tiles))
        fp = sum(map(lambda tile: 1 if tile.is_false_positive else 0, tiles))
        fn = sum(map(lambda tile: 1 if tile.is_false_negative else 0, tiles))

        return 0.0 if tp == 0 else tp / (tp + fn)


class FBeta(Metric):
    def __init__(self, beta):
        self.beta = beta

    def compute(self, tiles):
        """

        Args:
            tiles(Iterator[PredictionTile]):

        Returns:

        """
        tp = sum(map(lambda tile: 1 if tile.is_true_positive else 0, tiles))
        fp = sum(map(lambda tile: 1 if tile.is_false_positive else 0, tiles))
        fn = sum(map(lambda tile: 1 if tile.is_false_negative else 0, tiles))

        recall = 0.0 if tp == 0 else tp / (tp + fn)
        precision = 0.0 if tp == 0 else tp / (tp + fp)

        fbeta = (
            0.0 if tp == 0 else ((1 + self.beta ** 2) * precision * recall) / ((self.beta ** 2 * precision) + recall)
        )

        return fbeta


precision = Precision()
recall = Recall()
f1 = FBeta(1.0)
f2 = FBeta(2.0)

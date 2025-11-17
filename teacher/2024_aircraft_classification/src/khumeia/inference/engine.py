from khumeia import LOGGER
from khumeia.data.dataset import Dataset
from khumeia.inference.predictor import Predictor
from khumeia.roi.tile import PredictionTile
from tqdm.auto import tqdm


class InferenceEngine:
    """
    Classe qui se comporte comme `Dataset` mais donne accès à la prédiction sur chaque tuile à l'aide d'une fenêtre glissante

    ![](https://cdn-images-1.medium.com/max/1600/1*uLk0eLyS8sYCqXTgEYcO6w.png)
    """

    def __init__(self, sliding_windows, predictor):
        """

        Args:
            sliding_windows(SlidingWindow): The sliding window used to generate candidates
            predictor(Predictor): A Predictor object that encapsulates our model

        """
        if not isinstance(sliding_windows, (list, tuple)):
            sliding_windows = [sliding_windows]

        self.sliding_windows = sliding_windows
        self.predictor = predictor

    def predict_on_item(self, item):
        """

        Args:
            item(SatelliteImage): the item on which to apply the prediction

        Returns:

        """

        LOGGER.info("Generating tiles to predict")

        item_dataset = Dataset(items=[item])
        tiles = Dataset(items=[])
        for sliding_window in tqdm(self.sliding_windows, position=0, desc="Applying slider"):
            tiles = tiles.extend(item_dataset.flatmap(sliding_window))

        tiles = tiles.apply(lambda items: list(set(items)))

        LOGGER.info("Generating predicting on item {} with {} tiles".format(item.key, len(tiles)))

        image = item.image

        def _batch(items):
            return [items[i : i + self.predictor.batch_size] for i in range(0, len(items), self.predictor.batch_size)]

        batches = tiles.apply(_batch)

        print(len(tiles))
        print(len(batches))

        def _predict(batch):
            batch_data = list(map(lambda tile: tile.get_data(image), batch))
            batch_results = self.predictor.predict_on_batch(batch_data)
            batch_results = list(
                map(
                    lambda tpl: PredictionTile.from_labelled_tile_and_prediction(tpl[0], tpl[1]),
                    zip(batch, batch_results),
                )
            )

            return batch_results

        tiles_results = batches.flatmap(_predict, desc="Predicting on batch")

        return tiles_results

    def predict_on_dataset(self, items):
        """
            Apply predictor + sliding window on all items in self.items
        Args:
            items(Dataset):

        Returns:
            A list of PredictionTile (Tile + predicted_label + groundtruth label)

        """
        return items.flatmap(self.predict_on_item)

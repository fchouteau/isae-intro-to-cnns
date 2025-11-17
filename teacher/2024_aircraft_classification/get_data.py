from pathlib import Path
from khumeia.helpers import data_download

if __name__ == "__main__":
    raw_data_dir = Path(__file__).parent / "data" / "usgs"

    data_download.download_data(raw_data_dir)
    data_download.make_image_ids(raw_data_dir)
    data_download.make_labels(raw_data_dir, "trainval")
    data_download.make_labels(raw_data_dir, "eval")

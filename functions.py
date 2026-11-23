from functools import cache
from pathlib import Path
import pickle
from ucimlrepo import fetch_ucirepo
from retail_dataset import RetailDataset


@cache
def get_uci_dataset(dataset_id=352):
    cache_file = Path("retail-dataset.pkl")

    if cache_file.exists():
        try:
            with cache_file.open("rb") as file:
                return pickle.load(file)
        except (pickle.UnpicklingError, EOFError, OSError):
            cache_file.unlink(missing_ok=True)

    dataset = fetch_ucirepo(id=dataset_id)

    with cache_file.open("wb") as file:
        pickle.dump(dataset, file)

    return dataset


def get_retail_dataset():
    dataset = RetailDataset(get_uci_dataset().data.original)
    selected_code = "20725"
    dataset.set_selected_code(selected_code)
    return dataset

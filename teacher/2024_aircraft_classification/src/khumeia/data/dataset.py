import itertools
from typing import Any, Callable, List, Optional

from tqdm.auto import tqdm

try:
    import joblib
except ImportError:
    joblib = None


class Dataset:
    def __init__(self, items: [Any]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int) -> Any:
        return self.items[item]

    def map(self, func: Callable, desc: Optional[str] = None, n_jobs=1) -> "Dataset":
        if n_jobs > 1 and joblib is not None:
            items = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(func)(item) for item in tqdm(self.items, desc=desc))
        else:
            items = list(map(func, tqdm(self.items, desc=desc)))
        return Dataset(items=list(items))

    def flatmap(self, func: Callable, desc: Optional[str] = None, n_jobs=1) -> "Dataset":
        if n_jobs > 1 and joblib is not None:
            items = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(func)(item) for item in tqdm(self.items, desc=desc))
        else:
            items = list(map(func, tqdm(self.items, desc=desc)))
        items = list(itertools.chain.from_iterable(items))
        return Dataset(items=list(items))

    def filter(self, func: Callable[[Any], bool], desc=None) -> "Dataset":
        return Dataset(items=list(filter(func, tqdm(self.items, desc=desc))))

    def apply(self, func: Callable[[List[Any]], List[Any]]) -> "Dataset":
        return Dataset(items=func(self.items))

    def extend(self, dataset: "Dataset") -> "Dataset":
        return Dataset(items=self.items + dataset.items)

    def append(self, item: Any):
        self.items.append(item)

    def sorted(self, key: Optional[Callable] = None):
        if key is not None:
            self.items = sorted(self.items, key=key)
        else:
            self.items = sorted(self.items)

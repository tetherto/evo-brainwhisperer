"""Per-dataset NeuralSet recipes.

Each entry of `RECIPES` is a stateless `Recipe` instance (see `base.py`)
exposing `build_val_dataset(max_label_len)`, `build_train_dataset(max_label_len)`,
`build_chain(split)` and the model-shape metadata:
`NUM_DAYS` (day-axis size of the embedder, derived from the dataset's
`tether_scratch` session rule), `CHANNEL_FILL` (the fill-to-512 rule) and
`calendar_days()` (per-day calendar metadata the embedders' day encodings
consume). `train.py`/`eval.py`/`eval_lightbeam.py` reach every dataset
through this registry.
"""

from concurrent.futures import ThreadPoolExecutor

from brainwhisperer.data.recipes.base import Recipe
from brainwhisperer.data.recipes.card import CardRecipe
from brainwhisperer.data.recipes.jude import JudeRecipe
from brainwhisperer.data.recipes.kunz_t12 import KunzT12Recipe
from brainwhisperer.data.recipes.kunz_t15 import KunzT15Recipe
from brainwhisperer.data.recipes.wairagkar import WairagkarRecipe
from brainwhisperer.data.recipes.willet import WilletRecipe

RECIPES: dict[str, Recipe] = {
    "card": CardRecipe(),
    "willet": WilletRecipe(),
    "kunz_t12": KunzT12Recipe(),
    "kunz_t15": KunzT15Recipe(),
    "wairagkar": WairagkarRecipe(),
    "jude": JudeRecipe(),
}


def build_in_parallel(names, build, max_workers=None):
    """Build one dataset per recipe name, concurrently.

    Dataset preparation dominates start-up, threading it
    greatly reduces training time.

    Args:
        names: dataset names to build.
        build: callable mapping one name to its prepared dataset.
        max_workers: defaults to one thread per name.

    Returns:
        `{name: dataset}`; exceptions from `build` propagate.
    """
    names = list(names)
    with ThreadPoolExecutor(max_workers=max_workers or len(names)) as pool:
        futures = {name: pool.submit(build, name) for name in names}
        return {name: future.result() for name, future in futures.items()}


__all__ = ["RECIPES", "Recipe", "build_in_parallel"]

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

__all__ = ["RECIPES", "Recipe"]

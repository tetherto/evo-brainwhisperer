# Importing each module here fires its @register_dataset decorator.
# Any code that needs the registry populated only needs to import this package —
# it never has to know which dataset modules exist.
from . import card, kunz, willett  # noqa: F401

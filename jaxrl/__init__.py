from jaxtyping import install_import_hook


with install_import_hook("jaxrl", "beartype.beartype"):
    from . import value_learning

__all__ = ["value_learning"]

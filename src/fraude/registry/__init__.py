"""Cruce con el registro interno del CETPRO."""

from fraude.registry.build import build_registry_from_manifest
from fraude.registry.cross import (
    DEFAULT_THRESHOLD,
    RegistryEntry,
    cross_registry,
    load_registry,
    save_registry,
)

__all__ = [
    "DEFAULT_THRESHOLD",
    "RegistryEntry",
    "build_registry_from_manifest",
    "cross_registry",
    "load_registry",
    "save_registry",
]

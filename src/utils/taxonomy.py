"""Tiny helpers for taxonomy-aware reranking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaxonomyInfo:
    scientific_name: str = ""
    common_name: str = ""
    genus: str = ""


def genus_from_scientific_name(scientific_name: str) -> str:
    parts = scientific_name.strip().split()
    if not parts:
        return ""
    return parts[0].strip().lower()


def taxonomy_info(scientific_name: str = "", common_name: str = "") -> TaxonomyInfo:
    return TaxonomyInfo(
        scientific_name=scientific_name,
        common_name=common_name,
        genus=genus_from_scientific_name(scientific_name),
    )
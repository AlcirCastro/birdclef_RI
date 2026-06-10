"""Exp 3 — Segmentation strategy + late-fusion ranking with HNSW.

Switches segmentation to overlapping windows and uses a per-segment
document store with late-fusion RRF ranking. Tests how denser segment
coverage and rank fusion compare to the early-fusion / audio-doc baseline.
"""

from experiments._runner import main


if __name__ == "__main__":
    main(__file__)

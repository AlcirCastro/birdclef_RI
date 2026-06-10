"""Exp 2 — Noise robustness: same baseline but query-side noise sweep enabled.

Compare per-(noise_type × SNR) MAP/MRR/P@1 against the clean reference. Useful
for evaluating how Perch v2's segment embeddings degrade under realistic
field-recording conditions.
"""

from experiments._runner import main


if __name__ == "__main__":
    main(__file__)

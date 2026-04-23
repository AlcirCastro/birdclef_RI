#!/usr/bin/env python3
from panns_inference import AudioTagging
import inspect

# Get the __init__ signature
sig = inspect.signature(AudioTagging.__init__)
print("AudioTagging.__init__ signature:")
print(sig)

# Get the docstring
print("\nDocstring:")
print(AudioTagging.__doc__)

# List methods
print("\nMethods:")
for name in dir(AudioTagging):
    if not name.startswith('_'):
        print(f"  - {name}")

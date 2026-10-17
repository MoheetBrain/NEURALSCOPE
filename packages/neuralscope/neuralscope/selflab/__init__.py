"""SelfLab: audit the claims a model makes about its own behaviour against controlled experiments.

Public label: "Mirror" may be used on the website; the code and docs call it SelfLab.
Interface contract: docs/selflab/INTERFACES.md (schema version ``selflab-1``).
"""

from neuralscope.selflab.schemas import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION"]

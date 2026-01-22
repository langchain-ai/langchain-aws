from typing import Any, Dict, List

import numpy as np

# distance metrics
VALKEY_DISTANCE_METRICS: List[str] = ["COSINE", "IP", "L2"]

# supported vector datatypes
VALKEY_VECTOR_DTYPE_MAP: Dict[str, Any] = {
    "FLOAT32": np.float32,
    "FLOAT64": np.float64,
}

VALKEY_TAG_SEPARATOR = ","

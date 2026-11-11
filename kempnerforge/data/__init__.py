"""Data pipeline for KempnerForge.

Public API:
  - MemoryMappedDataset / HuggingFaceDataset: Dataset implementations
  - DistributedSampler: Rank-aware, deterministic sampling
  - StatefulDataLoader: Checkpoint-resumable data loading
  - StreamingDataLoader: Resumable loading of a streaming dataset
  - DataPositionError: A saved data position that cannot be resumed
"""

from kempnerforge.data.dataloader import (
    DataPositionError,
    StatefulDataLoader,
    StreamingDataLoader,
)
from kempnerforge.data.dataset import HuggingFaceDataset, MemoryMappedDataset
from kempnerforge.data.sampler import DistributedSampler

__all__ = [
    "DataPositionError",
    "DistributedSampler",
    "HuggingFaceDataset",
    "MemoryMappedDataset",
    "StatefulDataLoader",
    "StreamingDataLoader",
]

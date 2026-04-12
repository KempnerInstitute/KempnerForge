"""Distributed parallelism configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PipelineSchedule(StrEnum):
    schedule_1f1b = "1f1b"
    gpipe = "gpipe"
    interleaved_1f1b = "interleaved_1f1b"


@dataclass
class DistributedConfig:
    """Parallelism dimensions and distributed settings."""

    dp_shard: int = -1  # -1 -> auto (use all remaining GPUs)
    dp_replicate: int = 1
    tp: int = 1
    pp: int = 1
    pp_schedule: PipelineSchedule = PipelineSchedule.schedule_1f1b
    cp: int = 1
    ep: int = 1  # Expert parallelism degree (partitions MoE experts across ranks)
    nccl_timeout_sec: int = 1800
    backend: str = "cpu:gloo,cuda:nccl"

    def validate_world_size(self, world_size: int) -> None:
        """Validate that parallelism dimensions match world size."""
        dp_shard = self._resolve_dp_shard(world_size)
        expected = self.dp_replicate * dp_shard * self.tp * self.pp * self.cp * self.ep
        if expected != world_size:
            raise ValueError(
                f"Parallelism dimensions ({self.dp_replicate} \u00d7 {dp_shard} \u00d7 "
                f"{self.tp} \u00d7 {self.pp} \u00d7 {self.cp} \u00d7 {self.ep} = {expected}) "
                f"do not match world_size ({world_size})"
            )

    def _resolve_dp_shard(self, world_size: int) -> int:
        """Resolve dp_shard=-1 to actual value."""
        if self.dp_shard > 0:
            return self.dp_shard
        other = self.dp_replicate * self.tp * self.pp * self.cp * self.ep
        if world_size % other != 0:
            raise ValueError(
                f"world_size ({world_size}) not divisible by dp_replicate*tp*pp*cp*ep ({other})"
            )
        return world_size // other

    def resolve(self, world_size: int) -> DistributedConfig:
        """Return a copy with dp_shard resolved to a concrete value."""
        resolved = DistributedConfig(
            dp_shard=self._resolve_dp_shard(world_size),
            dp_replicate=self.dp_replicate,
            tp=self.tp,
            pp=self.pp,
            pp_schedule=self.pp_schedule,
            cp=self.cp,
            ep=self.ep,
            nccl_timeout_sec=self.nccl_timeout_sec,
            backend=self.backend,
        )
        resolved.validate_world_size(world_size)
        return resolved

    def __post_init__(self) -> None:
        if self.dp_shard == 0 or self.dp_shard < -1:
            raise ValueError("dp_shard must be -1 (auto) or positive")
        for name, val in [
            ("dp_replicate", self.dp_replicate),
            ("tp", self.tp),
            ("pp", self.pp),
            ("cp", self.cp),
            ("ep", self.ep),
        ]:
            if val < 1:
                raise ValueError(f"{name} must be >= 1")

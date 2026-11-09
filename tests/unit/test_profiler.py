"""Unit tests for the trace files that ``build_profiler`` writes."""

from __future__ import annotations

import re

import torch

from kempnerforge.config.schema import ProfilingConfig
from kempnerforge.profiling.profiler import build_profiler


class TestProfilerTraceFiles:
    def test_trace_file_is_named_after_the_rank(self, tmp_path):
        """One active step writes one trace, named after the rank zero padded to five digits."""
        config = ProfilingConfig(enable=True, start_step=1, end_step=2, trace_dir=str(tmp_path))
        prof = build_profiler(config, rank=3)
        assert prof is not None
        prof.start()
        try:
            for _ in range(2):
                _ = torch.ones(8, 8) @ torch.ones(8, 8)
                prof.step()
        finally:
            prof.stop()
        names = [path.name for path in tmp_path.iterdir()]
        assert len(names) == 1
        assert re.fullmatch(r"rank00003\.\d+\.pt\.trace\.json", names[0])

"""Headless wgpu measurement harness.

Renders a WGSL fragment shader (fullscreen triangle) to an offscreen texture
and measures GPU time per pass via timestamp queries. Wall-clock time around
submit+wait is recorded as a secondary signal. This is the substrate for the
gate #0 measurement-validity experiment; it makes no claims about validity —
gate #0 exists to establish that empirically.
"""

from __future__ import annotations

import hashlib
import platform
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import wgpu


TIMESTAMP_FEATURE = "timestamp-query"


def _request_adapter():
    gpu = wgpu.gpu
    if hasattr(gpu, "request_adapter_sync"):
        return gpu.request_adapter_sync(power_preference="high-performance")
    return gpu.request_adapter(power_preference="high-performance")


def _request_device(adapter, features):
    if hasattr(adapter, "request_device_sync"):
        return adapter.request_device_sync(required_features=features)
    return adapter.request_device(required_features=features)


def _adapter_info(adapter) -> dict:
    info = getattr(adapter, "info", None)
    if info is None and hasattr(adapter, "request_adapter_info"):
        info = adapter.request_adapter_info()
    if info is None:
        return {}
    try:
        return dict(info)
    except Exception:
        return {k: getattr(info, k) for k in dir(info) if not k.startswith("_")}


@dataclass
class Sample:
    gpu_ns: int | None  # timestamp-query delta; None if feature unavailable
    wall_ns: int  # wall clock around submit + wait


@dataclass
class RunResult:
    samples: list[Sample]
    warmup_samples: list[Sample]
    meta: dict = field(default_factory=dict)


class GpuContext:
    """One adapter + device, shareable by several ShaderTimers.

    Sharing matters: cross-process runs on Apple Silicon showed ~26% drift
    (2026-08-19 smoke run), so any paired comparison must interleave both
    shaders inside one process on one device.
    """

    def __init__(self):
        self.adapter = _request_adapter()
        self.adapter_features = set(str(f) for f in self.adapter.features)
        self.has_timestamps = any(
            "timestamp" in f and "query" in f for f in self.adapter_features
        )
        features = [TIMESTAMP_FEATURE] if self.has_timestamps else []
        self.device = _request_device(self.adapter, features)

    def meta(self) -> dict:
        return {
            "adapter": _adapter_info(self.adapter),
            "adapter_features": sorted(self.adapter_features),
            "timestamp_queries": self.has_timestamps,
            "wgpu_version": getattr(wgpu, "__version__", "unknown"),
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "utc_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


class ShaderTimer:
    """Owns pipeline and query machinery for one shader on a GpuContext."""

    def __init__(
        self,
        shader_path: Path,
        width: int = 1024,
        height: int = 1024,
        context: GpuContext | None = None,
    ):
        self.shader_path = Path(shader_path)
        self.width = width
        self.height = height

        self.ctx = context or GpuContext()
        self.adapter = self.ctx.adapter
        self.adapter_features = self.ctx.adapter_features
        self.has_timestamps = self.ctx.has_timestamps
        self.device = self.ctx.device

        self.shader_source = self.shader_path.read_text()
        self.shader_hash = hashlib.sha256(self.shader_source.encode()).hexdigest()

        self._build_pipeline()
        self._build_query_objects()

    # -- setup -------------------------------------------------------------

    def _build_pipeline(self):
        device = self.device
        self.module = device.create_shader_module(code=self.shader_source)

        self.uniform_buffer = device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        bind_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ]
        )
        self.bind_group = device.create_bind_group(
            layout=bind_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.uniform_buffer,
                        "offset": 0,
                        "size": 16,
                    },
                }
            ],
        )
        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[bind_layout]
        )
        self.texture = device.create_texture(
            size=(self.width, self.height, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        self.texture_view = self.texture.create_view()
        self.pipeline = device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={"module": self.module, "entry_point": "vs_main"},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            fragment={
                "module": self.module,
                "entry_point": "fs_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
        )

    def _build_query_objects(self):
        if not self.has_timestamps:
            self.query_set = None
            return
        device = self.device
        self.query_set = device.create_query_set(type="timestamp", count=2)
        self.query_resolve = device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC,
        )
        self.query_read = device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

    # -- measurement -------------------------------------------------------

    def set_uniforms(self, time_value: float):
        data = struct.pack(
            "ffff", float(self.width), float(self.height), time_value, 0.0
        )
        self.device.queue.write_buffer(self.uniform_buffer, 0, data)

    def _wait_idle(self):
        # Force completion of submitted work.
        if hasattr(self.device, "_poll_wait"):
            self.device._poll_wait()
        else:
            self.device._poll(block=True)

    def _read_timestamps(self) -> int:
        # map_sync blocks until the GPU copy into query_read has completed.
        self.query_read.map_sync(wgpu.MapMode.READ)
        try:
            t_begin, t_end = struct.unpack(
                "QQ", bytes(self.query_read.read_mapped(0, 16))[:16]
            )
        finally:
            self.query_read.unmap()
        return int(t_end - t_begin)

    def render_once(self, n_passes: int = 1) -> Sample:
        """One timed sample = one submit containing n_passes render passes.

        Batching (n_passes > 1) exists because per-pass timestamp queries on
        Metal/Apple GPUs proved unreliable (negative deltas observed on M5,
        wgpu 0.32) and single-pass wall clock is dominated by submit/readback
        overhead. Each pass is a separate render pass on purpose: Apple TBDR
        hidden-surface removal would eliminate overdraw if we issued N draws
        inside one pass, so per-pass load would not scale linearly.
        """
        device = self.device
        encoder = device.create_command_encoder()

        for i in range(n_passes):
            pass_kwargs = {
                "color_attachments": [
                    {
                        "view": self.texture_view,
                        "load_op": wgpu.LoadOp.clear,
                        "store_op": wgpu.StoreOp.store,
                        "clear_value": (0.0, 0.0, 0.0, 1.0),
                    }
                ]
            }
            if self.query_set is not None:
                writes = {"query_set": self.query_set}
                if i == 0:
                    writes["beginning_of_pass_write_index"] = 0
                if i == n_passes - 1:
                    writes["end_of_pass_write_index"] = 1
                if len(writes) > 1:
                    pass_kwargs["timestamp_writes"] = writes

            rpass = encoder.begin_render_pass(**pass_kwargs)
            rpass.set_pipeline(self.pipeline)
            rpass.set_bind_group(0, self.bind_group)
            rpass.draw(3, 1, 0, 0)
            rpass.end()

        if self.query_set is not None:
            encoder.resolve_query_set(self.query_set, 0, 2, self.query_resolve, 0)
            encoder.copy_buffer_to_buffer(
                self.query_resolve, 0, self.query_read, 0, 16
            )

        t0 = time.perf_counter_ns()
        device.queue.submit([encoder.finish()])
        self._wait_idle()
        wall_ns = time.perf_counter_ns() - t0

        gpu_ns = None
        if self.query_set is not None:
            gpu_ns = self._read_timestamps()

        return Sample(gpu_ns=gpu_ns, wall_ns=wall_ns)

    def measure(
        self,
        warmup: int,
        repeats: int,
        time_value: float = 1.0,
        n_passes: int = 1,
    ) -> RunResult:
        self.set_uniforms(time_value)
        warmup_samples = [self.render_once(n_passes) for _ in range(warmup)]
        samples = [self.render_once(n_passes) for _ in range(repeats)]
        return RunResult(
            samples=samples,
            warmup_samples=warmup_samples,
            meta=self.environment_meta()
            | {
                "warmup": warmup,
                "repeats": repeats,
                "time_uniform": time_value,
                "passes_per_sample": n_passes,
            },
        )

    # -- metadata ----------------------------------------------------------

    def read_pixels(self, time_value: float = 1.0):
        """Render once and read back the RGBA8 image as (h, w, 4) uint8."""
        import numpy as np

        self.set_uniforms(time_value)
        self.render_once(1)
        data = self.device.queue.read_texture(
            {"texture": self.texture, "mip_level": 0, "origin": (0, 0, 0)},
            {
                "offset": 0,
                "bytes_per_row": self.width * 4,
                "rows_per_image": self.height,
            },
            (self.width, self.height, 1),
        )
        return np.frombuffer(bytes(data), dtype=np.uint8).reshape(
            self.height, self.width, 4
        )

    def environment_meta(self) -> dict:
        return self.ctx.meta() | {
            "shader_path": str(self.shader_path),
            "shader_sha256": self.shader_hash,
            "resolution": [self.width, self.height],
        }

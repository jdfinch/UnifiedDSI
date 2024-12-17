
import ezpyzy as ez

import torch as pt
import contextlib as cl
import psutil as psu
import dataclasses as dc


@dc.dataclass
class PerformanceMetrics(ez.Config):
    run_time_h: float = None
    vram_used_gb: float = None
    ram_used_gb: float = None
    error_type: ... = None

    @cl.contextmanager
    def track(self):
        with ez.Timer() as timer:
            with measure_vram_usage_gb() as vram_gb:
                try:
                    yield
                except Exception as e:
                    self.error_type = str(type(e))
                    raise
                finally:
                    self.run_time_h = timer.elapsed.hours
                    self.ram_used_gb = measure_ram_usage_gb()
                    self.vram_used_gb, = vram_gb

    def max_update(self, submeasurements: list['PerformanceMetrics']):
        for submeasurement in submeasurements:
            vram = submeasurement.vram_used_gb
            if self.vram_used_gb is None or isinstance(vram, float) and vram > self.vram_used_gb:
                self.vram_used_gb = vram
            ram = submeasurement.ram_used_gb
            if self.ram_used_gb is None or isinstance(ram, float) and ram > self.ram_used_gb:
                self.ram_used_gb = ram


@cl.contextmanager
def measure_vram_usage_gb():
    """
    Use like:

    with measure_vram_usage_gb() as vram_gb:
    |   ... # processing here
    vram_used_gb = vram_gb[0]
    """
    pt.cuda.reset_peak_memory_stats()
    peak_memory_allocated = [None]
    try:
        yield peak_memory_allocated
        peak_memory_allocated[0] = pt.cuda.max_memory_allocated() / 1024 ** 3
    finally:
        pt.cuda.reset_peak_memory_stats()


def measure_ram_usage_gb():
    """Get current RAM usage in GB"""
    return psu.Process().memory_info().rss / 1024 ** 3

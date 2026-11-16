import unittest
from types import SimpleNamespace
from unittest import mock

from syntra_ai.hardware_telemetry import (
    device_memory_sample,
    process_peak_rss_bytes,
    reset_device_peak_stats,
)


class HardwareTelemetryTests(unittest.TestCase):
    def test_process_peak_rss_is_an_actual_integer_measurement_or_unavailable(self):
        value = process_peak_rss_bytes()
        self.assertTrue(value is None or (type(value) is int and value > 0))

    def test_cuda_peak_reset_and_sample_use_the_selected_allocator(self):
        selected = SimpleNamespace(type="cuda")
        cuda = SimpleNamespace(
            is_available=lambda: True,
            reset_peak_memory_stats=mock.Mock(),
            max_memory_allocated=lambda device: 987654,
        )
        torch_module = SimpleNamespace(device=lambda value: selected, cuda=cuda)

        self.assertTrue(reset_device_peak_stats("cuda:0", torch_module))
        cuda.reset_peak_memory_stats.assert_called_once_with(selected)
        self.assertEqual(device_memory_sample("cuda:0", torch_module), (987654, "cuda_peak_allocated"))

    def test_cuda_unavailable_or_failed_peak_reset_is_not_reported_as_a_peak(self):
        selected = SimpleNamespace(type="cuda")
        unavailable = SimpleNamespace(
            is_available=lambda: False,
            reset_peak_memory_stats=mock.Mock(),
        )
        torch_module = SimpleNamespace(device=lambda value: selected, cuda=unavailable)
        self.assertFalse(reset_device_peak_stats("cuda:0", torch_module))
        unavailable.reset_peak_memory_stats.assert_not_called()

        failing = SimpleNamespace(
            is_available=lambda: True,
            reset_peak_memory_stats=mock.Mock(side_effect=RuntimeError("allocator reset failed")),
        )
        self.assertFalse(reset_device_peak_stats("cuda:0", SimpleNamespace(device=lambda value: selected, cuda=failing)))

    def test_mps_uses_driver_current_allocation_when_available(self):
        selected = SimpleNamespace(type="mps")
        mps = SimpleNamespace(
            driver_allocated_memory=lambda: 345678,
            current_allocated_memory=lambda: 123456,
        )
        torch_module = SimpleNamespace(device=lambda value: selected, mps=mps)
        self.assertTrue(reset_device_peak_stats("mps", torch_module))
        self.assertEqual(device_memory_sample("mps", torch_module), (345678, "mps_driver_allocated_current"))

    def test_mps_falls_back_to_allocator_current_and_labels_it_exactly(self):
        selected = SimpleNamespace(type="mps")
        mps = SimpleNamespace(current_allocated_memory=lambda: 123456)
        torch_module = SimpleNamespace(device=lambda value: selected, mps=mps)
        self.assertEqual(device_memory_sample("mps", torch_module), (123456, "mps_allocator_current"))

    def test_mps_without_measurement_api_is_explicitly_unavailable(self):
        selected = SimpleNamespace(type="mps")
        torch_module = SimpleNamespace(device=lambda value: selected, mps=SimpleNamespace())
        self.assertEqual(device_memory_sample("mps", torch_module), (None, "mps_unavailable"))

    def test_cpu_device_memory_is_not_applicable_not_zero(self):
        selected = SimpleNamespace(type="cpu")
        torch_module = SimpleNamespace(device=lambda value: selected)
        self.assertTrue(reset_device_peak_stats("cpu", torch_module))
        self.assertEqual(device_memory_sample("cpu", torch_module), (None, "cpu_not_applicable"))

    def test_measurement_api_failures_remain_null_with_a_method(self):
        selected = SimpleNamespace(type="cuda")
        cuda = SimpleNamespace(
            is_available=lambda: True,
            max_memory_allocated=mock.Mock(side_effect=RuntimeError("allocator query failed")),
        )
        torch_module = SimpleNamespace(device=lambda value: selected, cuda=cuda)
        self.assertEqual(device_memory_sample("cuda", torch_module), (None, "device_memory_unavailable"))


if __name__ == "__main__":
    unittest.main()

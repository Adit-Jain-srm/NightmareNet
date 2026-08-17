"""Unit tests for nightmarenet.distributed.device_pool."""

from __future__ import annotations

import concurrent.futures
import unittest
from unittest import mock

from nightmarenet.distributed.device_pool import DevicePool


class TestDevicePoolUnit(unittest.TestCase):
    @mock.patch("torch.cuda.is_available", return_value=True)
    @mock.patch("torch.cuda.device_count", return_value=4)
    def test_pool_creation_with_mocked_cuda_devices(
        self, mock_count: mock.MagicMock, mock_avail: mock.MagicMock
    ) -> None:
        """Test DevicePool auto-discovery when CUDA is available."""
        pool = DevicePool()
        self.assertEqual(pool.available_devices, [0, 1, 2, 3])
        self.assertEqual(pool.get_num_devices(), 4)
        self.assertTrue(pool.should_use_ddp())

    def test_pool_creation_with_override_devices(self) -> None:
        """Test DevicePool initialization with explicit override device list."""
        pool = DevicePool(override_devices=[1, 3])
        self.assertEqual(pool.available_devices, [1, 3])
        self.assertEqual(pool.get_num_devices(), 2)
        self.assertTrue(pool.should_use_ddp())

    def test_pool_creation_empty_override(self) -> None:
        """Test DevicePool with empty override list."""
        pool = DevicePool(override_devices=[])
        self.assertEqual(pool.available_devices, [])
        self.assertEqual(pool.get_num_devices(), 0)
        self.assertFalse(pool.should_use_ddp())

    @mock.patch("torch.cuda.is_available", return_value=False)
    def test_no_gpu_cpu_fallback(self, mock_avail: mock.MagicMock) -> None:
        """Test DevicePool behavior in CPU-only environments."""
        pool = DevicePool()
        self.assertEqual(pool.available_devices, [])
        self.assertEqual(pool.get_num_devices(), 0)
        self.assertFalse(pool.should_use_ddp())

    @mock.patch("torch.cuda.is_available", return_value=True)
    @mock.patch("torch.cuda.device_count", return_value=1)
    def test_single_gpu_should_not_use_ddp(
        self, mock_count: mock.MagicMock, mock_avail: mock.MagicMock
    ) -> None:
        """Test single GPU setup does not trigger DDP recommendation."""
        pool = DevicePool()
        self.assertEqual(pool.get_num_devices(), 1)
        self.assertFalse(pool.should_use_ddp())

    def test_memory_requirements_estimation(self) -> None:
        """Test VRAM memory requirements estimation accuracy and scaling."""
        pool = DevicePool(override_devices=[0])
        num_params = 100_000_000
        # 100M * 4 bytes * 3 (model+grads+optimizer) * 1.2 (buffer) / (1024^3)
        expected_gb = (100_000_000 * 4 * 3 * 1.2) / (1024**3)
        calculated_gb = pool.estimate_memory_requirements(num_params)
        self.assertAlmostEqual(calculated_gb, expected_gb, places=5)
        self.assertGreater(calculated_gb, 1.0)
        self.assertEqual(pool.estimate_memory_requirements(0), 0.0)

    def test_concurrent_device_pool_access_safety(self) -> None:
        """Test concurrent multi-threaded inspection of DevicePool properties."""
        pool = DevicePool(override_devices=[0, 1, 2, 3])

        def worker(thread_id: int) -> tuple[int, bool, float]:
            return (
                pool.get_num_devices(),
                pool.should_use_ddp(),
                pool.estimate_memory_requirements(10_000 * thread_id),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, i) for i in range(1, 20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        self.assertEqual(len(results), 19)
        for num_devs, ddp, mem in results:
            self.assertEqual(num_devs, 4)
            self.assertTrue(ddp)
            self.assertGreater(mem, 0.0)


if __name__ == "__main__":
    unittest.main()

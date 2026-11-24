"""验证 NCU 按 worker 触发，不依赖卡数，也不改变原 base-GPU 模式。"""

import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.io_struct import ProfileReq
from sglang.srt.managers.scheduler_components import profiler_manager as legacy
from sglang.srt.utils import profile_utils as v2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNcuRangeProfiler(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.output = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.device = SimpleNamespace(gpu_id=0, base_gpu_id=0)
        self.cuda = SimpleNamespace(
            cudaProfilerStart=Mock(return_value=0),
            cudaProfilerStop=Mock(return_value=0),
        )
        self.stack.enter_context(
            patch.object(legacy.envs.SGLANG_PROFILE_V2, "get", return_value=False)
        )
        self.stack.enter_context(
            patch.object(legacy, "get_device", return_value=self.device)
        )
        self.stack.enter_context(
            patch.object(v2, "get_device", return_value=self.device)
        )
        self.is_cuda = self.stack.enter_context(
            patch.object(v2.current_platform, "is_cuda", return_value=True)
        )
        self.stack.enter_context(
            patch.object(v2.current_platform, "is_out_of_tree", return_value=False)
        )
        self.stack.enter_context(
            patch.object(v2.torch.cuda, "cudart", return_value=self.cuda)
        )
        self.stack.enter_context(
            patch.object(legacy, "set_detailed_annotations_enabled")
        )
        self.stack.enter_context(patch.object(v2, "set_detailed_annotations_enabled"))

    def manager(self):
        return legacy.SchedulerProfilerManager(
            dp_tp_cpu_group=None, get_forward_ct=lambda: 0
        )

    def configure(self, manager, activities, **kwargs):
        return manager._init_profile(
            self.output,
            None,
            None,
            activities,
            None,
            None,
            kwargs.get("profile_by_stage", False),
            "ncu-test",
        )

    def test_legacy_all_workers_for_different_gpu_counts_and_offsets(self):
        for count in (1, 4, 8, 16):
            for base in (0, 3):
                with self.subTest(count=count, base=base):
                    self.cuda.cudaProfilerStart.reset_mock()
                    self.cuda.cudaProfilerStop.reset_mock()
                    self.device.base_gpu_id = base
                    for gpu_id in range(base, base + count):
                        self.device.gpu_id = gpu_id
                        manager = self.manager()
                        self.assertTrue(self.configure(manager, ["NCU_RANGE"]).success)
                        self.assertTrue(manager._start_profile().success)
                        self.assertTrue(manager._stop_profile().success)
                    self.assertEqual(self.cuda.cudaProfilerStart.call_count, count)
                    self.assertEqual(self.cuda.cudaProfilerStop.call_count, count)

    def test_reindexed_worker_does_not_depend_on_base_gpu_id(self):
        self.device.gpu_id = 0
        self.device.base_gpu_id = 4
        manager = self.manager()
        self.assertTrue(self.configure(manager, ["NCU_RANGE"]).success)
        self.assertTrue(manager._start_profile().success)
        self.assertTrue(manager._stop_profile().success)
        self.cuda.cudaProfilerStart.assert_called_once()
        self.cuda.cudaProfilerStop.assert_called_once()

    def test_existing_activity_still_only_triggers_base_gpu(self):
        self.device.base_gpu_id = 4
        for gpu_id in range(4, 8):
            self.device.gpu_id = gpu_id
            manager = self.manager()
            self.assertTrue(self.configure(manager, ["CUDA_PROFILER"]).success)
            self.assertTrue(manager._start_profile().success)
            self.assertTrue(manager._stop_profile().success)
        self.cuda.cudaProfilerStart.assert_called_once()
        self.cuda.cudaProfilerStop.assert_called_once()

    def test_v2_stage_backend_covers_workers_and_preserves_old_activity(self):
        for activity, expected in (("NCU_RANGE", 8), ("CUDA_PROFILER", 1)):
            with self.subTest(activity=activity):
                self.cuda.cudaProfilerStart.reset_mock()
                self.cuda.cudaProfilerStop.reset_mock()
                for gpu_id in range(8):
                    self.device.gpu_id = gpu_id
                    manager = v2.ProfileManager(cpu_group=None)
                    result = manager.configure(
                        output_dir=self.output,
                        start_step=None,
                        num_steps=1,
                        activities=[activity],
                        with_stack=None,
                        record_shapes=None,
                        profile_by_stage=True,
                        profile_id="ncu-test",
                        merge_profiles=False,
                        profile_prefix="",
                    )
                    self.assertTrue(result.success)
                    manager._do_start("decode")
                    manager._do_stop()
                self.assertEqual(self.cuda.cudaProfilerStart.call_count, expected)
                self.assertEqual(self.cuda.cudaProfilerStop.call_count, expected)

    def test_mixed_activities_fail_without_starting_or_changing_stage_state(self):
        for other in ("CUDA_PROFILER", "CPU", "GPU", "MEM"):
            for by_stage in (False, True):
                with self.subTest(other=other, by_stage=by_stage):
                    manager = self.manager()
                    request = ProfileReq(
                        output_dir=self.output,
                        activities=["NCU_RANGE", other],
                        profile_by_stage=by_stage,
                        profile_id="ncu-test",
                    )
                    result = manager._profile(request)
                    self.assertFalse(result.success)
                    self.assertFalse(manager.profile_in_progress)
                    self.assertFalse(manager.profile_by_stage)
        self.cuda.cudaProfilerStart.assert_not_called()

    def test_non_cuda_and_missing_gpu_are_rejected(self):
        manager = self.manager()
        self.is_cuda.return_value = False
        self.assertFalse(self.configure(manager, ["NCU_RANGE"]).success)
        self.is_cuda.return_value = True
        self.device.gpu_id = None
        self.assertFalse(self.configure(manager, ["NCU_RANGE"]).success)
        self.cuda.cudaProfilerStart.assert_not_called()

    def test_repeated_start_is_rejected_without_second_native_call(self):
        manager = self.manager()
        request = ProfileReq(
            output_dir=self.output, activities=["NCU_RANGE"], profile_id="ncu-test"
        )
        self.assertTrue(manager._profile(request).success)
        self.assertFalse(manager._profile(request).success)
        self.cuda.cudaProfilerStart.assert_called_once()
        self.assertTrue(manager._stop_profile().success)

    def test_native_failures_do_not_report_success(self):
        manager = self.manager()
        self.configure(manager, ["NCU_RANGE"])
        self.cuda.cudaProfilerStart.return_value = 1
        self.assertFalse(manager._start_profile().success)
        self.assertFalse(manager.profile_in_progress)
        self.cuda.cudaProfilerStart.return_value = 0
        self.assertTrue(manager._start_profile().success)
        self.cuda.cudaProfilerStop.return_value = 1
        self.assertFalse(manager._stop_profile().success)
        self.assertTrue(manager.profile_in_progress)
        self.cuda.cudaProfilerStop.return_value = 0
        self.assertTrue(manager._stop_profile().success)


if __name__ == "__main__":
    unittest.main()

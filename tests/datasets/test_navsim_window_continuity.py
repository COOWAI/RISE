"""Regression tests for NavSim/Bench2Drive sliding-window construction.

Covers three invariants introduced/hardened this cycle:

1. Anchored windowing — windows reset the stride clock at each image-valid
   segment so sparse-image logs still produce windows regardless of raw offset.
2. Time-gap rejection — a window whose GT span crosses a recording gap
   (missing timesteps, detected via per-frame timestamps) is dropped. This is
   the only place a real temporal jump can be caught, since the raw frame
   indices inside a window are contiguous by construction.
3. Bench2Drive frame_idx continuity — the separate B2D loader catches dropped
   frames via real ``frame_idx`` deltas (it never had the synthetic-index bug).

The dataset classes are exercised via ``__new__`` + manual attribute injection
so the window-building logic is tested in isolation without any disk I/O.
"""

import unittest
from unittest.mock import patch

import numpy as np

from app.vjepa_cowa_world_model.training.b2d_data import B2DSceneRecord, Bench2DriveWorldModelDataset
from app.vjepa_cowa_world_model.training.navsim_data import NavSimWorldModelDataset, SceneRecord, WindowRecord


def _make_navsim_ds(
    *,
    frames_per_clip=12,
    window_stride=4,
    required_image_frames=4,
    max_time_gap_us=750000.0,
    tail_seconds=None,
):
    """Build a NavSim dataset shell wired only for window-index construction."""
    ds = NavSimWorldModelDataset.__new__(NavSimWorldModelDataset)
    ds.frames_per_clip = frames_per_clip
    ds.sample_step = 1
    ds.min_valid_frames = 1 + (frames_per_clip - 1) * ds.sample_step
    ds.window_stride = window_stride
    ds.required_image_frames = required_image_frames
    ds.num_observed_frames = required_image_frames
    ds.image_require_policy = "observed_only"
    ds.max_time_gap_us = max_time_gap_us
    ds.base_fps = 2.0
    ds.tail_seconds = tail_seconds
    ds.window_start_policy = "sliding"
    return ds


def _dense_timestamps(n, step_us=500000):
    return [i * step_us for i in range(n)]


def _navsim_scene(valid_frame_indices, frame_count, frame_timestamps=None):
    return SceneRecord(
        scene_name="s",
        pkl_path="s.pkl",
        camera_dir="d",
        valid_frame_indices=valid_frame_indices,
        frame_count=frame_count,
        frame_timestamps=frame_timestamps,
    )


class NavSimAnchoredWindowTests(unittest.TestCase):
    def test_segments_split_on_image_gaps(self):
        ds = _make_navsim_ds(required_image_frames=4)
        scene = _navsim_scene([0, 1, 2, 3, 10, 11, 12, 13, 14, 50, 51], frame_count=60)
        # runs of length >= 4: [0..3] and [10..14]; [50,51] dropped (len 2 < 4)
        self.assertEqual(ds._image_segment_starts(scene), [(0, 3), (10, 14)])

    def test_window_anchored_to_segment_start_off_grid(self):
        # Images live at raw frames 55..70 — not aligned to a stride-4 raw grid.
        # Anchored windowing must still produce a window at start=55.
        ds = _make_navsim_ds(frames_per_clip=12, window_stride=4, required_image_frames=4)
        ds.scenes = [_navsim_scene(list(range(55, 71)), frame_count=90, frame_timestamps=_dense_timestamps(90))]
        starts = [w.start_pos for w in ds._build_window_index()]
        self.assertIn(55, starts)  # off-grid segment start recovered
        # stride-4 within segment [55..70]: start <= min(70-4+1, 90-12) = 67
        self.assertEqual(starts, [55, 59, 63, 67])

    def test_observed_frames_do_not_overlap_at_stride_equals_required(self):
        ds = _make_navsim_ds(window_stride=4, required_image_frames=4)
        ds.scenes = [_navsim_scene(list(range(0, 40)), frame_count=60, frame_timestamps=_dense_timestamps(60))]
        windows = sorted(w.start_pos for w in ds._build_window_index())
        # adjacent observed windows are >= required apart; their observed
        # prefixes [s..s+3] and [s+4..s+7] are disjoint -> zero input overlap.
        for a, b in zip(windows, windows[1:]):
            self.assertGreaterEqual(b - a, ds.required_image_frames)

    def test_tail_seconds_keeps_only_late_counterfactual_windows(self):
        ds = _make_navsim_ds(frames_per_clip=10, window_stride=1, required_image_frames=4, tail_seconds=5.0)
        ds.scenes = [_navsim_scene(list(range(0, 20)), frame_count=20, frame_timestamps=_dense_timestamps(20))]

        starts = [w.start_pos for w in ds._build_window_index()]

        self.assertEqual(starts, [10])

    def test_twelve_frame_scene_with_null_tail_has_exactly_one_window(self):
        ds = _make_navsim_ds(frames_per_clip=12, window_stride=1, required_image_frames=12, tail_seconds=None)
        ds.scenes = [_navsim_scene(list(range(12)), frame_count=12, frame_timestamps=_dense_timestamps(12))]

        starts = [window.start_pos for window in ds._build_window_index()]

        self.assertEqual(starts, [0])

    def test_tail_seconds_preserves_stride_grid_when_filtering_early_windows(self):
        ds = _make_navsim_ds(frames_per_clip=6, window_stride=4, required_image_frames=4, tail_seconds=5.0)
        ds.scenes = [_navsim_scene(list(range(0, 24)), frame_count=24, frame_timestamps=_dense_timestamps(24))]

        starts = [w.start_pos for w in ds._build_window_index()]

        self.assertEqual(starts, [16])

    def test_counterfactual_scene_start_keeps_only_the_real_observed_context(self):
        ds = _make_navsim_ds(frames_per_clip=10, window_stride=4, required_image_frames=4)
        ds.window_start_policy = "counterfactual_scene_start"
        ds.scenes = [_navsim_scene(list(range(17)), frame_count=17, frame_timestamps=_dense_timestamps(17))]

        starts = [w.start_pos for w in ds._build_window_index()]

        self.assertEqual(starts, [0])

    def test_counterfactual_scene_start_rejects_generated_observed_frames(self):
        ds = _make_navsim_ds(frames_per_clip=10, required_image_frames=4)
        ds.window_start_policy = "counterfactual_scene_start"
        valid_contract = np.asarray([True] * 4 + [False] * 6, dtype=np.bool_)

        ds._validate_counterfactual_scene_start_metadata(window_start=0, metadata_valid_mask=valid_contract)
        with self.assertRaisesRegex(ValueError, "fully valid observed prefix"):
            ds._validate_counterfactual_scene_start_metadata(
                window_start=4,
                metadata_valid_mask=np.asarray([False] * 10, dtype=np.bool_),
            )


class NavSimTimeGapTests(unittest.TestCase):
    def _scene_with_gap(self, gap_at, gap_us):
        ts = _dense_timestamps(40)
        for i in range(gap_at + 1, len(ts)):  # inflate everything after the gap
            ts[i] += gap_us
        return _navsim_scene(list(range(0, 40)), frame_count=40, frame_timestamps=ts)

    def test_window_spanning_gap_is_rejected(self):
        ds = _make_navsim_ds(frames_per_clip=12, window_stride=1, required_image_frames=4)
        # a 1.0s extra delta (>0.75s threshold) right after raw frame 10
        ds.scenes = [self._scene_with_gap(gap_at=10, gap_us=1_000_000)]
        windows = {w.start_pos for w in ds._build_window_index()}
        # spans [start..start+11]; those including the 10->11 boundary (start 0..10) drop.
        self.assertTrue(all(s >= 11 for s in windows))
        self.assertNotIn(10, windows)
        self.assertIn(11, windows)

    def test_750945us_boundary_excludes_only_crossing_windows(self):
        ds = _make_navsim_ds(frames_per_clip=4, window_stride=1, required_image_frames=4)
        timestamps = _dense_timestamps(24)
        for frame_index in range(15, len(timestamps)):
            timestamps[frame_index] += 250_945
        ds.scenes = [_navsim_scene(list(range(24)), frame_count=24, frame_timestamps=timestamps)]

        windows = {window.start_pos for window in ds._build_window_index()}

        self.assertFalse(ds._is_window_time_continuous(timestamps, 12))
        self.assertIn(11, windows)
        self.assertTrue({12, 13, 14}.isdisjoint(windows))
        self.assertIn(15, windows)

    def test_no_gap_keeps_all_windows(self):
        ds = _make_navsim_ds(frames_per_clip=12, window_stride=1, required_image_frames=4)
        ds.scenes = [_navsim_scene(list(range(0, 40)), frame_count=40, frame_timestamps=_dense_timestamps(40))]
        self.assertEqual(len(ds._build_window_index()), 40 - 12 + 1)  # every start 0..28

    def test_fail_open_when_timestamps_missing(self):
        # No timestamps (e.g. un-migrated nuScenes pkls) -> check is skipped.
        ds = _make_navsim_ds(frames_per_clip=12, window_stride=1, required_image_frames=4)
        ds.scenes = [_navsim_scene(list(range(0, 40)), frame_count=40, frame_timestamps=None)]
        self.assertEqual(len(ds._build_window_index()), 40 - 12 + 1)

    def test_explicit_formal_timestamp_policy_rejects_missing_frame_timestamp(self):
        ds = _make_navsim_ds()
        ds.timestamp_policy_was_explicit = True

        with self.assertRaisesRegex(ValueError, "explicit timestamp_policy"):
            ds._extract_frame_timestamps([{"timestamp": 0}, {}])

    def test_is_window_time_continuous_unit(self):
        ds = _make_navsim_ds(frames_per_clip=4, max_time_gap_us=750000.0)
        self.assertTrue(ds._is_window_time_continuous([0, 500000, 1000000, 1500000], 0))
        self.assertFalse(ds._is_window_time_continuous([0, 500000, 1250945, 1750945], 0))
        self.assertFalse(ds._is_window_time_continuous([0, 500000, 1500000, 2000000], 0))  # 1.0s gap
        self.assertTrue(ds._is_window_time_continuous(None, 0))  # fail-open

    def test_constructor_computes_exact_2hz_single_gap_boundary(self):
        scene = _navsim_scene([0, 1], frame_count=2, frame_timestamps=[0, 750000])
        with (
            patch.object(NavSimWorldModelDataset, "_build_scene_index", return_value=[scene]),
            patch.object(
                NavSimWorldModelDataset,
                "_build_window_index",
                return_value=[WindowRecord(scene_idx=0, start_pos=0)],
            ),
        ):
            ds = NavSimWorldModelDataset(
                data_path="unused-data",
                sensor_blobs_path="unused-sensors",
                frames_per_clip=2,
                fps=2,
                tubelet_size=1,
                index_cache=False,
                max_frame_gap=1,
                load_agent_annotations=False,
            )

        self.assertEqual(ds.base_fps, 2.0)
        self.assertEqual(ds.max_time_gap_us, 750000.0)
        self.assertTrue(ds._is_window_time_continuous([0, 750000], 0))
        self.assertFalse(ds._is_window_time_continuous([0, 750001], 0))
        self.assertFalse(ds._is_window_time_continuous([0, 750945], 0))


class Bench2DriveContinuityTests(unittest.TestCase):
    def _make_b2d_ds(self, frames_per_clip=12, base_fps=10, fps=2, max_frame_gap=1):
        ds = Bench2DriveWorldModelDataset.__new__(Bench2DriveWorldModelDataset)
        ds.frames_per_clip = frames_per_clip
        ds.sample_step = max(1, round(base_fps / fps))  # 5
        ds.max_frame_gap = int(max(max_frame_gap, ds.sample_step))  # 5
        return ds

    def test_dense_frame_idx_passes(self):
        ds = self._make_b2d_ds()
        # 60 contiguous frames -> sampled every 5th, consecutive deltas == 5 == max_frame_gap
        scene = B2DSceneRecord("f", list(range(60)), list(range(60)), list(range(60)))
        self.assertTrue(ds._is_window_continuous(scene, start_pos=0))

    def test_dropped_frame_idx_is_rejected(self):
        ds = self._make_b2d_ds()
        # a 10-frame hole at array index 30 makes a sampled pair span > max_frame_gap(5)
        frame_indices = list(range(0, 30)) + list(range(40, 70))
        scene = B2DSceneRecord("f", list(range(60)), frame_indices, frame_indices)
        self.assertFalse(ds._is_window_continuous(scene, start_pos=5))


if __name__ == "__main__":
    unittest.main()

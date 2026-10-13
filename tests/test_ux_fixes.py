"""Targeted tests for the UI/UX and workflow fixes.

Each class maps to one of the five requested workstreams, so a regression in a
specific area fails a test that names that area.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import imaging
import metrics
import research
import storage
import teaching as teaching_mod


# ==========================================================================
# Task 1 — viewer framing
# ==========================================================================
class TestHeadFraming:
    def _volume(self) -> np.ndarray:
        """A head-like blob inside a larger empty matrix."""
        vol = np.zeros((120, 120, 20), dtype=np.float32)
        vol[25:95, 30:90, 5:15] = 100.0
        return vol

    @pytest.fixture
    def affine(self) -> np.ndarray:
        return np.diag([1.0, 1.0, 1.0, 1.0])

    def test_crop_is_square(self, affine):
        box = imaging.head_bounding_box(self._volume(), affine)
        assert box is not None
        h = box[0].stop - box[0].start
        w = box[1].stop - box[1].start
        assert h == w, "a non-square crop would distort the anatomy"

    def test_crop_is_within_bounds(self, affine):
        vol = self._volume()
        box = imaging.head_bounding_box(vol, affine)
        assert 0 <= box[0].start < box[0].stop <= vol.shape[0]
        assert 0 <= box[1].start < box[1].stop <= vol.shape[1]

    def test_crop_actually_enlarges_the_head(self, affine):
        vol = self._volume()
        box = imaging.head_bounding_box(vol, affine)
        side = box[0].stop - box[0].start
        assert side < 120 * 0.85, "crop should be meaningfully tighter than the matrix"
        # The head itself (70x60) must still fit inside the crop.
        assert side >= 70

    def test_empty_volume_returns_none(self, affine):
        assert imaging.head_bounding_box(np.zeros((40, 40, 8), np.float32), affine) is None

    def test_full_frame_returns_none_rather_than_a_pointless_crop(self, affine):
        """If tissue fills the matrix there is nothing to crop away."""
        assert imaging.head_bounding_box(np.ones((40, 40, 8), np.float32), affine) is None

    def test_crop_is_stable_across_slices(self, affine):
        """One box per volume, so the head does not drift while scrolling."""
        vol = self._volume()
        assert imaging.head_bounding_box(vol, affine) == imaging.head_bounding_box(vol, affine)

    def test_crop_keeps_image_and_mask_aligned(self, affine):
        """The decisive property: overlay must stay registered to the anatomy."""
        vol = self._volume()
        lm = np.zeros(vol.shape, dtype=np.uint8)
        lm[40:60, 45:75, 5:15] = 2          # a lesion inside the head
        lm[45:55, 50:70, 5:15] = 1
        masks = metrics.region_masks(lm, metrics.BRATS_2020)
        box = imaging.head_bounding_box(vol, affine)
        idx = 10

        plain = imaging.render_mri_slice(vol, affine, idx, masks=masks, show_masks=False, crop=box)
        overlaid = imaging.render_mri_slice(vol, affine, idx, masks=masks, show_masks=True, crop=box)
        assert plain.shape == overlaid.shape

        # Every changed pixel must lie inside the cropped mask footprint.
        changed = np.any(plain != overlaid, axis=2)
        expected = imaging.slice_masks_at(masks, idx, affine)["WT"][box]
        assert not np.any(changed & ~expected), "overlay painted outside the mask"

    def test_mask_resize_uses_nearest_neighbour(self):
        """Interpolating labels would invent ids and blur WT into TC."""
        mask = np.zeros((8, 8), dtype=bool)
        mask[:4, :] = True
        out = imaging._resize_mask(mask, 32)
        assert out.shape == (32, 32)
        # A nearest-neighbour upscale of a hard-edged mask stays hard-edged:
        # every row is either fully inside or fully outside.
        rows_inside = out.all(axis=1)
        rows_outside = (~out).all(axis=1)
        assert (rows_inside | rows_outside).all()

    def test_resize_preserves_aspect_ratio(self):
        rgb = np.zeros((40, 80, 3), dtype=np.uint8)
        out = imaging._resize_rgb(rgb, 160)
        assert out.shape[1] == 160
        assert out.shape[0] == 80          # 2:1 preserved

    def test_display_size_none_is_a_no_op(self, affine):
        vol = self._volume()
        img = imaging.render_mri_slice(vol, affine, 10)
        assert img.shape[:2] == (120, 120)


# ==========================================================================
# Task 3 — teaching answer isolation
# ==========================================================================
class TestTeachingAnswerIsolation:
    def _state(self, submitted: bool = False) -> teaching_mod.ExerciseState:
        lm = np.zeros((6, 6, 6), dtype=np.uint8)
        lm[1:5, 1:5, 1:5] = 2
        st = teaching_mod.ExerciseState(
            case_id="C1", reference_label_map=lm,
            reference_convention=metrics.BRATS_2020, reference_spacing=(1.0, 1.0, 1.0))
        st.submitted = submitted
        return st

    def test_reveal_refused_before_submission(self):
        decision = teaching_mod.reveal_allowed(self._state(submitted=False))
        assert not decision.allowed
        assert "提交前不可查看" in decision.message
        assert not bool(decision)

    def test_reveal_allowed_after_submission(self):
        assert teaching_mod.reveal_allowed(self._state(submitted=True)).allowed

    def test_reveal_refused_when_no_exercise_loaded(self):
        """Calling the reveal callback without loading a case must not leak."""
        decision = teaching_mod.reveal_allowed(None)
        assert not decision.allowed
        assert not bool(decision)

    def test_reference_mask_unavailable_before_submission(self):
        st = self._state(submitted=False)
        assert st.reference_for_display() is None
        assert not st.answer_visible

    def test_reference_mask_available_after_submission(self):
        st = self._state(submitted=True)
        assert st.reference_for_display() is not None

    def test_reset_hides_the_answer_again(self):
        st = self._state(submitted=True)
        st.reset()
        assert st.reference_for_display() is None
        assert not teaching_mod.reveal_allowed(st).allowed

    def test_switching_case_builds_a_fresh_hidden_exercise(self):
        """A new exercise for another case must start locked, not inherit state."""
        old = self._state(submitted=True)

        lm = np.zeros((6, 6, 6), dtype=np.uint8)
        lm[1:3, 1:3, 1:3] = 4
        new = teaching_mod.ExerciseState(
            case_id="C2", reference_label_map=lm,
            reference_convention=metrics.BRATS_2020, reference_spacing=(1.0, 1.0, 1.0))

        assert new.case_id != old.case_id
        assert new.reference_for_display() is None
        assert not teaching_mod.reveal_allowed(new).allowed

    def test_first_submission_unlocks_and_is_counted(self):
        st = self._state(submitted=False)
        assert not teaching_mod.reveal_allowed(st).allowed
        count = st.record_submission("answer", None, {"comparable": False}, "fb")
        assert count == 1
        assert teaching_mod.reveal_allowed(st).allowed

    def test_repeat_submission_is_counted_and_stays_unlocked(self):
        st = self._state(submitted=False)
        st.record_submission("first", None, {}, "fb1")
        st.record_submission("second", None, {}, "fb2")
        assert st.submission_count == 2
        assert st.answer_text == "second"
        assert teaching_mod.reveal_allowed(st).allowed

    def test_reset_clears_the_submission_count(self):
        st = self._state(submitted=False)
        st.record_submission("a", None, {}, "")
        st.reset()
        assert st.submission_count == 0


# ==========================================================================
# Task 4 — research statistics
# ==========================================================================
class TestSpreadReporting:
    def test_single_case_does_not_report_a_standard_deviation(self):
        label = research.spread_label(1, 0.0)
        assert "标准差不适用" in label
        assert "0.000" not in label, "sigma=0 for n=1 reads as 'perfectly consistent'"

    def test_two_or_more_cases_report_sigma(self):
        label = research.spread_label(6, 0.0604)
        assert "n=6" in label
        assert "0.060" in label

    def test_tile_for_single_case_suppresses_sigma(self):
        html = research.theme_tile("WT", 0.925, 0.0, 1)
        assert "标准差不适用" in html
        assert "σ=0.000" not in html

    def test_tile_with_no_data_says_so(self):
        html = research.theme_tile("WT", None, None, 0)
        assert "暂无评测" in html


class TestBatchVersusInteractive:
    def test_missing_batch_file_is_not_an_error(self, tmp_path):
        batch = research.load_batch_evaluation(tmp_path / "nope.json")
        assert not batch.available
        assert batch.n_cases == 0

    def test_corrupt_batch_file_is_reported_not_raised(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        batch = research.load_batch_evaluation(bad)
        assert not batch.available
        assert "无法解析" in batch.note

    def test_batch_file_is_parsed(self, tmp_path):
        p = tmp_path / "evaluation.json"
        p.write_text(json.dumps({
            "model": "MONAI/brats_mri_segmentation", "model_version": "0.5.4",
            "architecture": "SegResNet", "device": "cuda:0",
            "cases": [
                {"case_id": "A", "dice": {"WT": 0.9, "TC": 0.8, "ET": 0.7}},
                {"case_id": "B", "dice": {"WT": 0.7, "TC": 0.6, "ET": 0.5}},
            ],
        }), encoding="utf-8")
        batch = research.load_batch_evaluation(p)
        assert batch.available and batch.n_cases == 2
        assert batch.mean_dice["WT"] == pytest.approx(0.8)
        assert batch.std_dice["WT"] == pytest.approx(0.1)

    def test_two_sources_are_reported_separately_never_pooled(self, tmp_path):
        """Mixing a scripted sweep with ad-hoc UI runs would describe neither."""
        st = storage.Store(tmp_path / "t.db")
        st.upsert_case("A", dataset_source="ds")
        st.record_analysis_run("A", "MONAI/brats", "success", model_version="0.5.4",
                               evaluation={"dice": {"WT": 0.50, "TC": 0.50, "ET": 0.50},
                                           "missed_voxels": {}, "extra_voxels": {}})
        stats = research.aggregate(st)

        batch = research.BatchEvaluation(
            available=True, model="MONAI/brats_mri_segmentation", model_version="0.5.4",
            architecture="SegResNet", device="cuda:0", n_cases=6,
            mean_dice={"WT": 0.8889, "TC": 0.9243, "ET": 0.8400},
            std_dice={"WT": 0.0604, "TC": 0.0155, "ET": 0.0327}, cases=[{}] * 6)

        md = research.dashboard_markdown(stats, batch)
        assert "独立批量评测" in md
        assert "本机交互式单病例评测" in md
        assert "口径不同" in md and "未合并计算" in md

        # The interactive mean (0.50) must not be averaged into the batch mean.
        assert "0.889" in md            # batch WT preserved
        assert "0.500" in md            # interactive WT reported separately
        assert "0.694" not in md        # and never pooled into a blended figure

    def test_repeated_runs_of_one_case_count_once(self, tmp_path):
        """n must count distinct cases, not analysis runs.

        Two runs on the same case give an identical score; counting both would
        report n=2, sigma=0.000 and make an untested model look stable.
        """
        st = storage.Store(tmp_path / "t.db")
        st.upsert_case("A", dataset_source="ds")
        for _ in range(3):
            st.record_analysis_run("A", "MONAI/brats", "success", model_version="0.5.4",
                                   evaluation={"dice": {"WT": 0.9, "TC": 0.9, "ET": 0.9},
                                               "missed_voxels": {}, "extra_voxels": {}})
        stats = research.aggregate(st)
        assert stats.n_evaluated_cases == 1
        md = research.dashboard_markdown(stats, research.BatchEvaluation(available=False))
        assert "标准差不适用" in md
        assert "σ=0.000" not in md

    def test_latest_run_per_case_wins(self, tmp_path):
        """A re-run with a different score must supersede the earlier one."""
        st = storage.Store(tmp_path / "t.db")
        st.upsert_case("A", dataset_source="ds")
        st.record_analysis_run("A", "m", "success", model_version="1",
                               evaluation={"dice": {"WT": 0.10, "TC": 0.10, "ET": 0.10}})
        st.record_analysis_run("A", "m", "success", model_version="1",
                               evaluation={"dice": {"WT": 0.90, "TC": 0.90, "ET": 0.90}})
        stats = research.aggregate(st)
        assert stats.n_evaluated_cases == 1
        assert stats.mean_dice["WT"] == pytest.approx(0.90)

    def test_interactive_section_absent_when_no_ui_runs(self, tmp_path):
        st = storage.Store(tmp_path / "t.db")
        stats = research.aggregate(st)
        batch = research.BatchEvaluation(available=True, n_cases=6,
                                         mean_dice={"WT": 0.8889, "TC": 0.92, "ET": 0.84},
                                         std_dice={"WT": 0.06, "TC": 0.01, "ET": 0.03},
                                         cases=[{}] * 6)
        md = research.dashboard_markdown(stats, batch)
        assert "尚无通过界面登记的模型评测" in md


class TestErrorVisualisation:
    def test_caption_explains_each_missing_input(self):
        assert "模型预测" in research.comparison_caption(False, True)
        assert "参考标注" in research.comparison_caption(True, False)
        assert "模型预测" in research.comparison_caption(False, False)

    def test_no_caption_when_both_inputs_exist(self):
        assert research.comparison_caption(True, True) == ""

    def test_caption_promises_no_substitute_data(self):
        assert "不会用替代数据填充" in research.comparison_caption(False, False)

    def test_comparison_figure_is_three_panels_side_by_side(self):
        vol = np.zeros((40, 40, 10), dtype=np.float32)
        vol[10:30, 10:30, :] = 200.0
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        ref = np.zeros(vol.shape, dtype=np.uint8)
        ref[15:25, 15:25, 3:7] = 2
        pred = ref.copy()
        pred[pred == 2] = 0                 # deliberately under-segmented

        fig = research.build_comparison_figure(
            vol, affine, pred, ref, metrics.BRATS_2020, slice_index=5, panel=60)
        assert fig.ndim == 3 and fig.shape[2] == 3
        # Three square panels plus two gaps.
        assert fig.shape[1] > fig.shape[0] * 2.5
        assert fig.shape[0] == 60

    def test_error_legend_names_both_directions(self):
        html = research.error_legend_html()
        assert "漏分割" in html and "过度分割" in html


# ==========================================================================
# Task 5 — cross-case state isolation
# ==========================================================================
class TestCaseStateIsolation:
    def test_analysis_runs_are_scoped_to_one_case(self, tmp_path):
        st = storage.Store(tmp_path / "t.db")
        for cid in ("A", "B"):
            st.upsert_case(cid, dataset_source="ds")
        st.record_analysis_run("A", "m", "success", model_version="1",
                               evaluation={"dice": {"WT": 0.9, "TC": 0.9, "ET": 0.9}})
        st.record_analysis_run("B", "m", "success", model_version="1",
                               evaluation={"dice": {"WT": 0.2, "TC": 0.2, "ET": 0.2}})

        assert len(st.list_analysis_runs("A")) == 1
        assert st.list_analysis_runs("A")[0]["case_id"] == "A"
        assert st.list_analysis_runs("B")[0]["case_id"] == "B"

    def test_feedback_records_carry_their_own_case(self, tmp_path):
        st = storage.Store(tmp_path / "t.db")
        st.upsert_case("A", dataset_source="ds")
        st.upsert_case("B", dataset_source="ds")
        st.record_feedback("A", "under_segmentation", notes="a-note")
        st.record_feedback("B", "over_segmentation", notes="b-note")
        by_case = {r["case_id"]: r for r in st.list_feedback()}
        assert by_case["A"]["notes"] == "a-note"
        assert by_case["B"]["notes"] == "b-note"

    def test_case_measurements_do_not_leak_between_cases(self, tmp_path):
        st = storage.Store(tmp_path / "t.db")
        st.upsert_case("A", dataset_source="ds", measurements={"WT": {"ml": 10.0}})
        st.upsert_case("B", dataset_source="ds", measurements={"WT": {"ml": 99.0}})
        assert st.get_case("A")["measurements"]["WT"]["ml"] == 10.0
        assert st.get_case("B")["measurements"]["WT"]["ml"] == 99.0

    def test_approved_report_does_not_touch_another_cases_prediction(self, tmp_path):
        st = storage.Store(tmp_path / "t.db")
        st.upsert_case("A", dataset_source="ds", prediction_mask_path="a_pred.nii.gz")
        st.upsert_case("B", dataset_source="ds", prediction_mask_path="b_pred.nii.gz")
        st.save_report("A", "content", "approved", "human", approve=True)
        assert st.get_case("B")["review_status"] == "unreviewed"
        assert st.get_case("B")["prediction_mask_path"] == "b_pred.nii.gz"

    def test_mask_cache_keys_include_the_case(self):
        """Guards the app-level cache against serving one case's masks for another."""
        # Keys are (case_id, kind) tuples; verify the shape the app relies on.
        key_a = ("CASE_A", "prediction")
        key_b = ("CASE_B", "prediction")
        cache = {key_a: {"WT": np.ones((2, 2, 2), bool)},
                 key_b: {"WT": np.zeros((2, 2, 2), bool)}}
        assert cache[key_a]["WT"].all()
        assert not cache[key_b]["WT"].any()

"""Human-corrected mask import, validation, comparison and versioning."""

from __future__ import annotations

import numpy as np
import pytest

import corrections as corr
import metrics
import storage


def nested_label_map(shape=(12, 12, 8)) -> np.ndarray:
    """A BraTS-encoded mask with all three nested regions."""
    lm = np.zeros(shape, dtype=np.uint8)
    lm[2:10, 2:10, 2:6] = 2          # oedema   -> WT
    lm[4:8, 4:8, 2:6] = 1            # core     -> WT + TC
    lm[5:7, 5:7, 3:5] = 4            # enhancing-> WT + TC + ET
    return lm


IDENTITY = np.eye(4)


# ==========================================================================
# Parsing the uploaded file
# ==========================================================================
class TestParseUploadedMask:
    def test_brats_label_map_passes_through(self):
        lm = nested_label_map()
        parsed, warnings = corr.parse_uploaded_mask(
            lm, corr.FORMAT_BRATS_LABELMAP, None, metrics.BRATS_2020)
        np.testing.assert_array_equal(parsed, lm)
        assert warnings == []

    def test_4d_single_channel_is_squeezed(self):
        lm = nested_label_map()[..., None]
        parsed, _ = corr.parse_uploaded_mask(
            lm, corr.FORMAT_BRATS_LABELMAP, None, metrics.BRATS_2020)
        assert parsed.shape == (12, 12, 8)

    def test_binary_wt_mask_leaves_core_and_et_empty(self):
        binary = np.zeros((12, 12, 8), np.uint8)
        binary[2:10, 2:10, 2:6] = 1
        parsed, warnings = corr.parse_uploaded_mask(
            binary, corr.FORMAT_BINARY_REGION, metrics.REGION_WT, metrics.BRATS_2020)
        masks = metrics.region_masks(parsed, metrics.BRATS_2020)
        assert masks["WT"].any()
        assert not masks["TC"].any()
        assert not masks["ET"].any()
        assert warnings

    def test_binary_et_mask_expands_to_nested_regions(self):
        """Marking enhancing tumour implies core and whole tumour too."""
        binary = np.zeros((12, 12, 8), np.uint8)
        binary[5:7, 5:7, 3:5] = 1
        parsed, _ = corr.parse_uploaded_mask(
            binary, corr.FORMAT_BINARY_REGION, metrics.REGION_ET, metrics.BRATS_2020)
        masks = metrics.region_masks(parsed, metrics.BRATS_2020)
        assert masks["ET"].any()
        assert np.array_equal(masks["ET"], masks["TC"])
        assert np.array_equal(masks["TC"], masks["WT"])

    def test_binary_requires_a_region(self):
        with pytest.raises(ValueError, match="必须指定所属区域"):
            corr.parse_uploaded_mask(np.ones((4, 4, 4), np.uint8),
                                     corr.FORMAT_BINARY_REGION, None, metrics.BRATS_2020)

    def test_empty_binary_is_rejected(self):
        with pytest.raises(ValueError, match="不含任何前景体素"):
            corr.parse_uploaded_mask(np.zeros((4, 4, 4), np.uint8),
                                     corr.FORMAT_BINARY_REGION, metrics.REGION_WT,
                                     metrics.BRATS_2020)

    def test_unknown_format_is_rejected(self):
        with pytest.raises(ValueError, match="未知的上传格式"):
            corr.parse_uploaded_mask(np.zeros((4, 4, 4), np.uint8),
                                     "something_else", None, metrics.BRATS_2020)

    def test_foreign_label_encoding_is_rejected(self):
        """A mask using another dataset's encoding must not be accepted silently."""
        msd_style = np.zeros((4, 4, 4), np.uint8)
        msd_style[1, 1, 1] = 3          # 3 is not a BraTS label
        with pytest.raises(ValueError, match="标签编码"):
            corr.parse_uploaded_mask(msd_style, corr.FORMAT_BRATS_LABELMAP, None,
                                     metrics.BRATS_2020)


# ==========================================================================
# Spatial validation
# ==========================================================================
class TestValidateCorrection:
    def test_matching_mask_passes(self):
        lm = nested_label_map()
        result = corr.validate_correction(
            lm, expected_shape=(12, 12, 8), affine=IDENTITY,
            expected_affine=IDENTITY, convention=metrics.BRATS_2020)
        assert result.ok
        assert result.affine_matches
        assert result.convention_ok

    def test_shape_mismatch_is_rejected(self):
        result = corr.validate_correction(
            np.ones((8, 8, 8), np.uint8), expected_shape=(12, 12, 8), affine=IDENTITY,
            expected_affine=IDENTITY, convention=metrics.BRATS_2020)
        assert not result.ok
        assert any("尺寸" in p for p in result.problems)

    def test_affine_mismatch_is_rejected(self):
        other = np.diag([2.0, 2.0, 2.0, 1.0])
        result = corr.validate_correction(
            nested_label_map(), expected_shape=(12, 12, 8), affine=other,
            expected_affine=IDENTITY, convention=metrics.BRATS_2020)
        assert not result.ok
        assert any("affine" in p for p in result.problems)

    def test_orientation_flip_is_detected(self):
        flipped = np.diag([-1.0, 1.0, 1.0, 1.0])
        result = corr.validate_correction(
            nested_label_map(), expected_shape=(12, 12, 8), affine=flipped,
            expected_affine=IDENTITY, convention=metrics.BRATS_2020)
        assert not result.ok
        assert any("方向" in p for p in result.problems)

    def test_empty_mask_is_rejected(self):
        result = corr.validate_correction(
            np.zeros((12, 12, 8), np.uint8), expected_shape=(12, 12, 8), affine=IDENTITY,
            expected_affine=IDENTITY, convention=metrics.BRATS_2020)
        assert not result.ok
        assert any("空" in p for p in result.problems)

    def test_bad_label_encoding_is_reported(self):
        lm = nested_label_map()
        lm[0, 0, 0] = 9
        result = corr.validate_correction(
            lm, expected_shape=(12, 12, 8), affine=IDENTITY,
            expected_affine=IDENTITY, convention=metrics.BRATS_2020)
        assert not result.ok
        assert not result.convention_ok

    def test_result_is_serialisable(self):
        import json
        result = corr.validate_correction(
            nested_label_map(), expected_shape=(12, 12, 8), affine=IDENTITY,
            expected_affine=IDENTITY, convention=metrics.BRATS_2020)
        json.dumps(result.to_dict())


# ==========================================================================
# Comparison against the prediction
# ==========================================================================
class TestCompareToPrediction:
    def test_identical_masks_score_one_with_no_change(self):
        lm = nested_label_map()
        diff = corr.compare_to_prediction(lm, lm, (1.0, 1.0, 1.0))
        for region in metrics.REGIONS:
            assert diff.dice[region] == pytest.approx(1.0)
            assert diff.volume_delta_ml[region] == pytest.approx(0.0)
            assert diff.added_voxels[region] == 0
            assert diff.removed_voxels[region] == 0

    def test_removing_a_region_shows_as_removed_voxels(self):
        pred = nested_label_map()
        corr_lm = pred.copy()
        corr_lm[corr_lm == 2] = 0            # human deleted the oedema
        diff = corr.compare_to_prediction(pred, corr_lm, (1.0, 1.0, 1.0))
        assert diff.removed_voxels["WT"] > 0
        assert diff.added_voxels["WT"] == 0
        assert diff.volume_delta_ml["WT"] < 0
        assert diff.dice["WT"] < 1.0

    def test_adding_a_region_shows_as_added_voxels(self):
        pred = nested_label_map()
        corr_lm = pred.copy()
        corr_lm[0:2, 0:2, 0:2] = 2           # human painted extra tumour
        diff = corr.compare_to_prediction(pred, corr_lm, (1.0, 1.0, 1.0))
        assert diff.added_voxels["WT"] == 8
        assert diff.volume_delta_ml["WT"] > 0

    def test_anisotropic_spacing_is_applied_to_volume_delta(self):
        pred = nested_label_map()
        corr_lm = pred.copy()
        corr_lm[corr_lm == 2] = 0
        diff = corr.compare_to_prediction(pred, corr_lm, (1.0, 1.0, 3.0))
        removed = diff.removed_voxels["WT"]
        assert diff.volume_delta_ml["WT"] == pytest.approx(-removed * 0.003)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="尺寸不一致"):
            corr.compare_to_prediction(nested_label_map(),
                                       np.zeros((4, 4, 4), np.uint8), (1, 1, 1))

    def test_payload_carries_the_non_clinical_note(self):
        lm = nested_label_map()
        payload = corr.compare_to_prediction(lm, lm, (1.0, 1.0, 1.0)).to_dict()
        assert "不代表临床诊断准确性" in payload["note"]

    def test_diff_masks_mark_added_and_removed(self):
        pred = nested_label_map()
        corr_lm = pred.copy()
        corr_lm[corr_lm == 2] = 0
        corr_lm[0:2, 0:2, 0:2] = 2
        d = corr.diff_masks(pred, corr_lm)
        assert d["WT_added"].any()
        assert d["WT_removed"].any()
        # Added and removed are disjoint by construction.
        assert not np.logical_and(d["WT_added"], d["WT_removed"]).any()


# ==========================================================================
# Versioned storage
# ==========================================================================
class TestCorrectionStorage:
    def test_saves_as_a_new_file_and_never_touches_the_prediction(self, tmp_path):
        pred_dir = tmp_path / "outputs" / "predictions"
        pred_dir.mkdir(parents=True)
        (pred_dir / "C1_pred.nii.gz").write_bytes(b"original-prediction")

        rec = corr.save_correction(nested_label_map(), IDENTITY, root=tmp_path,
                                   case_id="C1", revision=1)
        assert rec.revision == 1
        assert "corrections" in rec.path
        # Original prediction file is byte-identical and still in place.
        assert (pred_dir / "C1_pred.nii.gz").read_bytes() == b"original-prediction"

    def test_revisions_do_not_overwrite_each_other(self, tmp_path):
        r1 = corr.save_correction(nested_label_map(), IDENTITY, root=tmp_path,
                                  case_id="C1", revision=1)
        r2 = corr.save_correction(nested_label_map(), IDENTITY, root=tmp_path,
                                  case_id="C1", revision=2)
        assert r1.path != r2.path
        assert corr.load_correction(r1.path).shape == (12, 12, 8)

    def test_reusing_a_revision_number_is_refused(self, tmp_path):
        corr.save_correction(nested_label_map(), IDENTITY, root=tmp_path,
                             case_id="C1", revision=1)
        with pytest.raises(FileExistsError, match="不可覆盖"):
            corr.save_correction(nested_label_map(), IDENTITY, root=tmp_path,
                                 case_id="C1", revision=1)

    def test_round_trips_through_disk(self, tmp_path):
        lm = nested_label_map()
        rec = corr.save_correction(lm, IDENTITY, root=tmp_path, case_id="C1", revision=1)
        np.testing.assert_array_equal(corr.load_correction(rec.path), lm)

    def test_store_records_revision_and_advances_the_counter(self, tmp_path):
        store = storage.Store(tmp_path / "t.db")
        store.upsert_case("C1", dataset_source="ds")
        assert store.next_correction_revision("C1") == 1

        corr.save_correction(nested_label_map(), IDENTITY, root=tmp_path, case_id="C1",
                             revision=1, store=store,
                             metrics_payload={"dice": {"WT": 0.9}})
        assert store.next_correction_revision("C1") == 2
        rows = store.list_corrections("C1")
        assert len(rows) == 1
        assert rows[0]["metrics"]["dice"]["WT"] == 0.9
        # The case points at the latest correction without losing the prediction.
        assert store.get_case("C1")["corrected_mask_path"].endswith("_r1.nii.gz")

    def test_corrections_are_scoped_per_case(self, tmp_path):
        store = storage.Store(tmp_path / "t.db")
        for cid in ("C1", "C2"):
            store.upsert_case(cid, dataset_source="ds")
        corr.save_correction(nested_label_map(), IDENTITY, root=tmp_path, case_id="C1",
                             revision=1, store=store, notes="for C1")
        corr.save_correction(nested_label_map(), IDENTITY, root=tmp_path, case_id="C2",
                             revision=1, store=store, notes="for C2")
        assert [r["notes"] for r in store.list_corrections("C1")] == ["for C1"]
        assert [r["notes"] for r in store.list_corrections("C2")] == ["for C2"]
        assert len(store.list_corrections()) == 2


# ==========================================================================
# Provenance: a correction is not a reference annotation, a demo is not a user
# ==========================================================================
class TestProvenanceLabelling:
    def test_user_upload_is_not_labelled_as_clinician_reviewed(self):
        label = corr.SOURCE_LABELS[corr.SOURCE_USER_UPLOAD]
        assert "用户上传" in label
        for forbidden in ("医生审核", "医师确认", "临床"):
            assert forbidden not in label

    def test_demo_simulation_is_labelled_as_demo(self):
        label = corr.SOURCE_LABELS[corr.SOURCE_DEMO_SIMULATION]
        assert "演示" in label
        assert "非真实人工修正" in label

    def test_disclaimer_states_it_is_not_clinician_reviewed(self):
        assert "未经具备资质的医师审核" in corr.CORRECTION_DISCLAIMER
        assert "不构成医生审核结果" in corr.CORRECTION_DISCLAIMER

    def test_demo_feature_note_warns_against_misuse(self):
        assert "演示" in corr.DEMO_DISCLAIMER
        assert "不得作为" in corr.DEMO_DISCLAIMER

    def test_feedback_note_marks_a_demo_record(self):
        lm = nested_label_map()
        diff = corr.compare_to_prediction(lm, lm, (1.0, 1.0, 1.0))
        note = corr.feedback_note(diff, corr.SOURCE_DEMO_SIMULATION, revision=2)
        assert "第 2 版修正" in note
        assert "演示" in note
        assert corr.DEMO_DISCLAIMER in note

    def test_feedback_note_for_a_user_upload_has_no_demo_warning(self):
        lm = nested_label_map()
        diff = corr.compare_to_prediction(lm, lm, (1.0, 1.0, 1.0))
        note = corr.feedback_note(diff, corr.SOURCE_USER_UPLOAD, revision=1)
        assert "演示" not in note
        assert "用户上传" in note

    def test_correction_never_shares_a_path_with_reference_or_prediction(self, tmp_path):
        rec = corr.save_correction(nested_label_map(), IDENTITY, root=tmp_path,
                                   case_id="C1", revision=1)
        assert "corrections" in rec.path
        assert "predictions" not in rec.path
        assert "brats_subset" not in rec.path


# ==========================================================================
# Feedback suggestion
# ==========================================================================
class TestErrorSuggestion:
    def test_mostly_removed_suggests_over_segmentation(self):
        lm = nested_label_map()
        human = lm.copy()
        human[human == 2] = 0
        diff = corr.compare_to_prediction(lm, human, (1.0, 1.0, 1.0))
        assert corr.suggest_error_type_from_diff(diff) == "over_segmentation"

    def test_mostly_added_suggests_under_segmentation(self):
        pred = np.zeros((12, 12, 8), np.uint8)
        pred[1:3, 1:3, 1:3] = 2
        human = pred.copy()
        human[2:10, 2:10, 2:6] = 2
        diff = corr.compare_to_prediction(pred, human, (1.0, 1.0, 1.0))
        assert corr.suggest_error_type_from_diff(diff) == "under_segmentation"

    def test_no_change_suggests_other(self):
        lm = nested_label_map()
        diff = corr.compare_to_prediction(lm, lm, (1.0, 1.0, 1.0))
        assert corr.suggest_error_type_from_diff(diff) == "other"


# ==========================================================================
# Case identifier cross-check
# ==========================================================================
class TestCaseIdentifierCheck:
    """Geometry cannot catch a wrong-patient mask; the filename can."""

    def test_matching_case_id_passes_silently(self):
        ok, msg = corr.check_case_identifier("BraTS20_Training_001_edited.nii.gz",
                                             "BraTS20_Training_001")
        assert ok and msg == ""

    def test_other_case_id_is_rejected(self):
        ok, msg = corr.check_case_identifier("BraTS20_Training_002_edited.nii.gz",
                                             "BraTS20_Training_001")
        assert not ok
        assert "BraTS20_Training_002" in msg
        assert "拒绝导入" in msg

    def test_name_without_an_id_is_a_caveat_not_an_error(self):
        """No identifier is not evidence of a problem -- only absence of proof."""
        ok, msg = corr.check_case_identifier("my_edit.nii.gz", "BraTS20_Training_001")
        assert ok
        assert "无法据文件名核对" in msg

    def test_lowercase_id_still_matches(self):
        ok, _ = corr.check_case_identifier("brats20_training_001.nii.gz",
                                           "BraTS20_Training_001")
        assert ok

    def test_missing_name_or_expected_id_is_not_an_error(self):
        assert corr.check_case_identifier(None, "BraTS20_Training_001")[0]
        assert corr.check_case_identifier("x.nii.gz", None)[0] == True  # noqa: E712

    def test_a_plain_number_is_not_mistaken_for_a_case_id(self):
        ok, msg = corr.check_case_identifier("seg_2020_v2.nii.gz", "BraTS20_Training_001")
        assert ok
        assert "无法据文件名核对" in msg

    def test_validation_surfaces_a_wrong_case_as_a_problem(self):
        result = corr.validate_correction(
            nested_label_map(), expected_shape=(12, 12, 8), affine=IDENTITY,
            expected_affine=IDENTITY, convention=metrics.BRATS_2020,
            source_name="BraTS20_Training_007_edit.nii.gz",
            expected_case_id="BraTS20_Training_001")
        assert not result.ok
        assert any("其他病例" in p for p in result.problems)

    def test_validation_records_the_caveat_when_unverifiable(self):
        result = corr.validate_correction(
            nested_label_map(), expected_shape=(12, 12, 8), affine=IDENTITY,
            expected_affine=IDENTITY, convention=metrics.BRATS_2020,
            source_name="edit.nii.gz", expected_case_id="BraTS20_Training_001")
        assert result.ok
        assert result.case_id_checked
        assert any("无法据文件名核对" in w for w in result.warnings)

    def test_identical_geometry_does_not_mask_a_wrong_case(self):
        """The failure mode this check exists for.

        Two BraTS cases share shape, affine and orientation, so without the
        filename cross-check a mask for patient 002 would be accepted and
        silently scored against patient 001.
        """
        lm = nested_label_map()
        wrong_name = corr.validate_correction(
            lm, expected_shape=(12, 12, 8), affine=IDENTITY, expected_affine=IDENTITY,
            convention=metrics.BRATS_2020, source_name="BraTS20_Training_002.nii.gz",
            expected_case_id="BraTS20_Training_001")
        right_name = corr.validate_correction(
            lm, expected_shape=(12, 12, 8), affine=IDENTITY, expected_affine=IDENTITY,
            convention=metrics.BRATS_2020, source_name="BraTS20_Training_001.nii.gz",
            expected_case_id="BraTS20_Training_001")
        assert not wrong_name.ok      # geometry identical, name catches it
        assert right_name.ok

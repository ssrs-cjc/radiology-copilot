"""Metric correctness: label decoding, nesting, volumes and Dice."""

from __future__ import annotations

import numpy as np
import pytest

import metrics


# --------------------------------------------------------------------------
# Label conventions
# --------------------------------------------------------------------------
class TestLabelConventions:
    def test_brats_encoding_matches_real_data(self):
        """BraTS 2018-2021 uses 1/2/4 with no label 3 (verified on real cases)."""
        assert metrics.BRATS_2020.wt_labels == (1, 2, 4)
        assert metrics.BRATS_2020.tc_labels == (1, 4)
        assert metrics.BRATS_2020.et_labels == (4,)
        assert 3 not in metrics.BRATS_2020.all_labels

    def test_msd_encoding_differs_from_brats(self):
        """MSD Task01 reuses the integers with different meanings - a real trap."""
        assert metrics.MSD_BRAIN_TUMOUR.et_labels == (3,)
        assert metrics.MSD_BRAIN_TUMOUR.tc_labels == (2, 3)
        assert metrics.MSD_BRAIN_TUMOUR.wt_labels == (1, 2, 3)
        # Same integer, different region: comparing across conventions is wrong.
        assert 3 not in metrics.BRATS_2020.all_labels

    def test_verify_convention_flags_unexpected_labels(self, brats_label_map):
        check = metrics.verify_convention(brats_label_map, metrics.BRATS_2020)
        assert check.ok

    def test_verify_convention_detects_foreign_labels(self):
        lm = np.zeros((4, 4, 4), dtype=np.uint8)
        lm[1, 1, 1] = 7                      # not part of any known convention
        check = metrics.verify_convention(lm, metrics.BRATS_2020)
        assert not check.ok
        assert "7" in check.summary()

    def test_missing_region_is_not_an_error(self):
        """~29% of BraTS cases have no enhancing tumour; that is not a fault."""
        lm = np.zeros((4, 4, 4), dtype=np.uint8)
        lm[1:3, 1:3, 1:3] = 1                 # core only, no label 4, no label 2
        check = metrics.verify_convention(lm, metrics.BRATS_2020)
        assert check.ok
        assert "absent" in check.summary()


# --------------------------------------------------------------------------
# Region decoding and nesting
# --------------------------------------------------------------------------
class TestRegions:
    def test_regions_are_nested(self, brats_label_map):
        """ET subset of TC subset of WT, which must hold for any BraTS label map."""
        m = metrics.region_masks(brats_label_map, metrics.BRATS_2020)
        assert np.all(np.logical_or(~m["ET"], m["TC"]))
        assert np.all(np.logical_or(~m["TC"], m["WT"]))

    def test_no_label_3_region_in_brats(self, brats_label_map):
        lm = brats_label_map.copy()
        lm[0, 0, 0] = 3
        m = metrics.region_masks(lm, metrics.BRATS_2020)
        # label 3 belongs to no region under the BraTS convention
        assert not m["WT"][0, 0, 0]
        assert not m["TC"][0, 0, 0]

    def test_msd_decoding_assigns_3_to_all_three_regions(self):
        lm = np.zeros((3, 3, 3), dtype=np.uint8)
        lm[1, 1, 1] = 3
        m = metrics.region_masks(lm, metrics.MSD_BRAIN_TUMOUR)
        assert m["WT"][1, 1, 1] and m["TC"][1, 1, 1] and m["ET"][1, 1, 1]


class TestMasksToLabelMap:
    def test_nested_channels_collapse_with_et_precedence(self):
        """Overlapping model channels must not corrupt the stored label map."""
        shape = (4, 4, 2)
        w = np.zeros(shape, bool); w[0:3, 0:3, :] = True
        t = np.zeros(shape, bool); t[0:2, 0:2, :] = True
        e = np.zeros(shape, bool); e[0:1, 0:1, :] = True
        lm = metrics.masks_to_label_map({"WT": w, "TC": t, "ET": e})
        assert lm[0, 0, 0] == 4          # ET wins where all three overlap
        assert lm[1, 1, 0] == 1          # TC only
        assert lm[2, 2, 0] == 2          # WT only
        assert lm[3, 3, 0] == 0          # background

    def test_round_trip_through_label_map(self, brats_label_map):
        m = metrics.region_masks(brats_label_map, metrics.BRATS_2020)
        rebuilt = metrics.masks_to_label_map(m)
        np.testing.assert_array_equal(rebuilt, brats_label_map)


# --------------------------------------------------------------------------
# Volumes
# --------------------------------------------------------------------------
class TestVolumes:
    def test_isotropic_1mm_volume_equals_voxel_count_in_ml(self, brats_label_map, spacing):
        vols = metrics.all_region_volumes(
            metrics.region_masks(brats_label_map, metrics.BRATS_2020), spacing)
        for region in metrics.REGIONS:
            assert vols[region].volume_ml == pytest.approx(vols[region].voxel_count * 0.001)

    def test_anisotropic_spacing_is_applied(self):
        lm = np.zeros((4, 4, 4), dtype=np.uint8)
        lm[0, 0, 0] = 1
        masks = metrics.region_masks(lm, metrics.BRATS_2020)
        vols = metrics.all_region_volumes(masks, (1.0, 1.0, 3.0))
        # one voxel of 1x1x3 mm = 3 mm^3 = 0.003 mL
        assert vols["WT"].volume_ml == pytest.approx(0.003)

    def test_empty_mask_is_zero_not_none(self):
        lm = np.zeros((4, 4, 4), dtype=np.uint8)
        vols = metrics.all_region_volumes(metrics.region_masks(lm, metrics.BRATS_2020), (1, 1, 1))
        assert vols["WT"].volume_ml == 0.0
        assert vols["WT"].voxel_count == 0

    def test_volumes_are_ordered_wt_ge_tc_ge_et(self, brats_label_map, spacing):
        vols = metrics.all_region_volumes(
            metrics.region_masks(brats_label_map, metrics.BRATS_2020), spacing)
        assert vols["WT"].volume_ml >= vols["TC"].volume_ml >= vols["ET"].volume_ml


# --------------------------------------------------------------------------
# Dice
# --------------------------------------------------------------------------
class TestDice:
    def test_identical_masks_score_one(self, brats_label_map):
        m = metrics.region_masks(brats_label_map, metrics.BRATS_2020)
        for region in metrics.REGIONS:
            assert metrics.dice(m[region], m[region], region).dice == pytest.approx(1.0)

    def test_disjoint_masks_score_zero(self):
        a = np.zeros((4, 4, 4), bool); a[0, 0, 0] = True
        b = np.zeros((4, 4, 4), bool); b[1, 1, 1] = True
        assert metrics.dice(a, b, "WT").dice == pytest.approx(0.0)

    def test_half_overlap_gives_known_value(self):
        a = np.zeros((2, 1, 1), bool); a[0, 0, 0] = True; a[1, 0, 0] = True
        b = np.zeros((2, 1, 1), bool); b[0, 0, 0] = True
        # 2*1 / (2+1) = 0.6667
        assert metrics.dice(a, b, "WT").dice == pytest.approx(2 / 3)

    def test_both_empty_scores_one_with_a_note(self):
        z = np.zeros((4, 4, 4), bool)
        r = metrics.dice(z, z, "ET")
        assert r.dice == pytest.approx(1.0)
        assert "both masks empty" in r.note

    def test_one_empty_scores_zero_with_a_note(self):
        a = np.zeros((4, 4, 4), bool); a[0, 0, 0] = True
        z = np.zeros((4, 4, 4), bool)
        r = metrics.dice(a, z, "ET")
        assert r.dice == pytest.approx(0.0)
        assert "one mask empty" in r.note

    def test_shape_mismatch_raises_rather_than_broadcasting(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            metrics.dice(np.zeros((4, 4, 4), bool), np.zeros((5, 5, 5), bool))

    def test_dice_is_symmetric(self, brats_label_map):
        m = metrics.region_masks(brats_label_map, metrics.BRATS_2020)
        shifted = np.roll(brats_label_map, 1, axis=0)
        ms = metrics.region_masks(shifted, metrics.BRATS_2020)
        for region in metrics.REGIONS:
            assert (metrics.dice(m[region], ms[region]).dice
                    == pytest.approx(metrics.dice(ms[region], m[region]).dice))


# --------------------------------------------------------------------------
# Full evaluation
# --------------------------------------------------------------------------
class TestEvaluation:
    def test_perfect_prediction(self, brats_label_map, spacing):
        ev = metrics.evaluate_segmentation(brats_label_map, brats_label_map, spacing)
        assert ev.mean_dice() == pytest.approx(1.0)
        assert sum(ev.missed.values()) == 0
        assert sum(ev.extra.values()) == 0

    def test_undesegmentation_shows_missed_voxels(self, brats_label_map, spacing):
        pred = brats_label_map.copy()
        pred[pred == 2] = 0                       # drop all oedema
        ev = metrics.evaluate_segmentation(pred, brats_label_map, spacing)
        assert ev.missed["WT"] > 0
        assert ev.extra["WT"] == 0
        assert ev.dice["WT"].dice < 1.0

    def test_oversegmentation_shows_extra_voxels(self, brats_label_map, spacing):
        pred = brats_label_map.copy()
        pred[0:2, 0:2, :] = 2                     # add spurious region
        ev = metrics.evaluate_segmentation(pred, brats_label_map, spacing)
        assert ev.extra["WT"] > 0

    def test_shape_mismatch_is_rejected(self, brats_label_map, spacing):
        with pytest.raises(ValueError, match="resample before evaluating"):
            metrics.evaluate_segmentation(np.zeros((2, 2, 2), np.uint8), brats_label_map, spacing)

    def test_as_dict_is_json_serialisable(self, brats_label_map, spacing):
        import json
        ev = metrics.evaluate_segmentation(brats_label_map, brats_label_map, spacing)
        json.dumps(ev.as_dict())          # must not raise

    def test_error_profile_classifies_direction(self):
        assert metrics.error_profile({"WT": 1000}, {"WT": 10}) == "疑似漏分割"
        assert metrics.error_profile({"WT": 10}, {"WT": 1000}) == "疑似过度分割"
        assert metrics.error_profile({"WT": 100}, {"WT": 100}) == "疑似边界偏差"
        assert metrics.error_profile({}, {}) == "无差异"

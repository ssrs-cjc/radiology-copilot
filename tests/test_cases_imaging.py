"""Case registry, modality ordering and MRI input validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import cases as cases_mod
import imaging


def write_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> Path:
    """Local helper: the name ``tests`` is shadowed by a site-packages package,
    so shared helpers are duplicated rather than imported across test modules."""
    import nibabel as nib

    if affine is None:
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), str(path))
    return path


# --------------------------------------------------------------------------
# Modality ordering - the single highest-risk mapping in the project
# --------------------------------------------------------------------------
class TestChannelOrdering:
    def test_monai_bundle_order_is_t1ce_t1_t2_flair(self):
        """Verified against the bundle's own scripts/prepare_datalist.py:
        ``"image": t1ce + t1 + t2 + flair``."""
        assert cases_mod.MONAI_BRATS_CHANNEL_ORDER == ("t1ce", "t1", "t2", "flair")

    def test_order_channels_stacks_in_model_order_not_alphabetical(self):
        c = cases_mod.Case(
            case_id="x", dataset_source="t", modalities=("flair", "t1", "t1ce", "t2"),
            image_paths={}, label_convention="brats_2020",
        )
        # Distinct constant per modality so the stacking order is observable.
        vols = {
            "t1": np.full((2, 2, 2), 1.0),
            "t1ce": np.full((2, 2, 2), 2.0),
            "t2": np.full((2, 2, 2), 3.0),
            "flair": np.full((2, 2, 2), 4.0),
        }
        out = cases_mod.order_channels(c, vols)
        assert out.shape == (2, 2, 2, 4)
        assert out[..., 0].mean() == 2.0    # T1ce first
        assert out[..., 1].mean() == 1.0    # then T1
        assert out[..., 2].mean() == 3.0    # then T2
        assert out[..., 3].mean() == 4.0    # then FLAIR

    def test_order_channels_rejects_shape_mismatch(self):
        c = cases_mod.Case(case_id="x", dataset_source="t", modalities=("t1",),
                           image_paths={}, label_convention="brats_2020")
        vols = {"t1": np.zeros((2, 2, 2)), "t1ce": np.zeros((3, 3, 3)),
                "t2": np.zeros((2, 2, 2)), "flair": np.zeros((2, 2, 2))}
        with pytest.raises(ValueError, match="空间尺寸不一致"):
            cases_mod.order_channels(c, vols)

    def test_order_channels_reports_missing_modality(self):
        c = cases_mod.Case(case_id="x", dataset_source="t", modalities=("t1",),
                           image_paths={}, label_convention="brats_2020")
        with pytest.raises(ValueError, match="缺少模型所需序列"):
            cases_mod.order_channels(c, {"t1": np.zeros((2, 2, 2))})

    def test_channel_order_description_is_explicit(self):
        c = cases_mod.Case(case_id="x", dataset_source="t", modalities=(), image_paths={},
                           label_convention="brats_2020")
        desc = cases_mod.channel_order_description(c)
        assert "ch0:T1ce" in desc and "ch1:T1" in desc and "ch3:FLAIR" in desc


# --------------------------------------------------------------------------
# Case integrity
# --------------------------------------------------------------------------
class TestCaseIntegrity:
    def test_verify_reports_missing_files(self):
        c = cases_mod.Case(
            case_id="ghost", dataset_source="t", modalities=("t1",),
            image_paths={"t1": "definitely/not/here.nii"},
            label_convention="brats_2020")
        problems = c.verify()
        assert any("缺少影像文件" in p for p in problems)

    def test_verify_checks_label_convention_against_real_labels(self, brats_label_map):
        c = cases_mod.Case(case_id="x", dataset_source="t", modalities=(), image_paths={},
                           label_convention="brats_2020")
        assert c.verify(brats_label_map) == []

        bad = brats_label_map.copy()
        bad[0, 0, 0] = 9
        assert any("9" in p for p in c.verify(bad))

    def test_unknown_convention_is_reported(self):
        c = cases_mod.Case(case_id="x", dataset_source="t", modalities=(), image_paths={},
                           label_convention="nope")
        assert any("未知的标签编码约定" in p for p in c.verify())

    def test_synthetic_cases_are_labelled_as_synthetic(self):
        """The UI must never be able to present synthetic data as a real case."""
        c = cases_mod.Case(case_id="s", dataset_source="synth", modalities=(),
                           image_paths={}, label_convention="brats_2020",
                           data_origin=cases_mod.ORIGIN_SYNTHETIC)
        assert c.origin_label == "合成测试数据"
        assert c.data_origin != cases_mod.ORIGIN_PUBLIC

    def test_registry_loads_without_data(self, tmp_path):
        """A machine with no dataset must still produce a usable (empty) registry."""
        reg = cases_mod.load_registry(tmp_path / "empty")
        assert reg.ids() == []
        assert reg.warnings                       # must explain how to get data


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------
class TestValidation:
    def test_complete_case_passes(self, synth_case):
        rep = imaging.validate_case(synth_case["image_paths"], synth_case["reference_mask_path"])
        assert rep.ok
        assert set(rep.modalities_present) == {"t1", "t1ce", "t2", "flair"}
        assert rep.shape == (16, 16, 8)

    def test_missing_modality_is_an_error(self, synth_case):
        paths = dict(synth_case["image_paths"])
        paths.pop("flair")
        rep = imaging.validate_case(paths)
        assert not rep.ok
        assert any("flair" in i.message.lower() for i in rep.errors)

    def test_unreadable_file_is_reported_not_raised(self, tmp_path):
        bad = tmp_path / "broken.nii"
        bad.write_bytes(b"this is not a nifti file")
        rep = imaging.validate_case({"t1": str(bad)})
        assert not rep.ok
        assert any("无法读取" in i.message for i in rep.errors)

    def test_reference_shape_mismatch_is_detected(self, synth_case, tmp_path):
        wrong = write_nifti(tmp_path / "wrong_seg.nii", np.zeros((8, 8, 8), np.uint8))
        rep = imaging.validate_case(synth_case["image_paths"], str(wrong))
        assert not rep.ok
        assert any("参考标注与影像空间尺寸不一致" in i.message for i in rep.errors)

    def test_missing_reference_is_flagged_as_info_not_error(self, synth_case):
        rep = imaging.validate_case(synth_case["image_paths"], None)
        assert rep.ok
        assert any("没有参考标注" in i.message for i in rep.issues)

    def test_report_renders_as_markdown(self, synth_case):
        rep = imaging.validate_case(synth_case["image_paths"], synth_case["reference_mask_path"])
        assert "输入检查通过" in rep.to_markdown()


# --------------------------------------------------------------------------
# Geometry and rendering
# --------------------------------------------------------------------------
class TestGeometry:
    def test_axial_axis_follows_the_affine(self, synth_case):
        """Axis choice is derived, not assumed to be the last one."""
        assert imaging.axial_axis(synth_case["affine"]) == 2
        # A volume stored with the axial direction first must be detected.
        reoriented = np.asarray([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
        assert imaging.axial_axis(reoriented) == 1

    def test_slice_count_matches_the_axial_axis(self, synth_case):
        vol = imaging.load_volume(synth_case["image_paths"]["t1"])
        assert imaging.axial_slice_count(vol.data, vol.affine) == 8

    def test_out_of_range_index_is_clamped(self, synth_case):
        vol = imaging.load_volume(synth_case["image_paths"]["t1"])
        sl = imaging.take_axial_slice(vol.data, 999, vol.affine)
        assert sl.shape == (16, 16)

    def test_tumour_center_slice_finds_the_lesion(self, synth_case):
        vol = imaging.load_volume(synth_case["image_paths"]["t1"])
        center = imaging.tumour_center_slice(synth_case["label_map"], vol.affine)
        assert center is not None and 2 <= center <= 5

    def test_tumour_center_slice_returns_none_for_empty_mask(self, synth_case):
        vol = imaging.load_volume(synth_case["image_paths"]["t1"])
        assert imaging.tumour_center_slice(np.zeros((16, 16, 8), np.uint8), vol.affine) is None


class TestRendering:
    def test_render_returns_uint8_rgb(self, synth_case):
        vol = imaging.load_volume(synth_case["image_paths"]["t1ce"])
        img = imaging.render_mri_slice(vol.data, vol.affine, 4)
        assert img.dtype == np.uint8
        assert img.ndim == 3 and img.shape[2] == 3

    def test_overlay_changes_pixels_only_inside_the_mask(self, synth_case):
        vol = imaging.load_volume(synth_case["image_paths"]["t1ce"])
        masks = {"WT": synth_case["label_map"] > 0}
        plain = imaging.render_mri_slice(vol.data, vol.affine, 4, masks=masks, show_masks=False)
        overlaid = imaging.render_mri_slice(vol.data, vol.affine, 4, masks=masks, show_masks=True)
        assert not np.array_equal(plain, overlaid)
        # A slice with no mask content must be untouched.
        empty = imaging.render_mri_slice(vol.data, vol.affine, 0, masks=masks, show_masks=True)
        plain0 = imaging.render_mri_slice(vol.data, vol.affine, 0, masks=masks, show_masks=False)
        np.testing.assert_array_equal(empty, plain0)

    def test_windowing_handles_constant_slices_without_dividing_by_zero(self):
        flat = np.full((8, 8), 5.0)
        out = imaging.window_slice(flat)
        assert out.dtype == np.uint8
        assert np.all(out == 0)          # constant input, no NaN/inf produced

    def test_percentile_bounds_change_contrast(self):
        ramp = np.arange(1, 101, dtype=np.float32).reshape(10, 10)
        narrow = imaging.window_slice(ramp, low_pct=50, high_pct=60)
        wide = imaging.window_slice(ramp, low_pct=0, high_pct=100)
        assert narrow.mean() < wide.mean()

    def test_background_does_not_collapse_the_window(self):
        """Regression: percentiles must ignore the zero background.

        Brain MRI is skull-stripped, so most voxels are exact zeros.  If those
        are included in the percentile sample, a bright-tissue slice maps almost
        entirely to white - the bug that made the viewer show a blank blob.
        """
        sl = np.zeros((100, 100), dtype=np.float32)
        sl[20:80, 20:80] = np.arange(1, 3601, dtype=np.float32).reshape(60, 60)

        out = imaging.window_slice(sl, low_pct=1, high_pct=99)
        tissue = out[20:80, 20:80]

        # A real window must spread tissue across the range, not saturate it.
        assert tissue.min() < 60, "tissue should reach dark greys, not start bright"
        assert tissue.max() > 200, "brightest tissue should still reach white"
        assert 40 < tissue.mean() < 210, "tissue mean should sit mid-range"

    def test_window_never_produces_nan(self):
        for sl in (np.zeros((8, 8), np.float32), np.full((8, 8), 7.0, np.float32),
                   np.arange(64, dtype=np.float32).reshape(8, 8)):
            out = imaging.window_slice(sl)
            assert out.dtype == np.uint8
            assert np.isfinite(out).all()

    def test_legend_lists_all_three_regions(self):
        html = imaging.legend_html()
        for label in ("全肿瘤区", "肿瘤核心区", "强化肿瘤区"):
            assert label in html

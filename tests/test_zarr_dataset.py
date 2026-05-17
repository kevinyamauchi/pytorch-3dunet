import numpy as np
import pytest
import zarr

from pytorch3dunet.datasets.zarr_dataset import LazyZarrDataset


def _create_zarr_store(tmp_path, raw_dtype=np.float64):
    store_path = tmp_path / "sample.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    root.create_array(
        "raw",
        data=np.ones((1, 80, 80, 80), dtype=raw_dtype),
        chunks=(1, 40, 40, 40),
    )
    root.create_array(
        "label",
        data=np.ones((80, 80, 80), dtype=np.uint8),
        chunks=(40, 40, 40),
    )
    return str(store_path)


def _slice_builder_config():
    return {
        "name": "FilterSliceBuilder",
        "patch_shape": [80, 80, 80],
        "stride_shape": [80, 80, 80],
        "threshold": 0.1,
    }


def _transformer_config():
    return {
        "raw": [{"name": "Identity"}],
        "label": [{"name": "ToTensor", "expand_dims": False, "dtype": "long"}],
    }


def test_lazy_zarr_raw_dtype_casts_patch_before_transform(tmp_path):
    store_path = _create_zarr_store(tmp_path, raw_dtype=np.float64)

    dataset = LazyZarrDataset(
        store_path,
        phase="train",
        slice_builder_config=_slice_builder_config(),
        transformer_config=_transformer_config(),
        raw_internal_path="raw",
        label_internal_path="label",
        raw_dtype="float32",
    )

    raw_patch, _label_patch = dataset[0]

    assert raw_patch.dtype == np.float32
    assert raw_patch.shape == (1, 80, 80, 80)


def test_lazy_zarr_construction_does_not_materialize_raw_array(tmp_path, monkeypatch):
    store_path = _create_zarr_store(tmp_path)
    original_asarray = np.asarray

    def guarded_asarray(value, *args, **kwargs):
        if type(value).__module__.startswith("zarr"):
            raise AssertionError("lazy zarr arrays must not be materialized during construction")
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", guarded_asarray)

    LazyZarrDataset(
        store_path,
        phase="train",
        slice_builder_config=_slice_builder_config(),
        transformer_config=_transformer_config(),
        raw_internal_path="raw",
        label_internal_path="label",
    )


def test_filter_slice_builder_does_not_copy_label_patches(tmp_path, monkeypatch):
    store_path = _create_zarr_store(tmp_path)

    def fail_copy(*_args, **_kwargs):
        raise AssertionError("FilterSliceBuilder should not copy label patches while filtering")

    monkeypatch.setattr(np, "copy", fail_copy)

    dataset = LazyZarrDataset(
        store_path,
        phase="train",
        slice_builder_config=_slice_builder_config(),
        transformer_config=_transformer_config(),
        raw_internal_path="raw",
        label_internal_path="label",
    )

    assert len(dataset) == 1

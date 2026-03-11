"""
Adapter to cleanly load generated cases for evaluation.

Responsibilities
----------------
- scan generated case directories
- load `ensemble_members.npz`, `ensemble_mean.npz`, `pmm.npz`
- load lightweight per-case metadata
- expose clean typed case objects to the evaluator
- provide consistent access to the forecast product needed by each metric family

Design notes
------------
This loader is intentionally generation-output-first. It treats the saved generation
artifacts as the source of truth for forecast products and associated metadata. The
reference target currently comes from the saved generation artifacts as well, which keeps
this first evaluation iteration simple and robust.

A later iteration can optionally re-load reference data directly from the dataset adapter
if needed for stricter provenance checks or for additional reference-side diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
import json

import numpy as np

from stride_core.evaluation.evaluation_config import EvaluationRunConfig


# -----------------------------------------------------------------------------
# Typed containers
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedProduct:
    name: str
    npz_path: Path
    json_path: Path | None
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LoadedEvaluationCase:
    case_id: str
    case_dir: Path
    date: str | None
    domain_tag: str | None
    variable_names: list[str]
    products: dict[str, LoadedProduct]

    @property
    def available_products(self) -> tuple[str, ...]:
        return tuple(sorted(self.products.keys()))

    def get_product(self, name: str) -> LoadedProduct:
        if name not in self.products:
            raise KeyError(
                f"Requested product {name!r} is not available for case {self.case_id}. "
                f"Available products: {sorted(self.products.keys())}"
            )
        return self.products[name]


# -----------------------------------------------------------------------------
# Public loader
# -----------------------------------------------------------------------------


class EvaluationDataLoader:
    """
    Load generated evaluation products from a STRIDE generation run.

    Expected directory layout
    -------------------------
    <generation_output_dir>/
        generation_metadata.json
        samples/
            <case_id>/
                ensemble_members.npz/json    (optional, depending on storage mode)
                member_0000.npz/json         (optional, depending on storage mode)
                ensemble_mean.npz/json       (optional)
                pmm.npz/json                 (optional)
    """

    def __init__(self, cfg: EvaluationRunConfig) -> None:
        self.cfg = cfg
        self.generation_output_dir = cfg.paths.generation_output_dir
        self.samples_dir = self.generation_output_dir / "samples"
        self.generation_metadata_path = self.generation_output_dir / "generation_metadata.json"

        self.generation_metadata = self._load_optional_json(self.generation_metadata_path)
        self.case_dirs = self._discover_case_dirs()
        self.storage_mode = self._infer_storage_mode()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_case_dirs(self) -> list[Path]:
        if not self.samples_dir.exists():
            raise FileNotFoundError(
                f"Generation samples directory does not exist: {self.samples_dir}"
            )

        case_dirs = sorted(path for path in self.samples_dir.iterdir() if path.is_dir())
        if len(case_dirs) == 0:
            raise FileNotFoundError(
                f"No case directories were found under generation samples dir: {self.samples_dir}"
            )

        if self.cfg.data.use_case_limit is not None:
            case_dirs = case_dirs[: self.cfg.data.use_case_limit]
        return case_dirs

    def _infer_storage_mode(self) -> str:
        metadata_mode = self.generation_metadata.get("storage_mode")
        if isinstance(metadata_mode, str) and metadata_mode:
            return metadata_mode

        first_case_dir = self.case_dirs[0]
        if (first_case_dir / "ensemble_members.npz").exists():
            return "per_member_bundle"
        if (first_case_dir / "member_0000.npz").exists():
            return "per_member"
        return "unknown"

    # ------------------------------------------------------------------
    # Case iteration
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.case_dirs)

    def __iter__(self) -> Iterator[LoadedEvaluationCase]:
        for case_dir in self.case_dirs:
            yield self.load_case(case_dir)

    def iter_cases(self) -> Iterator[LoadedEvaluationCase]:
        return iter(self)

    def load_all_cases(self) -> list[LoadedEvaluationCase]:
        return list(self.iter_cases())

    # ------------------------------------------------------------------
    # Case loading
    # ------------------------------------------------------------------

    def load_case(self, case_dir: str | Path) -> LoadedEvaluationCase:
        case_dir = Path(case_dir)
        if not case_dir.exists():
            raise FileNotFoundError(f"Case directory does not exist: {case_dir}")

        products: dict[str, LoadedProduct] = {}

        member_bundle_npz = case_dir / "ensemble_members.npz"
        if member_bundle_npz.exists():
            products["ensemble_members"] = self._load_product(
                name="ensemble_members",
                npz_path=member_bundle_npz,
                json_path=case_dir / "ensemble_members.json",
            )

        ensemble_mean_npz = case_dir / "ensemble_mean.npz"
        if ensemble_mean_npz.exists():
            products["ensemble_mean"] = self._load_product(
                name="ensemble_mean",
                npz_path=ensemble_mean_npz,
                json_path=case_dir / "ensemble_mean.json",
            )

        pmm_npz = case_dir / "pmm.npz"
        if pmm_npz.exists():
            products["pmm"] = self._load_product(
                name="pmm",
                npz_path=pmm_npz,
                json_path=case_dir / "pmm.json",
            )

        if self.storage_mode == "per_member":
            member_paths = sorted(case_dir.glob("member_*.npz"))
            if len(member_paths) > 0:
                products["ensemble_members"] = self._build_ensemble_from_member_files(
                    case_dir=case_dir,
                    member_npz_paths=member_paths,
                )

        if len(products) == 0:
            raise FileNotFoundError(
                f"No evaluation products were found in case directory: {case_dir}"
            )

        representative = self._choose_representative_product(products)
        metadata = representative.metadata

        variable_names_raw = metadata.get("variable_names", [])
        if isinstance(variable_names_raw, list):
            variable_names = [str(x) for x in variable_names_raw]
        else:
            variable_names = []

        return LoadedEvaluationCase(
            case_id=case_dir.name,
            case_dir=case_dir,
            date=self._maybe_str(metadata.get("date")),
            domain_tag=self._maybe_str(metadata.get("domain_tag")),
            variable_names=variable_names,
            products=products,
        )

    def _choose_representative_product(
        self,
        products: dict[str, LoadedProduct],
    ) -> LoadedProduct:
        for preferred_name in ("pmm", "ensemble_mean", "ensemble_members"):
            product = products.get(preferred_name)
            if product is not None:
                return product
        return next(iter(products.values()))

    # ------------------------------------------------------------------
    # Family-specific product selection
    # ------------------------------------------------------------------

    def get_spatial_product(self, case: LoadedEvaluationCase) -> LoadedProduct:
        return case.get_product(self.cfg.data.forecast_product_for_spatial)

    def get_climatology_product(self, case: LoadedEvaluationCase) -> LoadedProduct:
        return case.get_product(self.cfg.data.forecast_product_for_climatology)

    def get_temporal_product(self, case: LoadedEvaluationCase) -> LoadedProduct:
        return case.get_product(self.cfg.data.forecast_product_for_temporal)

    def get_probabilistic_product(self, case: LoadedEvaluationCase) -> LoadedProduct:
        return case.get_product("ensemble_members")

    # ------------------------------------------------------------------
    # Product loading
    # ------------------------------------------------------------------

    def _load_product(
        self,
        *,
        name: str,
        npz_path: Path,
        json_path: Path | None,
    ) -> LoadedProduct:
        arrays = self._load_npz(npz_path)
        metadata = self._load_optional_json(json_path)
        return LoadedProduct(
            name=name,
            npz_path=npz_path,
            json_path=json_path,
            arrays=arrays,
            metadata=metadata,
        )

    def _build_ensemble_from_member_files(
        self,
        *,
        case_dir: Path,
        member_npz_paths: list[Path],
    ) -> LoadedProduct:
        member_arrays: list[np.ndarray] = []
        reference_arrays: dict[str, np.ndarray] | None = None
        representative_metadata: dict[str, Any] = {}

        for idx, member_path in enumerate(member_npz_paths):
            arrays = self._load_npz(member_path)
            metadata = self._load_optional_json(member_path.with_suffix(".json"))

            generated = arrays.get("generated_physical")
            if generated is None:
                generated = arrays.get("generated")
            if generated is None:
                raise KeyError(
                    f"Member file does not contain 'generated_physical' or 'generated': {member_path}"
                )

            member_arrays.append(np.asarray(generated))

            if idx == 0:
                reference_arrays = {
                    key: value
                    for key, value in arrays.items()
                    if key not in {"generated", "generated_physical"}
                }
                representative_metadata = metadata

        if reference_arrays is None:
            raise ValueError(
                f"Could not build ensemble from member files in case directory: {case_dir}"
            )

        stacked = np.stack(member_arrays, axis=0)
        if stacked.ndim >= 2 and stacked.shape[1] == 1:
            stacked = stacked[:, 0]

        bundled_arrays = dict(reference_arrays)
        bundled_arrays["generated_members"] = stacked

        json_path = case_dir / "ensemble_members.json"
        metadata = dict(representative_metadata)
        metadata["aggregate_name"] = "ensemble_members"
        metadata["ensemble_size"] = int(len(member_npz_paths))

        return LoadedProduct(
            name="ensemble_members",
            npz_path=case_dir / "ensemble_members.npz",
            json_path=json_path if json_path.exists() else None,
            arrays=bundled_arrays,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Convenience array accessors
    # ------------------------------------------------------------------

    def get_target_array(self, product: LoadedProduct) -> np.ndarray:
        if "target_physical" in product.arrays:
            arr = np.asarray(product.arrays["target_physical"])
        elif "target" in product.arrays:
            arr = np.asarray(product.arrays["target"])
        else:
            raise KeyError(
                f"Product {product.name!r} does not contain target arrays. Keys: {sorted(product.arrays.keys())}"
            )

        if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 1:
            return arr[0, 0, ...]
        if arr.ndim == 4 and arr.shape[1] == 1:
            return arr[:, 0, ...]
        return arr

    def get_forecast_array(self, product: LoadedProduct) -> np.ndarray:
        if product.name == "ensemble_members":
            if "generated_members" not in product.arrays:
                raise KeyError(
                    f"Ensemble product is missing 'generated_members'. Keys: {sorted(product.arrays.keys())}"
                )
            arr = np.asarray(product.arrays["generated_members"])
            if arr.ndim == 5 and arr.shape[1] == 1 and arr.shape[2] == 1:
                return arr[:, 0, 0, ...]
            if arr.ndim == 4 and arr.shape[1] == 1:
                return arr[:, 0, ...]
            return arr

        if "generated_physical" in product.arrays:
            arr = np.asarray(product.arrays["generated_physical"])
        elif "generated" in product.arrays:
            arr = np.asarray(product.arrays["generated"])
        else:
            raise KeyError(
                f"Product {product.name!r} does not contain generated arrays. Keys: {sorted(product.arrays.keys())}"
            )

        if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 1:
            return arr[0, 0, ...]
        if arr.ndim == 4 and arr.shape[1] == 1:
            return arr[:, 0, ...]
        return arr

    def get_cond_dynamic_array(self, product: LoadedProduct) -> np.ndarray | None:
        if "cond_dynamic_physical" in product.arrays:
            return np.asarray(product.arrays["cond_dynamic_physical"])
        if "cond_dynamic" in product.arrays:
            return np.asarray(product.arrays["cond_dynamic"])
        return None

    def get_cond_static_array(self, product: LoadedProduct) -> np.ndarray | None:
        if "cond_static_physical" in product.arrays:
            return np.asarray(product.arrays["cond_static_physical"])
        if "cond_static" in product.arrays:
            return np.asarray(product.arrays["cond_static"])
        return None

    # ------------------------------------------------------------------
    # Internal file helpers
    # ------------------------------------------------------------------

    def _load_npz(self, path: Path) -> dict[str, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(f"Expected NPZ file does not exist: {path}")
        loaded = np.load(path, allow_pickle=False)
        return {key: loaded[key] for key in loaded.files}

    def _load_optional_json(self, path: Path | None) -> dict[str, Any]:
        if path is None or not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {path}, got {type(payload)}")
        return payload

    def _maybe_str(self, value: Any) -> str | None:
        if value is None:
            return None
        return str(value)
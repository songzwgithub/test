from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from io_utils import ROOT, load_config, ensure_dir, write_json
from run_pipeline import _phase4_cached_factory, _identifiability_classes
from storage_inversion import decode_fields


def circular_width(low, high, period=365.2425):
    delta = (np.asarray(high, float) - np.asarray(low, float)) % float(period)
    return np.minimum(delta, float(period) - delta)


def _draws(config, output):
    post = np.load(output / "posterior_coefficients.npz")
    rng = np.random.default_rng(config["project"]["random_seed"] + 17)
    n = int(config["uncertainty"]["posterior_draws"])
    return post["mean"], post["covariance"], rng.multivariate_normal(post["mean"], post["covariance"], size=n)


def _screen_mask(draws, diagnostics, config, factory):
    from run_pipeline import _screen_draws
    mask, summary = _screen_draws(draws, diagnostics.get("model_variant", "two_aquifer"), factory,
                                  config["uncertainty"].get("physical_screening", {}),
                                  config["temporal"]["annual_period_days"])
    return mask, summary


def write_revision_products(config_path="config.yaml", out_dir=None):
    config = load_config(config_path)
    output = ROOT / config["project"]["output_dir"]
    target = ensure_dir(out_dir or output / "revision_tmp")
    diagnostics = json.loads((output / "map_diagnostics.json").read_text(encoding="utf-8"))
    cache = Path(diagnostics["cache_path"])
    factory = _phase4_cached_factory(cache)
    mean, cov, draws = _draws(config, output)
    mask, summary = _screen_mask(draws, diagnostics, config, factory)
    write_json(summary, target / "posterior_draw_screening_summary.json")
    period = config["temporal"]["annual_period_days"]
    prior_precision = np.asarray(diagnostics["prior_precision"], float)
    prior_cov = np.linalg.pinv(prior_precision)
    model_variant = diagnostics.get("model_variant", "two_aquifer")
    geology = output / "geological_model_covariates.tif"
    names = ["Ske_posterior_median_screened.tif", "Ske_ci95_low_screened.tif", "Ske_ci95_high_screened.tif",
             "Ske_ci95_width_screened.tif", "Ske_relative_ci95_width_screened.tif", "logSke_posterior_std.tif",
             "Cu_posterior_median_screened.tif", "Cu_ci95_low_screened.tif", "Cu_ci95_high_screened.tif",
             "Cu_ci95_width_screened.tif", "Cu_relative_ci95_width_screened.tif",
             "lag_c_ci95_width_days.tif", "lag_u_ci95_width_days.tif"]
    with rasterio.open(geology) as src:
        profile = src.profile.copy(); profile.update(count=1, dtype="float32", nodata=np.nan)
        handles = [(target / name, rasterio.open((target / name).with_suffix(".tif.tmp"), "w", **profile)) for name in names]
        try:
            for _, dst in handles:
                dst.write(np.full(src.shape, np.nan, "float32"), 1)
            for block_i, (_, _, _, Z0, _, block_id) in enumerate(factory(), 1):
                print(f"revision_products_block {block_i}", flush=True)
                r, c, h, w = block_id["row"], block_id["col"], block_id["height"], block_id["width"]
                flat = np.asarray(block_id["flat_index"], int)
                arrays = [np.full(h * w, np.nan, "float32") for _ in names]
                for start in range(0, len(Z0), 5000):
                    end = min(len(Z0), start + 5000)
                    design = np.column_stack([np.ones(end - start), Z0[start:end]])
                    decoded = [decode_fields(draw, design, model_variant) for draw in draws[mask]]
                    ske = np.asarray([d[0] for d in decoded])
                    lagc = np.asarray([d[1] for d in decoded])
                    cu = np.asarray([d[2] for d in decoded])
                    lagu = np.asarray([d[3] for d in decoded])
                    log_ske = np.log(np.clip(ske, 1e-300, None))
                    ske_q = np.nanquantile(ske, [.025, .5, .975], axis=0)
                    cu_q = np.nanquantile(cu, [.025, .5, .975], axis=0)
                    lagc_low, lagc_high = np.nanquantile(lagc, [.025, .975], axis=0)
                    lagu_low, lagu_high = np.nanquantile(lagu, [.025, .975], axis=0)
                    ske_width = ske_q[2] - ske_q[0]
                    cu_width = cu_q[2] - cu_q[0]
                    values = [
                        ske_q[1], ske_q[0], ske_q[2], ske_width,
                        ske_width / np.maximum(2 * np.abs(ske_q[1]), 1e-12),
                        np.nanstd(log_ske, axis=0),
                        cu_q[1], cu_q[0], cu_q[2], cu_width,
                        cu_width / np.maximum(2 * np.abs(cu_q[1]), 1e-12),
                        circular_width(lagc_low, lagc_high, period),
                        circular_width(lagu_low, lagu_high, period),
                    ]
                    for arr, value in zip(arrays, values):
                        arr[flat[start:end]] = np.asarray(value, "float32")
                for arr, (_, dst) in zip(arrays, handles):
                    dst.write(arr.reshape(h, w), 1, window=Window(c, r, w, h))
        finally:
            for _, dst in handles:
                dst.close()
        for path, _ in handles:
            path.with_suffix(".tif.tmp").replace(path)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    print(write_revision_products(args.config, args.out_dir))

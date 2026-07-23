"""Phase-gated entry point for the Hengshui global inversion project."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import glob
import time
import shutil
from pathlib import Path

import pandas as pd
import numpy as np

from audit import require_audit, run_phase1_audit
from groundwater_processing import add_fixed_head_anomaly, apply_quality_flags, build_weighted_daily_series, well_summary
from insar_processing import compute_reference_series, write_vertical_h5
from insar_processing import GeoTiffCube, sample_points
from io_utils import ROOT, ensure_dir, file_fingerprint, load_config, resolve_config_path, resolve_path, runtime_provenance, write_json, write_table
from lag_analysis import infer_lag
from latent_head_model import fit_aquifer_models
from temporal_analysis import decompose_groups, fit_linear_annual_harmonic
from geological_prior import rasterize_geology
from storage_inversion import harmonic_map_inversion_streaming, decode_fields, rotate_coefficients
from uncertainty import harmonic_streaming_gauss_newton
from bulletin_processing import prepare_bulletin_constraints


def _vertical_h5_path(config,output,manifest=None,allow_legacy=True):
    cache_dir=ensure_dir(Path(output)/"cache")
    source={"config_sha256":config.get("_config_sha256")}
    if manifest is not None:
        source.update({k:manifest.get(k) for k in ("cube_shape","n_epochs","time_start","time_end","crs","transform","reference_frame_id")})
    digest=hashlib.sha256(json.dumps(source,sort_keys=True,default=str).encode("utf-8")).hexdigest()[:16]
    path=cache_dir/f"vertical_timeseries_{digest}.h5"
    legacy=Path(output)/"vertical_timeseries.h5"
    return path if path.exists() or not (allow_legacy and legacy.exists()) else legacy


def run_phase1(config):
    output = ensure_dir(ROOT / config["project"]["output_dir"])
    audit_report, cube, groundwater = run_phase1_audit(config)
    write_json(audit_report, output / "audit_report.json")
    write_table(pd.DataFrame(audit_report["gates"]), output / "audit_gates.csv")
    require_audit(audit_report)
    prepare_bulletin_constraints(resolve_config_path(config,config["bulletin"]["file"]),resolve_config_path(config,config["bulletin"]["verified_file"]),output)

    groundwater_config = config["groundwater"]
    groundwater = apply_quality_flags(
        groundwater,
        spike_threshold_m_per_day=groundwater_config["spike_threshold_m_per_day"],
        plateau_min_days=groundwater_config["plateau_min_days"],
        plateau_tolerance_m=groundwater_config["plateau_tolerance_m"],
        minimum_allowed_water_depth_m=groundwater_config["minimum_allowed_water_depth_m"],
    )
    groundwater = build_weighted_daily_series(groundwater,groundwater_config["max_interpolated_gap_days"],
        groundwater_config["observed_weight"],groundwater_config["interpolated_weight"],
        groundwater_config["suspicious_weight"],groundwater_config["invalid_weight"])
    groundwater = add_fixed_head_anomaly(
        groundwater,
        groundwater_config["baseline_start"],
        groundwater_config["baseline_end"],
    )
    contract_columns = [
        "well_id", "lon", "lat", "well_depth_m", "ground_elevation_m",
        "date", "water_depth_m", "hydraulic_head_m", "head_anomaly_m",
        "aquifer_type", "quality_flag", "is_valid_for_model","is_observed","is_interpolated","observation_weight",
        "sensor_reset_flag","hampel_outlier_flag","neighbor_inconsistency_flag",
        "head_baseline_m", "baseline_n", "baseline_sufficient",
    ]
    write_table(groundwater[contract_columns], output / "well_timeseries.csv")
    write_table(well_summary(groundwater), output / "well_summary.csv")

    reference_config = config["reference"]
    reference_series = None
    if reference_config.get("enabled", False):
        if reference_config.get("lon") is None or reference_config.get("lat") is None:
            raise ValueError("Reference is enabled but lon/lat is not configured")
        reference_series, reference_metadata = compute_reference_series(
            cube,
            reference_config["lon"],
            reference_config["lat"],
            reference_config["radius_m"],
            reference_config["method"],
            reference_config["min_valid_pixels"],
            reference_config["minimum_valid_epoch_fraction"],
        )
        reference_table = reference_series.rename_axis("date").reset_index()
        reference_table = pd.concat([reference_table,
            pd.DataFrame(reference_metadata.pop("epoch_diagnostics"))], axis=1)
        write_table(reference_table, output / "reference_series.csv")
    else:
        reference_metadata = {
            "reference_frame_id": "source_reference_unchanged",
            "reference_lon": None,
            "reference_lat": None,
            "reference_radius_m": None,
            "reference_method": "disabled",
        }
    cube_manifest=cube.manifest(reference_metadata)
    write_json(cube_manifest, output / "insar_cube_manifest.json")
    write_table(cube.epochs, output / "insar_epochs.csv")

    insar_config = config["insar"]
    if insar_config.get("materialize_vertical_h5", False):
        write_vertical_h5(
            cube,
            resolve_config_path(config,insar_config["incidence_grid"]),
            _vertical_h5_path(config,output,cube_manifest,allow_legacy=False),
            reference_series_mm=reference_series.to_numpy() if reference_series is not None else None,
            reference_metadata=reference_metadata,
            block_rows=insar_config["h5_chunk_rows"],
            block_cols=insar_config["h5_chunk_cols"],
        )
    phase_status = {
        "phase_1": "complete",
        "phase_2": "not_run",
        "phase_3": "blocked_until_phase_2_validation",
        "phase_4": "blocked_until_latent_head_validation",
        "phase_5": "blocked_until_identifiable_map_solution",
        "synthetic_or_placeholder_results_generated": False,
    }
    write_json(phase_status, output / "phase_status.json")
    return phase_status


def _require_previous(output, phase):
    path = output / "phase_status.json"
    if not path.exists():
        raise RuntimeError("Run Phase 1 first; phase_status.json is missing")
    import json
    status = json.loads(path.read_text(encoding="utf-8"))
    if status.get(f"phase_{phase-1}") != "complete":
        raise RuntimeError(f"Phase {phase} blocked until Phase {phase-1} is complete")
    return status


def _reference_for_cube(config, output, cube):
    rc = config["reference"]
    if not rc.get("enabled", False):
        return None
    series, _ = compute_reference_series(cube, rc["lon"], rc["lat"], rc["radius_m"], rc["method"],
                                         rc["min_valid_pixels"], rc["minimum_valid_epoch_fraction"])
    return series.to_numpy()


def run_phase2(config):
    output = ensure_dir(ROOT / config["project"]["output_dir"])
    status = _require_previous(output, 2)
    pc = config["phase2"]
    wells = pd.read_csv(output / "well_timeseries.csv", parse_dates=["date"])
    valid_mask=wells["is_valid_for_model"].astype(str).str.lower().isin(["true", "1"])
    if "baseline_sufficient" in wells:
        valid_mask &= wells["baseline_sufficient"].astype(str).str.lower().isin(["true","1"])
    valid = wells[valid_mask]
    temporal=config["temporal"];lag_config=config["lag"]
    write_table(decompose_groups(valid,"well_id","date","head_anomaly_m",harmonic_origin=temporal["harmonic_origin"],
        annual_period_days=temporal["annual_period_days"],include_semiannual=temporal["include_semiannual"],
        min_observations=lag_config["minimum_insar_pairs"]),output/"well_harmonic_decomposition.csv")
    metadata = wells.drop_duplicates("well_id").sort_values("well_id")
    cube = GeoTiffCube.from_glob(resolve_config_path(config,config["insar"]["geotiff_glob"]), config["insar"]["displacement_unit"])
    sampled_path=output/"insar_at_wells.csv"
    if sampled_path.exists():
        sampled=pd.read_csv(sampled_path,parse_dates=["date"])
    else:
        sampled = sample_points(cube, metadata[["lon", "lat"]].to_numpy(),
                                resolve_config_path(config,config["insar"]["incidence_grid"]),
                                _reference_for_cube(config, output, cube),
                                pc["insar_well_sampling_method"], pc["insar_well_buffer_radius_m"])
        sampled["well_id"] = sampled["point_index"].map(dict(enumerate(metadata["well_id"])))
        write_table(sampled, sampled_path)
    rows, spectra = [], []
    import json
    reference_manifest=json.loads((output/"insar_cube_manifest.json").read_text(encoding="utf-8"))
    grouped=list(valid.groupby("well_id"))
    for lag_index,(well_id, gw) in enumerate(grouped,1):
        print(f"lag_well {lag_index}/{len(grouped)} {well_id}",flush=True)
        ins = sampled[sampled["well_id"] == well_id]
        if gw["observation_weight"].gt(0).sum()<lag_config["minimum_insar_pairs"] or ins["vertical_mm"].notna().sum()<lag_config["minimum_insar_pairs"]:
            continue
        reference_metadata = {k: v for k, v in reference_manifest.items() if k.startswith("reference_")}
        spectrum, summary = infer_lag(gw["date"], gw["head_anomaly_m"], ins["date"], ins["vertical_mm"],
            groundwater_weights=gw["observation_weight"],harmonic_origin=temporal["harmonic_origin"],annual_period_days=temporal["annual_period_days"],
            minimum_days=lag_config["minimum_days"],maximum_days=lag_config["maximum_days"],coarse_step_days=lag_config["coarse_step_days"],
            fine_step_days=lag_config["fine_step_days"],fine_half_width_days=lag_config["fine_half_width_days"],
            minimum_insar_pairs=lag_config["minimum_insar_pairs"],bootstrap_replicates=lag_config["bootstrap_replicates"],
            surrogate_replicates=lag_config["surrogate_replicates"],bootstrap_block_days=lag_config["bootstrap_block_days"],
            maximum_ci_width_days=lag_config["maximum_ci_width_days"],expected_correlation_sign=lag_config["expected_correlation_sign"],
            minimum_annual_snr=lag_config.get("minimum_annual_snr",1.0),maximum_phase_std_days=lag_config.get("maximum_phase_std_days",45),
            aquifer_type=str(gw["aquifer_type"].iloc[0]), reference_metadata=reference_metadata)
        rows.append({"well_id": well_id, **summary})
        spectrum.insert(0, "well_id", well_id); spectra.append(spectrum)
    lag_summary = pd.DataFrame(rows)
    if lag_summary.empty:
        raise RuntimeError("Phase 2 validation failed: no well produced a lag estimate")
    write_table(lag_summary, output / "lag_summary.csv")
    write_table(pd.concat(spectra, ignore_index=True), output / "tlcc_spectra.csv")
    status["phase_2"]="complete"
    if status.get("phase_3")!="complete":
        status["phase_3"]="not_run"
    write_json(status, output / "phase_status.json")
    return status


def run_phase3(config):
    output = ensure_dir(ROOT / config["project"]["output_dir"])
    status = _require_previous(output, 3)
    wells = pd.read_csv(output / "well_timeseries.csv", parse_dates=["date"])
    pc = config["phase3"]
    fitted = fit_aquifer_models(wells, ranks=pc["candidate_ranks"],
                                holdout_fraction=pc["holdout_fraction"],
                                random_seed=config["project"]["random_seed"], ridge=pc["ridge"],
                                repeats=pc["validation_repeats"], spatial_block_size_m=pc["spatial_block_size_m"],
                                projected_crs=pc["projected_crs"], validation_schemes=pc.get("validation_schemes",("random","spatial_block")))
    validation = []
    models = {}
    for aquifer, result in fitted.items():
        table = result["metrics"].copy(); table.insert(0, "aquifer_type", aquifer)
        validation.append(table); models[aquifer] = result["model"]
    validation = pd.concat(validation, ignore_index=True)
    gate_schemes=[s for s in ("spatial_block","temporal_block") if s in set(validation["scheme"])]
    gate = validation[validation["scheme"].isin(gate_schemes)]
    best = (gate.groupby(["aquifer_type", "rank"], as_index=False)["RMSE"].mean()
            .sort_values("RMSE").groupby("aquifer_type").head(1))
    write_table(validation, output / "latent_head_leave_well_out_validation.csv")
    passed = bool(best["RMSE"].notna().all() and (best["RMSE"] <= pc["validation_max_rmse_m"]).all())
    if not passed:
        raise RuntimeError("Phase 3 validation gate failed; MAP inversion remains blocked")
    with (output / "latent_head_models.pkl").open("wb") as stream:
        pickle.dump(models, stream)
    status["phase_3"]="complete"
    if status.get("phase_4")!="complete":
        status["phase_4"]="ready_requires_full_pixel_design"
    write_json(status, output / "phase_status.json")
    return status


def _phase4_priors(n_covariates,variant,prior):
    p=n_covariates+1;groups=2 if variant=="confined_only" else 4
    mean=np.full(groups*p,float(prior["coefficient_mean"]));std=np.full(groups*p,float(prior["coefficient_std"]))
    mean[0]=prior["log_ske_intercept_mean"];mean[p]=prior["lag_logit_intercept_mean"]
    if groups==4:mean[2*p]=np.log(np.expm1(prior["cu_prior_m_per_m"]));mean[3*p]=prior["lag_logit_intercept_mean"]
    for g in range(groups):std[g*p]=prior["intercept_std"]
    return mean,std


def _harmonic_coefficients(dates,values,origin,period_days):
    values=np.asarray(values,float)
    output=np.full((values.shape[0],2),np.nan,float);covariances=np.full((values.shape[0],2,2),np.nan,float)
    t=(pd.DatetimeIndex(pd.to_datetime(dates))-pd.Timestamp(origin)).days.to_numpy(float)
    X=np.column_stack([np.ones(len(t)),t/365.2425,np.sin(2*np.pi*t/period_days),np.cos(2*np.pi*t/period_days)])
    finite=np.isfinite(values);nobs=finite.sum(axis=1);good=nobs>=24
    if good.any():
        Y=np.where(finite[good],values[good],0.0);W=finite[good].astype(float)
        xtwx=np.einsum("ti,nt,tj->nij",X,W,X,optimize=True)
        xtwy=np.einsum("ti,nt->ni",X,Y,optimize=True)
        xtwx+=np.eye(X.shape[1])[None,:,:]*1e-8
        beta=np.linalg.solve(xtwx,xtwy[...,None])[...,0]
        output[good]=beta[:,2:4]
        fitted=np.einsum("ti,ni->nt",X,beta,optimize=True)
        resid=np.where(finite[good],values[good]-fitted,0.0)
        dof=np.maximum(1,nobs[good]-X.shape[1])
        sigma2=np.sum(W*resid*resid,axis=1)/dof
        inv=np.linalg.pinv(xtwx)
        covariances[good]=inv[:,2:4,2:4]*sigma2[:,None,None]
    return output,covariances


def _spatial_basis_builder(src,spatial_config,projected_crs="EPSG:32650"):
    if not spatial_config or spatial_config.get("type")!="rbf":
        return None,[],lambda coords: np.empty((len(coords),0),float)
    from pyproj import Transformer
    left,bottom,right,top=src.bounds
    transformer=Transformer.from_crs(src.crs or "EPSG:4326",projected_crs,always_xy=True)
    xs,ys=transformer.transform([left,right,left,right],[bottom,bottom,top,top])
    spacing_km=float(spatial_config.get("candidate_spacing_km",[10])[0]);spacing=max(1000,spacing_km*1000)
    gx=np.arange(min(xs),max(xs)+spacing,spacing);gy=np.arange(min(ys),max(ys)+spacing,spacing)
    centers=np.array([(x,y) for y in gy for x in gx],float)
    if len(centers)>int(spatial_config.get("maximum_centers",64)):
        idx=np.linspace(0,len(centers)-1,int(spatial_config.get("maximum_centers",64))).round().astype(int)
        centers=centers[np.unique(idx)]
    scale=max(spacing,1.0)
    to_projected=Transformer.from_crs(src.crs or "EPSG:4326",projected_crs,always_xy=True)
    names=[f"spatial_rbf_{i:02d}" for i in range(len(centers))]
    def basis(source_crs_coords):
        px,py=to_projected.transform(source_crs_coords[:,0],source_crs_coords[:,1])
        points=np.column_stack([px,py])
        if len(centers)==0:return np.empty((len(points),0),float)
        diff=points[:,None,:]-centers[None,:,:]
        return np.exp(-0.5*np.sum(diff*diff,axis=2)/(scale*scale))
    metadata={"centers":centers.tolist(),"scale_m":scale,"projected_crs":projected_crs,"names":names}
    return metadata,names,basis


def _build_phase4_harmonic_factory(vertical_h5,geology_tif,models,block_rows,block_cols,temporal_config,spatial_config=None,projected_crs="EPSG:32650",observation_sigma_mm=5.0):
    import h5py,rasterio
    from rasterio.transform import xy
    with h5py.File(vertical_h5,"r") as h5:
        insar_dates=pd.to_datetime([x.decode() for x in h5["date"]])
    common=pd.DatetimeIndex(insar_dates)
    for model in models.values():
        common=common[(common>=model.dates.min())&(common<=model.dates.max())]
    if len(common)<24:raise RuntimeError("Phase 4 requires at least 24 real SAR epochs covered by latent head models")
    time_index=np.array([np.where(insar_dates==date)[0][0] for date in common])
    origin=temporal_config.get("harmonic_origin","2018-01-01");period=temporal_config.get("annual_period_days",365.2425)
    design_metadata={}
    with rasterio.open(geology_tif) as geo:
        basis_metadata,basis_names,_=_spatial_basis_builder(geo,spatial_config,projected_crs)
        design_metadata={"n_geology":int(geo.count),"n_spatial_basis":len(basis_names),"spatial_basis":basis_metadata,"geology_names":[f"geology_{i}" for i in range(int(geo.count))],"spatial_basis_names":basis_names}
    def factory():
        with h5py.File(vertical_h5,"r") as h5,rasterio.open(geology_tif) as geo:
            from spatial_utils import iter_windows
            _,_,basis_fn=_spatial_basis_builder(geo,spatial_config,projected_crs)
            for window in iter_windows(geo.height,geo.width,block_rows,block_cols):
                r,c=int(window.row_off),int(window.col_off);height,width=int(window.height),int(window.width)
                u=np.asarray(h5["vertical_displacement_mm"][time_index,r:r+height,c:c+width]).reshape(len(common),-1).T
                z=geo.read(window=window).reshape(geo.count,-1).T
                rr,cc=np.indices((height,width));xs,ys=xy(geo.transform,rr+r,cc+c,offset="center")
                coords_source=np.column_stack([np.ravel(xs),np.ravel(ys)])
                coords_lonlat=coords_source
                if str(geo.crs)!="EPSG:4326":
                    from pyproj import Transformer
                    transformer=Transformer.from_crs(geo.crs,"EPSG:4326",always_xy=True)
                    lon,lat=transformer.transform(coords_source[:,0],coords_source[:,1]);coords_lonlat=np.column_stack([lon,lat])
                valid=np.isfinite(z).all(axis=1)&(np.isfinite(u).sum(axis=1)>=24)
                if not valid.any():continue
                hc=models["confined"].predict(coords_lonlat[valid],common);hu=models["unconfined"].predict(coords_lonlat[valid],common)
                obs_coeff,obs_cov=_harmonic_coefficients(common,u[valid],origin,period)
                hc_coeff,_=_harmonic_coefficients(common,hc,origin,period)
                hu_coeff,_=_harmonic_coefficients(common,hu,origin,period)
                good=np.isfinite(obs_coeff).all(1)&np.isfinite(hc_coeff).all(1)&np.isfinite(hu_coeff).all(1)
                if not good.any():continue
                obs_var=np.nanmean(np.diagonal(obs_cov,axis1=1,axis2=2),axis=1)
                obs_var=np.where(np.isfinite(obs_var)&(obs_var>0),obs_var,float(observation_sigma_mm)**2)
                weights=1/np.maximum(obs_var[good],float(observation_sigma_mm)**2)
                block_valid=np.flatnonzero(valid)[good]
                z_design=np.column_stack([z[valid][good],basis_fn(coords_source[valid][good])])
                yield obs_coeff[good],hc_coeff[good],hu_coeff[good],z_design,weights,{"row":r,"col":c,"height":height,"width":width,"flat_index":block_valid}
    return factory,common,design_metadata


def _cache_phase4_harmonic_blocks(source_factory,cache_path,n_covariates,cache_key=None,design_metadata=None):
    import h5py
    cache_path=Path(cache_path)
    def cache_is_complete(path):
        if not path.exists():
            return False
        with h5py.File(path,"r") as h5:
            key_ok=(cache_key is None or h5.attrs.get("cache_key","")==cache_key)
            return h5.attrs.get("complete",0)==1 and h5.attrs.get("n_covariates")==n_covariates and key_ok
    if cache_path.exists():
        if cache_is_complete(cache_path):
            return _phase4_cached_factory(cache_path)
    ensure_dir(cache_path.parent)
    tmp=cache_path.with_suffix(cache_path.suffix+".tmp")
    if tmp.exists():tmp.unlink()
    starts=[];counts=[];rows=[];cols=[];heights=[];widths=[];total=0;total_weight=0.0
    with h5py.File(tmp,"w") as h5:
        h5.attrs["complete"]=0;h5.attrs["n_covariates"]=n_covariates
        if cache_key is not None:h5.attrs["cache_key"]=cache_key
        if design_metadata is not None:h5.attrs["design_metadata"]=json.dumps(design_metadata,ensure_ascii=False)
        datasets={
            "obs":h5.create_dataset("obs",shape=(0,2),maxshape=(None,2),chunks=(65536,2),dtype="float32",compression="lzf"),
            "hc":h5.create_dataset("hc",shape=(0,2),maxshape=(None,2),chunks=(65536,2),dtype="float32",compression="lzf"),
            "hu":h5.create_dataset("hu",shape=(0,2),maxshape=(None,2),chunks=(65536,2),dtype="float32",compression="lzf"),
            "z":h5.create_dataset("z",shape=(0,n_covariates),maxshape=(None,n_covariates),chunks=(65536,n_covariates),dtype="float32",compression="lzf"),
            "weights":h5.create_dataset("weights",shape=(0,),maxshape=(None,),chunks=(65536,),dtype="float32",compression="lzf"),
            "flat_index":h5.create_dataset("flat_index",shape=(0,),maxshape=(None,),chunks=(65536,),dtype="uint32",compression="lzf"),
        }
        def append(name,values):
            ds=datasets[name];values=np.asarray(values,dtype=ds.dtype)
            old=ds.shape[0];new=old+len(values);ds.resize((new,)+ds.shape[1:]);ds[old:new]=values
        for block_i,(obs,hc,hu,Z0,weights,block_id) in enumerate(source_factory(),1):
            n=len(obs)
            if n==0:continue
            print(f"cache_phase4_block {block_i} pixels {n}",flush=True)
            starts.append(total);counts.append(n);rows.append(block_id["row"]);cols.append(block_id["col"])
            heights.append(block_id["height"]);widths.append(block_id["width"])
            append("obs",obs);append("hc",hc);append("hu",hu);append("z",Z0);append("weights",weights);append("flat_index",block_id["flat_index"])
            total+=n;total_weight+=float(np.asarray(weights,float).sum())
        h5.create_dataset("block_start",data=np.asarray(starts,dtype="uint64"))
        h5.create_dataset("block_count",data=np.asarray(counts,dtype="uint32"))
        h5.create_dataset("block_row",data=np.asarray(rows,dtype="uint32"))
        h5.create_dataset("block_col",data=np.asarray(cols,dtype="uint32"))
        h5.create_dataset("block_height",data=np.asarray(heights,dtype="uint32"))
        h5.create_dataset("block_width",data=np.asarray(widths,dtype="uint32"))
        h5.attrs["n_blocks"]=len(starts);h5.attrs["n_pixels"]=total;h5.attrs["total_weight"]=total_weight;h5.attrs["complete"]=1
    try:
        tmp.replace(cache_path)
    except FileNotFoundError:
        if not cache_is_complete(cache_path):
            raise
    return _phase4_cached_factory(cache_path)


def _phase4_cache_key(config,vertical_h5,geology_tif,models_path,manifest_path,design_metadata):
    def stamp(path):
        path=Path(path)
        if not path.exists():return None
        stat=path.stat()
        return {"path":str(path.resolve()),"size_bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns}
    parts={
        "config_sha256":config.get("_config_sha256"),
        "vertical_h5":stamp(vertical_h5),
        "geology":file_fingerprint(geology_tif)["sha256"] if Path(geology_tif).exists() else None,
        "geology_metadata":file_fingerprint(Path(geology_tif).parent/"geological_covariate_metadata.json")["sha256"] if (Path(geology_tif).parent/"geological_covariate_metadata.json").exists() else None,
        "geology_raw_stack":file_fingerprint(ROOT/config["geology"]["raw_rasters"]["stack"])["sha256"] if config.get("geology",{}).get("raw_rasters",{}).get("stack") and (ROOT/config["geology"]["raw_rasters"]["stack"]).exists() else None,
        "latent_models":file_fingerprint(models_path)["sha256"] if Path(models_path).exists() else None,
        "manifest":file_fingerprint(manifest_path)["sha256"] if Path(manifest_path).exists() else None,
        "harmonic_origin":config["temporal"].get("harmonic_origin"),
        "annual_period_days":config["temporal"].get("annual_period_days"),
        "design_metadata":design_metadata,
    }
    return hashlib.sha256(json.dumps(parts,sort_keys=True,default=str).encode("utf-8")).hexdigest()


def _phase4_cached_factory(cache_path):
    import h5py
    cache_path=Path(cache_path)
    def factory():
        with h5py.File(cache_path,"r") as h5:
            starts=h5["block_start"][:];counts=h5["block_count"][:]
            rows=h5["block_row"][:];cols=h5["block_col"][:];heights=h5["block_height"][:];widths=h5["block_width"][:]
            total_weight=float(h5.attrs.get("total_weight",1.0)) or 1.0
            for i,(start,count) in enumerate(zip(starts,counts),1):
                start=int(start);end=start+int(count)
                block_id={"row":int(rows[i-1]),"col":int(cols[i-1]),"height":int(heights[i-1]),"width":int(widths[i-1]),"flat_index":h5["flat_index"][start:end]}
                yield h5["obs"][start:end],h5["hc"][start:end],h5["hu"][start:end],h5["z"][start:end],h5["weights"][start:end]/total_weight,block_id
    return factory


def _write_phase4_maps(result,geology_tif,output):
    import rasterio
    from rasterio.windows import Window
    outputs=[("Ske_MAP.tif",0),("lag_c_MAP_days.tif",1),("Cu_MAP.tif",2),("lag_u_MAP_days.tif",3)]
    with rasterio.open(geology_tif) as src:
        profile=src.profile.copy();profile.update(count=1,dtype="float32",nodata=np.nan)
        handles=[]
        try:
            for name,_ in outputs+[("residual_rmse_mm.tif",4),("geological_contribution.tif",5),("spatial_basis_contribution.tif",6)]:
                path=output/name;tmp=path.with_suffix(path.suffix+".tmp")
                handles.append((path,tmp,rasterio.open(tmp,"w",**profile)))
            variant_profile=profile.copy();variant_profile.update(dtype="uint8",nodata=0)
            variant_path=output/"model_variant.tif";variant_tmp=variant_path.with_suffix(".tif.tmp")
            variant_dst=rasterio.open(variant_tmp,"w",**variant_profile)
            for _,_,dst in handles:dst.write(np.full(src.shape,np.nan,"float32"),1)
            variant_dst.write(np.zeros(src.shape,"uint8"),1)
            for block_id,fields,residual in result["field_blocks"]:
                r,c,h,w=block_id["row"],block_id["col"],block_id["height"],block_id["width"]
                flat=np.asarray(block_id["flat_index"],int);window=Window(c,r,w,h)
                for handle_index,(path,tmp,dst) in enumerate(handles):
                    arr=np.full(h*w,np.nan,"float32")
                    if handle_index<4:
                        arr[flat]=np.asarray(fields[handle_index],"float32")
                    elif handle_index==4:
                        arr[flat]=np.asarray(residual,"float32")
                    else:
                        arr[flat]=0.
                    dst.write(arr.reshape(h,w),1,window=window)
                var=np.zeros(h*w,"uint8");var[flat]=1 if result["model_variant"]=="confined_only" else 2
                variant_dst.write(var.reshape(h,w),1,window=window)
        finally:
            for _,_,dst in handles:dst.close()
            variant_dst.close()
        for path,tmp,_ in handles:
            tmp.replace(path)
        variant_tmp.replace(variant_path)


def _write_phase4_maps_streaming(result,factory,geology_tif,output,period_days=365.2425):
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(geology_tif) as src:
        profile=src.profile.copy();profile.update(count=1,dtype="float32",nodata=np.nan)
        names=["Ske_MAP.tif","lag_c_MAP_days.tif","Cu_MAP.tif","lag_u_MAP_days.tif","residual_rmse_mm.tif","geological_contribution.tif","spatial_basis_contribution.tif"]
        handles=[]
        try:
            for name in names:
                path=output/name;tmp=path.with_suffix(path.suffix+".tmp");handles.append((path,tmp,rasterio.open(tmp,"w",**profile)))
            variant_profile=profile.copy();variant_profile.update(dtype="uint8",nodata=0)
            variant_path=output/"model_variant.tif";variant_tmp=variant_path.with_suffix(".tif.tmp")
            variant_dst=rasterio.open(variant_tmp,"w",**variant_profile)
            for _,_,dst in handles:dst.write(np.full(src.shape,np.nan,"float32"),1)
            variant_dst.write(np.zeros(src.shape,"uint8"),1)
            for block_i,(obs,hc,hu,Z0,weights,block_id) in enumerate(factory(),1):
                print(f"write_phase4_map_block {block_i}",flush=True)
                design=np.column_stack([np.ones(len(Z0)),Z0])
                fields=decode_fields(result["coefficients"],design,result["model_variant"])
                pred=fields[0][:,None]*rotate_coefficients(hc,fields[1],period_days)
                if result["model_variant"]=="two_aquifer":
                    pred+=fields[2][:,None]*rotate_coefficients(hu,fields[3],period_days)
                residual=np.sqrt(np.mean((obs-1000*pred)**2,axis=1))
                n_geo=int(result.get("design_metadata",{}).get("n_geology",Z0.shape[1]))
                p=Z0.shape[1]+1
                log_ske_beta=result["coefficients"][:p]
                geo_contrib=Z0[:,:n_geo]@log_ske_beta[1:1+n_geo] if n_geo else np.zeros(len(Z0))
                spatial_contrib=Z0[:,n_geo:]@log_ske_beta[1+n_geo:p] if Z0.shape[1]>n_geo else np.zeros(len(Z0))
                r,c,h,w=block_id["row"],block_id["col"],block_id["height"],block_id["width"]
                flat=np.asarray(block_id["flat_index"],int);window=Window(c,r,w,h)
                for handle_index,(_,_,dst) in enumerate(handles):
                    arr=np.full(h*w,np.nan,"float32")
                    if handle_index<4:arr[flat]=np.asarray(fields[handle_index],"float32")
                    elif handle_index==4:arr[flat]=np.asarray(residual,"float32")
                    elif handle_index==5:arr[flat]=np.asarray(geo_contrib,"float32")
                    else:arr[flat]=np.asarray(spatial_contrib,"float32")
                    dst.write(arr.reshape(h,w),1,window=window)
                var=np.zeros(h*w,"uint8");var[flat]=1 if result["model_variant"]=="confined_only" else 2
                variant_dst.write(var.reshape(h,w),1,window=window)
        finally:
            for _,_,dst in handles:dst.close()
            variant_dst.close()
        for path,tmp,_ in handles:tmp.replace(path)
        variant_tmp.replace(variant_path)


def _eta_variance_reduction(design,covariance,prior_cov,start,p):
    post_var=np.einsum("ni,ij,nj->n",design,covariance[start:start+p,start:start+p],design,optimize=True)
    prior_var=np.einsum("ni,ij,nj->n",design,prior_cov[start:start+p,start:start+p],design,optimize=True)
    return 1-np.divide(post_var,prior_var,out=np.full_like(post_var,np.nan),where=prior_var>0)


def _class_from_reduction_and_cv(reduction,cv,strong_reduction=.5,weak_reduction=.1,strong_cv=.5,weak_cv=1.0):
    return np.select([np.isfinite(reduction)&(reduction>=strong_reduction)&(cv<=strong_cv),
                      np.isfinite(reduction)&(reduction>=weak_reduction)&(cv<=weak_cv),
                      np.isfinite(reduction)&np.isfinite(cv)],
                     [3.0,2.0,1.0],default=0.0)


def _class_from_reduction_and_std(reduction,std,strong_reduction=.5,weak_reduction=.1,strong_std=30.0,weak_std=90.0):
    return np.select([np.isfinite(reduction)&(reduction>=strong_reduction)&(std<=strong_std),
                      np.isfinite(reduction)&(reduction>=weak_reduction)&(std<=weak_std),
                      np.isfinite(reduction)&np.isfinite(std)],
                     [3.0,2.0,1.0],default=0.0)


def _identifiability_classes(design,covariance,prior_cov,ske,lagc,cu,lagu,model_variant):
    p=design.shape[1];cursor=0
    ske_reduction=_eta_variance_reduction(design,covariance,prior_cov,cursor,p);cursor+=p
    ske_mean=np.nanmean(ske,axis=0);ske_cv=np.divide(np.nanstd(ske,axis=0),np.abs(ske_mean),out=np.full(design.shape[0],np.nan),where=np.abs(ske_mean)>0)
    ske_class=_class_from_reduction_and_cv(ske_reduction,ske_cv)
    lagc_reduction=_eta_variance_reduction(design,covariance,prior_cov,cursor,p);cursor+=p
    lagc_class=_class_from_reduction_and_std(lagc_reduction,np.nanstd(lagc,axis=0))
    if model_variant=="two_aquifer":
        cu_reduction=_eta_variance_reduction(design,covariance,prior_cov,cursor,p);cursor+=p
        cu_mean=np.nanmean(cu,axis=0);cu_cv=np.divide(np.nanstd(cu,axis=0),np.abs(cu_mean),out=np.full(design.shape[0],np.nan),where=np.abs(cu_mean)>0)
        cu_class=_class_from_reduction_and_cv(cu_reduction,cu_cv)
        lagu_reduction=_eta_variance_reduction(design,covariance,prior_cov,cursor,p)
        lagu_class=_class_from_reduction_and_std(lagu_reduction,np.nanstd(lagu,axis=0))
        combined=np.nanmin(np.vstack([ske_class,lagc_class,cu_class,lagu_class]),axis=0)
    else:
        cu_class=np.zeros(design.shape[0],float);lagu_class=np.zeros(design.shape[0],float)
        combined=np.nanmin(np.vstack([ske_class,lagc_class]),axis=0)
    return {"ske":ske_class,"lag_c":lagc_class,"cu":cu_class,"lag_u":lagu_class,"combined":combined}


def _write_phase5_posterior_maps(coefficients,covariance,prior_precision,model_variant,factory,geology_tif,output,design_metadata,n_draws,random_seed,period_days):
    import rasterio
    from rasterio.windows import Window
    rng=np.random.default_rng(random_seed)
    n_draws=int(max(500,n_draws))
    draws=rng.multivariate_normal(coefficients,covariance,size=n_draws)
    prior_cov=np.linalg.pinv(prior_precision)
    names=["Ske_posterior_mean.tif","Ske_posterior_std.tif","lag_c_ci95_low_days.tif","lag_c_ci95_high_days.tif",
           "Cu_posterior_mean.tif","Cu_posterior_std.tif","lag_u_ci95_low_days.tif","lag_u_ci95_high_days.tif",
           "Ske_identifiability.tif","lag_c_identifiability.tif","Cu_identifiability.tif","lag_u_identifiability.tif",
           "combined_deformation_identifiability.tif","combined_identifiability.tif","parameter_identifiability.tif",
           "confined_storage_identifiability.tif","confined_response_identifiability.tif",
           "unconfined_storage_identifiability.tif","storage_identifiability.tif"]
    with rasterio.open(geology_tif) as src:
        profile=src.profile.copy();profile.update(count=1,dtype="float32",nodata=np.nan)
        handles=[]
        try:
            for name in names:
                path=output/name;tmp=path.with_suffix(path.suffix+".tmp");handles.append((path,tmp,rasterio.open(tmp,"w",**profile)))
            for _,_,dst in handles:dst.write(np.full(src.shape,np.nan,"float32"),1)
            for block_i,(_,_,_,Z0,_,block_id) in enumerate(factory(),1):
                print(f"write_phase5_posterior_block {block_i}",flush=True)
                r,c,h,w=block_id["row"],block_id["col"],block_id["height"],block_id["width"]
                flat=np.asarray(block_id["flat_index"],int);window=Window(c,r,w,h)
                arrays=[np.full(h*w,np.nan,"float32") for _ in handles]
                for start in range(0,len(Z0),5000):
                    end=min(len(Z0),start+5000);design=np.column_stack([np.ones(end-start),Z0[start:end]])
                    decoded=[decode_fields(draw,design,model_variant) for draw in draws]
                    ske=np.asarray([d[0] for d in decoded]);lagc=np.asarray([d[1] for d in decoded])
                    cu=np.asarray([d[2] for d in decoded]);lagu=np.asarray([d[3] for d in decoded])
                    p=design.shape[1]
                    classes=_identifiability_classes(design,covariance,prior_cov,ske,lagc,cu,lagu,model_variant)
                    combined=classes["combined"]
                    confined_storage=classes["ske"]
                    confined_response=np.nanmin(np.vstack([classes["ske"],classes["lag_c"]]),axis=0)
                    unconfined_storage=np.where(np.isfinite(np.nanmean(lagu,axis=0)),1.0,0.0)
                    storage_ident=np.where((confined_storage>=2)&(unconfined_storage>=1),2.0,
                                           np.where((confined_storage>=1)|(unconfined_storage>=1),1.0,0.0))
                    values=[ske.mean(0),ske.std(0),np.quantile(lagc,.025,axis=0),np.quantile(lagc,.975,axis=0),
                            np.nanmean(cu,axis=0),np.nanstd(cu,axis=0),np.nanquantile(lagu,.025,axis=0),np.nanquantile(lagu,.975,axis=0),
                            classes["ske"],classes["lag_c"],classes["cu"],classes["lag_u"],combined,combined,combined,
                            confined_storage,confined_response,unconfined_storage,storage_ident]
                    for arr,value in zip(arrays,values):arr[flat[start:end]]=np.asarray(value,"float32")
                for arr,(_,_,dst) in zip(arrays,handles):dst.write(arr.reshape(h,w),1,window=window)
        finally:
            for _,_,dst in handles:dst.close()
        for path,tmp,_ in handles:tmp.replace(path)


def _pixel_area_m2_for_block(src,block_id):
    from rasterio.transform import xy
    r,c,h,w=block_id["row"],block_id["col"],block_id["height"],block_id["width"]
    rr,cc=np.indices((h,w))
    if src.crs and src.crs.is_projected:
        if abs(src.transform.b)>1e-12 or abs(src.transform.d)>1e-12:
            raise RuntimeError("Projected storage integration does not support rotated/sheared rasters")
        unit=getattr(src.crs,"linear_units",None)
        if unit and str(unit).lower() not in {"metre","meter","metres","meters"}:
            raise RuntimeError(f"Projected storage integration requires metre linear units, got {unit}")
        return np.full(h*w,abs(src.transform.a*src.transform.e),float)
    if not (src.crs and src.crs.to_epsg()==4326):
        raise RuntimeError(f"Storage integration requires projected metre CRS or EPSG:4326 geodetic areas, got {src.crs}")
    from pyproj import Geod
    geod=Geod(ellps="WGS84")
    cols=np.arange(c,c+w);rows=np.arange(r,r+h)
    left=np.asarray(xy(src.transform,np.zeros(w,dtype=int)+r,cols,offset="ul")[0],float)
    right=np.asarray(xy(src.transform,np.zeros(w,dtype=int)+r,cols,offset="lr")[0],float)
    top=np.asarray(xy(src.transform,rows,np.zeros(h,dtype=int)+c,offset="ul")[1],float)
    bottom=np.asarray(xy(src.transform,rows,np.zeros(h,dtype=int)+c,offset="lr")[1],float)
    col_width=np.abs(right-left)
    row_area=np.empty(h,float)
    lon0=float(left[0]);lon1=float(left[0]+np.nanmean(col_width))
    for i,(la,ba) in enumerate(zip(top,bottom)):
        row_area[i]=abs(geod.polygon_area_perimeter([lon0,lon1,lon1,lon0],[la,la,ba,ba])[0])
    return np.repeat(row_area[:,None],w,axis=1).ravel()


def _sum_or_nan(values):
    values=np.asarray(values,float)
    return float(np.nansum(values)) if np.isfinite(values).any() else np.nan


def _quantile_or_nan(values,q):
    values=np.asarray(values,float)
    return float(np.nanquantile(values,q)) if np.isfinite(values).any() else np.nan


def _median_or_nan(values):
    values=np.asarray(values,float)
    return float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan


def _storage_region_definitions():
    return {
        "all_valid":"All Phase 4 valid pixels with finite harmonic head predictions; no identifiability masking or area extrapolation.",
        "confined_identified":"Pixels where Ske_identifiability >= 2. Used for confined elastic storage diagnostics only; lag_c is not required for groundwater-storage timing.",
        "unconfined_scenario_valid":"Pixels where scenario-based unconfined storage can be computed from finite unconfined head harmonics and configured Sy scenarios. Classification is not based on Cu or lag_u.",
        "joint_storage_valid":"Pixels satisfying confined_identified and unconfined_scenario_valid. No area extrapolation is applied."
    }


def _write_storage_region_definitions(output):
    write_json({
        "storage_definition":"seasonal elastic groundwater-storage change",
        "no_area_extrapolation":True,
        "regions":_storage_region_definitions(),
        "unconfined_storage_identifiability_metadata":{
            "classification_status":"scenario_based_not_parameter_inverted",
            "does_not_use":"Cu or lag_u",
            "reason":"No pixel-level latent unconfined-head uncertainty map is currently available."
        }
    },Path(output)/"storage_region_definitions.json")


def _storage_time_basis(dates,period_days,origin):
    t=(pd.DatetimeIndex(pd.to_datetime(dates))-pd.Timestamp(origin)).days.to_numpy(float)
    return np.sin(2*np.pi*t/period_days),np.cos(2*np.pi*t/period_days)


def _series_from_sincos(sin_coeff,cos_coeff,sin_t,cos_t):
    if not np.isfinite(sin_coeff) or not np.isfinite(cos_coeff):
        return np.full(len(sin_t),np.nan)
    return sin_coeff*sin_t+cos_coeff*cos_t


def _write_storage_harmonic_timeseries(coefficients,model_variant,factory,geology_tif,dates,output,storage_config,period_days,origin):
    import rasterio
    sin,cos=_storage_time_basis(dates,period_days,origin)
    confined_sin=confined_cos=unconf_sin=unconf_cos=0.0
    valid_area=0.0
    n_valid=0
    with rasterio.open(geology_tif) as src:
        crs_text=str(src.crs)
        for block_i,(_,hc,hu,Z0,_,block_id) in enumerate(factory(),1):
            print(f"storage_harmonic_block {block_i}",flush=True)
            area_full=_pixel_area_m2_for_block(src,block_id)
            area=area_full[np.asarray(block_id["flat_index"],int)]
            design=np.column_stack([np.ones(len(Z0)),Z0]);ske,lagc,cu,lagu=decode_fields(coefficients,design,model_variant)
            valid=np.isfinite(ske)&np.isfinite(hc).all(1)&np.isfinite(hu).all(1)&np.isfinite(area)
            n_valid+=int(valid.sum());valid_area+=float(np.nansum(area[valid]))
            confined_sin+=_sum_or_nan(ske[valid]*hc[valid,0]*area[valid])
            confined_cos+=_sum_or_nan(ske[valid]*hc[valid,1]*area[valid])
            unconf_sin+=_sum_or_nan(hu[valid,0]*area[valid])
            unconf_cos+=_sum_or_nan(hu[valid,1]*area[valid])
    confined=_series_from_sincos(confined_sin,confined_cos,sin,cos)
    unconf_base=_series_from_sincos(unconf_sin,unconf_cos,sin,cos)
    manifest=json.loads((Path(output)/"insar_cube_manifest.json").read_text(encoding="utf-8")) if (Path(output)/"insar_cube_manifest.json").exists() else {}
    reference_frame_id=manifest.get("reference_frame_id")
    wide_rows=[];long_rows=[]
    for i,date in enumerate(pd.DatetimeIndex(pd.to_datetime(dates))):
        row={"date":date,"storage_crs":crs_text,"confined_elastic_storage_change_m3":confined[i],
             "valid_area_km2":valid_area/1e6,"n_valid_pixels":n_valid,
             "storage_definition":"seasonal elastic groundwater-storage change","reference_frame_id":reference_frame_id}
        for name in storage_config["specific_yield_scenarios"]:
            unconf=float(storage_config["specific_yield_scenarios"][name])*unconf_base[i]
            total=confined[i]+unconf
            row[f"unconfined_storage_change_{name}_m3"]=unconf
            row[f"total_elastic_storage_change_{name}_m3"]=total
            long_rows.append({"date":date,"specific_yield_scenario":name,
                              "confined_elastic_storage_change_m3":confined[i],
                              "unconfined_storage_change_m3":unconf,
                              "total_elastic_storage_change_m3":total,
                              "valid_area_km2":valid_area/1e6,
                              "storage_definition":"seasonal elastic groundwater-storage change",
                              "reference_frame_id":reference_frame_id})
        wide_rows.append(row)
    write_table(pd.DataFrame(wide_rows),output/"storage_harmonic_timeseries.csv")
    write_table(pd.DataFrame(long_rows),output/"storage_harmonic_map_timeseries.csv")


def _screen_draws(draws,model_variant,factory,screening,period_days):
    enabled=bool(screening.get("enabled",False))
    n=len(draws)
    accepted=np.ones(n,bool)
    reasons={k:np.zeros(n,bool) for k in ("ske_below_min","ske_above_max","lag_below_min","lag_above_max","cu_below_min","cu_above_max")}
    observed={"ske_min_observed":np.inf,"ske_max_observed":-np.inf,"lag_min_observed":np.inf,"lag_max_observed":-np.inf}
    if not enabled:
        return accepted,{"n_requested":n,"n_generated":n,"n_accepted":n,"n_rejected":0,"acceptance_fraction":1.0,
                         "rejection_reason_counts":{k:0 for k in reasons},
                         **{k:(None if not np.isfinite(v) else float(v)) for k,v in observed.items()}}
    ske_min=screening.get("ske_min",0);ske_min=0.0 if ske_min is None else float(ske_min)
    ske_max=screening.get("ske_max",np.inf);ske_max=np.inf if ske_max is None else float(ske_max)
    lag_min=screening.get("lag_min_days",0);lag_min=0.0 if lag_min is None else float(lag_min)
    lag_max=screening.get("lag_max_days",period_days);lag_max=float(period_days) if lag_max is None else float(lag_max)
    cu_min=float(screening.get("cu_min",0));cu_max=screening.get("cu_max")
    cu_max=np.inf if cu_max is None else float(cu_max)
    max_per_block=int(screening.get("max_screening_pixels_per_block",2000) or 2000)
    sampled_pixels=0;available_pixels=0
    for block_i,(_,_,_,Z0,_,_) in enumerate(factory(),1):
        print(f"screen_posterior_draw_block {block_i}",flush=True)
        available_pixels+=len(Z0)
        if len(Z0)>max_per_block:
            take=np.linspace(0,len(Z0)-1,max_per_block,dtype=int)
            Zscreen=Z0[take]
        else:
            Zscreen=Z0
        sampled_pixels+=len(Zscreen)
        for start in range(0,len(Zscreen),5000):
            end=min(len(Zscreen),start+5000);design=np.column_stack([np.ones(end-start),Zscreen[start:end]])
            decoded=[decode_fields(draw,design,model_variant) for draw in draws]
            ske=np.asarray([d[0] for d in decoded]);lagc=np.asarray([d[1] for d in decoded])
            cu=np.asarray([d[2] for d in decoded]);lagu=np.asarray([d[3] for d in decoded])
            lag_values=lagc if model_variant!="two_aquifer" else np.concatenate([lagc,lagu],axis=1)
            observed["ske_min_observed"]=min(observed["ske_min_observed"],float(np.nanmin(ske)))
            observed["ske_max_observed"]=max(observed["ske_max_observed"],float(np.nanmax(ske)))
            observed["lag_min_observed"]=min(observed["lag_min_observed"],float(np.nanmin(lag_values)))
            observed["lag_max_observed"]=max(observed["lag_max_observed"],float(np.nanmax(lag_values)))
            reasons["ske_below_min"]|=np.nanmin(ske,axis=1)<ske_min
            reasons["ske_above_max"]|=np.nanmax(ske,axis=1)>ske_max
            reasons["lag_below_min"]|=np.nanmin(lag_values,axis=1)<lag_min
            reasons["lag_above_max"]|=np.nanmax(lag_values,axis=1)>lag_max
            if model_variant=="two_aquifer":
                reasons["cu_below_min"]|=np.nanmin(cu,axis=1)<cu_min
                reasons["cu_above_max"]|=np.nanmax(cu,axis=1)>cu_max
    for mask in reasons.values():
        accepted&=~mask
    summary={"n_requested":n,"n_generated":n,"n_accepted":int(accepted.sum()),"n_rejected":int((~accepted).sum()),
             "acceptance_fraction":float(accepted.mean()),
             "screening_sample_policy":"deterministic_even_pixel_subsample_per_phase4_block",
             "max_screening_pixels_per_block":max_per_block,
             "sampled_pixels":int(sampled_pixels),"available_pixels":int(available_pixels),
             "rejection_reason_counts":{k:int(v.sum()) for k,v in reasons.items()},
             **{k:(None if not np.isfinite(v) else float(v)) for k,v in observed.items()}}
    minimum=int(screening.get("minimum_accepted_draws",0) or 0)
    if accepted.sum()<minimum:
        raise RuntimeError(f"Physical posterior screening accepted {accepted.sum()} draws, below minimum_accepted_draws={minimum}")
    return accepted,summary


def _storage_quantile_rows(series_by_draw,dates,base_row):
    rows=[]
    for i,date in enumerate(pd.DatetimeIndex(pd.to_datetime(dates))):
        confined=series_by_draw["confined"][:,i] if series_by_draw["confined"].size else np.array([np.nan])
        unconf=series_by_draw["unconfined"][:,i] if series_by_draw["unconfined"].size else np.array([np.nan])
        total=series_by_draw["total"][:,i] if series_by_draw["total"].size else np.array([np.nan])
        rows.append({**base_row,"date":date,
                     "confined_median_m3":_median_or_nan(confined),"confined_ci95_low_m3":_quantile_or_nan(confined,.025),"confined_ci95_high_m3":_quantile_or_nan(confined,.975),
                     "unconfined_median_m3":_median_or_nan(unconf),"unconfined_ci95_low_m3":_quantile_or_nan(unconf,.025),"unconfined_ci95_high_m3":_quantile_or_nan(unconf,.975),
                     "total_median_m3":_median_or_nan(total),"total_ci95_low_m3":_quantile_or_nan(total,.025),"total_ci95_high_m3":_quantile_or_nan(total,.975)})
    return rows


def _make_storage_summary(posterior,output,screening_summary,manifest):
    summary={"storage_definition":"seasonal elastic groundwater-storage change",
             "posterior_acceptance":screening_summary,
             "reference_frame_id":manifest.get("reference_frame_id"),
             "regions":{}}
    for (region,scenario,ptype),group in posterior.groupby(["region","specific_yield_scenario","posterior_type"]):
        if group["total_median_m3"].notna().any():
            idx_max=group["total_median_m3"].idxmax();idx_min=group["total_median_m3"].idxmin()
            amp=(group["total_median_m3"].max()-group["total_median_m3"].min())/2
            width=(group["total_ci95_high_m3"]-group["total_ci95_low_m3"]).mean()
        else:
            idx_max=idx_min=None;amp=np.nan;width=np.nan
        summary["regions"][f"{region}|{scenario}|{ptype}"]={
            "seasonal_amplitude_m3":None if not np.isfinite(amp) else float(amp),
            "peak_date":None if idx_max is None else str(pd.Timestamp(group.loc[idx_max,"date"]).date()),
            "trough_date":None if idx_min is None else str(pd.Timestamp(group.loc[idx_min,"date"]).date()),
            "mean_ci95_width_m3":None if not np.isfinite(width) else float(width),
            "valid_area_km2":float(group["valid_area_km2"].iloc[0]),
            "identified_area_km2":float(group["identified_area_km2"].iloc[0]),
            "region_status":str(group["region_status"].iloc[0])}
    write_json(summary,Path(output)/"storage_summary.json")


def _write_storage_harmonic_posterior_timeseries(coefficients,covariance,prior_precision,model_variant,factory,geology_tif,dates,output,storage_config,period_days,origin,n_draws,random_seed,uncertainty_config=None):
    import rasterio
    rng=np.random.default_rng(random_seed+17)
    n_draws=int(max(500,n_draws))
    draws=rng.multivariate_normal(coefficients,covariance,size=n_draws)
    uncertainty_config=uncertainty_config or {}
    screening=dict(uncertainty_config.get("physical_screening",{}))
    raw_mask=np.ones(n_draws,bool)
    screened_mask,screening_summary=_screen_draws(draws,model_variant,factory,screening,period_days)
    write_json(screening_summary,Path(output)/"posterior_draw_screening_summary.json")
    prior_cov=np.linalg.pinv(prior_precision)
    sin_t,cos_t=_storage_time_basis(dates,period_days,origin)
    regions=("all_valid","confined_identified","unconfined_scenario_valid","joint_storage_valid")
    aggregates={region:{"confined_sin":np.zeros(n_draws,float),"confined_cos":np.zeros(n_draws,float),
                        "unconf_sin":np.zeros(n_draws,float),"unconf_cos":np.zeros(n_draws,float)} for region in regions}
    counts={region:0 for region in regions};areas={region:0.0 for region in regions};area_total=0.;n_total=0
    with rasterio.open(geology_tif) as src:
        crs_text=str(src.crs)
        for block_i,(_,hc,hu,Z0,_,block_id) in enumerate(factory(),1):
            print(f"storage_posterior_block {block_i}",flush=True)
            area_full=_pixel_area_m2_for_block(src,block_id)
            area=area_full[np.asarray(block_id["flat_index"],int)]
            for start in range(0,len(Z0),5000):
                end=min(len(Z0),start+5000)
                design=np.column_stack([np.ones(end-start),Z0[start:end]])
                decoded=[decode_fields(draw,design,model_variant) for draw in draws]
                ske=np.asarray([d[0] for d in decoded]);lagc=np.asarray([d[1] for d in decoded])
                cu=np.asarray([d[2] for d in decoded]);lagu=np.asarray([d[3] for d in decoded])
                classes=_identifiability_classes(design,covariance,prior_cov,ske,lagc,cu,lagu,model_variant)
                valid_area=area[start:end]
                finite=np.isfinite(valid_area)&np.isfinite(hc[start:end]).all(1)&np.isfinite(hu[start:end]).all(1)
                confined_identified=(classes["ske"]>=2)&finite
                unconf_valid=finite
                joint=confined_identified&unconf_valid
                area_total+=float(np.nansum(valid_area[finite]));n_total+=int(finite.sum())
                hcc=hc[start:end];huu=hu[start:end]
                area_col=valid_area[None,:]
                masks={"all_valid":finite,"confined_identified":confined_identified,
                       "unconfined_scenario_valid":unconf_valid,"joint_storage_valid":joint}
                for region,mask in masks.items():
                    counts[region]+=int(mask.sum());areas[region]+=float(np.nansum(valid_area[mask]))
                for d0 in range(0,n_draws,50):
                    d1=min(n_draws,d0+50)
                    ske_d=ske[d0:d1]
                    for region,mask in masks.items():
                        weights=area_col[:,mask]
                        aggregates[region]["confined_sin"][d0:d1]+=np.nansum(ske_d[:,mask]*hcc[mask,0][None,:]*weights,axis=1)
                        aggregates[region]["confined_cos"][d0:d1]+=np.nansum(ske_d[:,mask]*hcc[mask,1][None,:]*weights,axis=1)
                        aggregates[region]["unconf_sin"][d0:d1]+=np.nansum(huu[mask,0][None,:]*weights,axis=1)
                        aggregates[region]["unconf_cos"][d0:d1]+=np.nansum(huu[mask,1][None,:]*weights,axis=1)
    _write_storage_region_definitions(output)
    manifest=json.loads((Path(output)/"insar_cube_manifest.json").read_text(encoding="utf-8")) if (Path(output)/"insar_cube_manifest.json").exists() else {}
    masks_by_type={"raw_laplace":raw_mask,"physically_screened":screened_mask}
    sensitivity_rows=[]
    for ske_max in screening.get("sensitivity_ske_max",[]) or []:
        s2=dict(screening);s2["ske_max"]=ske_max;s2["minimum_accepted_draws"]=0
        mask,summary=_screen_draws(draws,model_variant,factory,s2,period_days)
        masks_by_type[f"sensitivity_ske_max_{ske_max:g}"]=mask
        sensitivity_rows.append({"posterior_type":f"sensitivity_ske_max_{ske_max:g}","ske_max":ske_max,
                                 "n_accepted":summary["n_accepted"],"acceptance_fraction":summary["acceptance_fraction"]})
    rows=[]
    for region in regions:
        region_status="valid" if counts[region]>0 else ("no_identified_pixels" if "identified" in region or "joint" in region else "not_applicable")
        identified_area=areas[region] if region!="all_valid" else area_total
        area_fraction=identified_area/area_total if area_total>0 else np.nan
        for scenario,sy in storage_config["specific_yield_scenarios"].items():
            for posterior_type,draw_mask in masks_by_type.items():
                if counts[region]==0 or not draw_mask.any():
                    series={"confined":np.empty((0,len(sin_t))),"unconfined":np.empty((0,len(sin_t))),"total":np.empty((0,len(sin_t)))}
                else:
                    confined=aggregates[region]["confined_sin"][draw_mask,None]*sin_t[None,:]+aggregates[region]["confined_cos"][draw_mask,None]*cos_t[None,:]
                    unconf=float(sy)*(aggregates[region]["unconf_sin"][draw_mask,None]*sin_t[None,:]+aggregates[region]["unconf_cos"][draw_mask,None]*cos_t[None,:])
                    series={"confined":confined,"unconfined":unconf,"total":confined+unconf}
                base={"region":region,"specific_yield_scenario":scenario,"posterior_type":posterior_type,
                      "region_status":region_status if draw_mask.any() else "insufficient_coverage",
                      "n_valid_pixels":counts[region],"valid_area_km2":area_total/1e6,
                      "identified_area_km2":identified_area/1e6,"identified_area_fraction":area_fraction,
                      "storage_crs":crs_text}
                rows.extend(_storage_quantile_rows(series,dates,base))
    posterior=pd.DataFrame(rows)
    write_table(posterior,Path(output)/"storage_harmonic_posterior_timeseries.csv")
    write_table(pd.DataFrame(sensitivity_rows),Path(output)/"storage_posterior_sensitivity.csv")
    _make_storage_summary(posterior,output,screening_summary,manifest)


def run_phase4(config):
    output=ensure_dir(ROOT/config["project"]["output_dir"]);status=_require_previous(output,4)
    with (output/"latent_head_models.pkl").open("rb") as stream:models=pickle.load(stream)
    cube=GeoTiffCube.from_glob(resolve_config_path(config,config["insar"]["geotiff_glob"]),config["insar"]["displacement_unit"])
    reference=_reference_for_cube(config,output,cube)
    import json
    manifest=json.loads((output/"insar_cube_manifest.json").read_text(encoding="utf-8"))
    vertical_h5=_vertical_h5_path(config,output,manifest)
    if not vertical_h5.exists():write_vertical_h5(cube,resolve_config_path(config,config["insar"]["incidence_grid"]),vertical_h5,reference,manifest)
    geology=output/"geological_model_covariates.tif";quality=output/"geological_quality_layers.tif"
    if not geology.exists():
        gc=dict(config["geology"])
        if gc.get("input_mode")=="pre_rasterized":
            gc["raw_rasters"]=dict(gc["raw_rasters"])
            gc["raw_rasters"]["stack"]=str(resolve_config_path(config,gc["raw_rasters"]["stack"]))
        else:
            for key in [*['clay_group_1','clay_group_2','clay_group_3','clay_group_4'],'quaternary_thickness']:gc[key]=str(resolve_config_path(config,gc[key]))
            gc["extraction_layer_zone"]=dict(gc["extraction_layer_zone"]);gc["extraction_layer_zone"]["file"]=str(resolve_config_path(config,gc["extraction_layer_zone"]["file"]))
        rasterize_geology(gc,cube.epochs.source_file.iloc[0],geology,quality,output/"geological_standardization.json")
    pc=config["phase4"];inv=config.get("inversion",{})
    raw_factory,dates,design_metadata=_build_phase4_harmonic_factory(vertical_h5,geology,models,pc["spatial_block_rows"],pc["spatial_block_cols"],config["temporal"],
        config["geology"].get("spatial_basis"),config["project"].get("projected_crs","EPSG:32650"),pc["observation_sigma_mm"])
    import rasterio
    with rasterio.open(geology) as src:n_cov=src.count+design_metadata.get("n_spatial_basis",0)
    cache_dir=ensure_dir(output/"cache")
    cache_key=_phase4_cache_key(config,vertical_h5,geology,output/"latent_head_models.pkl",output/"insar_cube_manifest.json",design_metadata)
    cache_path=cache_dir/f"phase4_harmonic_blocks_{cache_key[:16]}.h5"
    if config.get("_force_phase4_cache_rebuild") and cache_path.exists():
        cache_path.unlink()
    factory=_cache_phase4_harmonic_blocks(raw_factory,cache_path,n_cov,cache_key,design_metadata)
    model_variant=inv.get("model_variant",pc.get("model_variant","auto"))
    mean,std=_phase4_priors(n_cov,model_variant,pc["prior"])
    result=harmonic_map_inversion_streaming(factory,n_cov,mean,std,period_days=config["temporal"]["annual_period_days"],
        collinearity_threshold=inv.get("collinearity_threshold",pc["collinearity_threshold"]),huber_delta=inv.get("huber_delta",pc["huber_delta"]),
        maxiter=inv.get("maximum_iterations",pc["max_iterations"]),model_variant=model_variant,return_field_blocks=False,
        observation_sigma_mm=pc["observation_sigma_mm"])
    result["design_metadata"]=design_metadata;result["cache_key"]=cache_key;result["cache_path"]=str(cache_path)
    serial={k:v for k,v in result.items() if k!="field_blocks"};serial["coefficients"]=result["coefficients"].tolist();serial["prior_mean"]=result["prior_mean"].tolist();serial["prior_precision"]=result["prior_precision"].tolist()
    write_json(serial,output/"map_diagnostics.json")
    write_table(pd.DataFrame({"parameter_name":result["parameter_names"],"map_value":result["coefficients"],"prior_mean":result["prior_mean"],
                  "prior_std":1/np.sqrt(np.diag(result["prior_precision"]))}),output/"map_coefficients.csv")
    np.savez_compressed(output/"map_coefficients.npz",coefficients=result["coefficients"],prior_mean=result["prior_mean"],prior_precision=result["prior_precision"],dates=np.asarray(dates.astype(str)))
    if not result["success"]:raise RuntimeError("Phase 4 MAP optimization failed; Phase 5 blocked")
    _write_phase4_maps_streaming(result,factory,geology,output,config["temporal"]["annual_period_days"])
    status.update({"phase_4":"complete","phase_5":"not_run"})
    write_json(status,output/"phase_status.json")
    return status


def run_phase5(config):
    output=ensure_dir(ROOT/config["project"]["output_dir"]);status=_require_previous(output,5)
    coefficient_file=np.load(output/"map_coefficients.npz",allow_pickle=True);coefficients=coefficient_file["coefficients"]
    prior_mean=coefficient_file["prior_mean"];prior_precision=coefficient_file["prior_precision"]
    with (output/"latent_head_models.pkl").open("rb") as stream:models=pickle.load(stream)
    pc=config["phase4"];diag_path=output/"map_diagnostics.json"
    diagnostics=json.loads(diag_path.read_text(encoding="utf-8")) if diag_path.exists() else {}
    manifest=json.loads((output/"insar_cube_manifest.json").read_text(encoding="utf-8"))
    vertical_h5=_vertical_h5_path(config,output,manifest)
    if Path(vertical_h5).exists():
        raw_factory,dates,design_metadata=_build_phase4_harmonic_factory(vertical_h5,output/"geological_model_covariates.tif",models,pc["spatial_block_rows"],pc["spatial_block_cols"],config["temporal"],
            config["geology"].get("spatial_basis"),config["project"].get("projected_crs","EPSG:32650"),pc["observation_sigma_mm"])
        import rasterio
        with rasterio.open(output/"geological_model_covariates.tif") as src:n_cov=src.count+design_metadata.get("n_spatial_basis",0)
        cache_key=_phase4_cache_key(config,vertical_h5,output/"geological_model_covariates.tif",output/"latent_head_models.pkl",output/"insar_cube_manifest.json",design_metadata)
        cache_path=output/"cache"/f"phase4_harmonic_blocks_{cache_key[:16]}.h5"
        factory=_cache_phase4_harmonic_blocks(raw_factory,cache_path,n_cov,cache_key,design_metadata)
    else:
        import h5py
        cache_path=Path(diagnostics.get("cache_path",""))
        if not cache_path.exists():
            raise FileNotFoundError(f"Neither vertical HDF5 nor recorded Phase 4 cache is available: {vertical_h5}, {cache_path}")
        with h5py.File(cache_path,"r") as h5:
            design_metadata=json.loads(h5.attrs.get("design_metadata","{}"))
        factory=_phase4_cached_factory(cache_path)
    model_variant=diagnostics.get("model_variant",pc["model_variant"])
    covariance,hessian=harmonic_streaming_gauss_newton(coefficients,model_variant,factory,prior_precision,config["temporal"]["annual_period_days"],
        observation_sigma_mm=pc["observation_sigma_mm"],huber_delta=config.get("inversion",{}).get("huber_delta",pc["huber_delta"]))
    posterior_std=np.sqrt(np.clip(np.diag(covariance),0,None));prior_variance=np.diag(np.linalg.pinv(prior_precision));reduction=1-np.diag(covariance)/prior_variance
    table=pd.DataFrame({"parameter_index":np.arange(len(coefficients)),"posterior_mean":coefficients,"posterior_std":posterior_std,
        "ci95_low":coefficients-1.96*posterior_std,"ci95_high":coefficients+1.96*posterior_std,
        "prior_to_posterior_variance_reduction":reduction,
        "identifiability_class":np.where(reduction>=.5,"identified",np.where(reduction>=.1,"weak","not_identified"))})
    write_table(table,output/"posterior_coefficient_summary.csv");np.savez_compressed(output/"posterior_coefficients.npz",mean=coefficients,covariance=covariance,hessian=hessian)
    _write_phase5_posterior_maps(coefficients,covariance,prior_precision,model_variant,factory,output/"geological_model_covariates.tif",output,design_metadata,config["uncertainty"]["posterior_draws"],config["project"]["random_seed"],config["temporal"]["annual_period_days"])
    _write_storage_harmonic_timeseries(coefficients,model_variant,factory,output/"geological_model_covariates.tif",coefficient_file["dates"],output,config["storage"],config["temporal"]["annual_period_days"],config["temporal"]["harmonic_origin"])
    _write_storage_harmonic_posterior_timeseries(coefficients,covariance,prior_precision,model_variant,factory,output/"geological_model_covariates.tif",coefficient_file["dates"],output,config["storage"],config["temporal"]["annual_period_days"],config["temporal"]["harmonic_origin"],config["uncertainty"]["posterior_draws"],config["project"]["random_seed"],config.get("uncertainty",{}))
    status.update({"phase_5":"complete","engineering_execution":"complete","phase_4_parameter_inversion":"complete",
                   "phase_5_parameter_posterior":"complete_provisional","Ske_identifiability":"complete",
                   "other_parameter_identifiability":"complete_provisional",
                   "phase_5_storage":"experimental_area_corrected_posterior_seasonal_elastic_storage",
                   "storage_posterior_uncertainty":"complete_provisional",
                   "long_term_compaction_storage":"not_complete","model_variant_comparison":"not_complete",
                   "independent_validation":"not_integrated"})
    write_json(status,output/"phase_status.json");return status


def run_model_compare(config):
    output=ensure_dir(ROOT/config["project"]["output_dir"])
    diagnostics=json.loads((output/"map_diagnostics.json").read_text(encoding="utf-8")) if (output/"map_diagnostics.json").exists() else {}
    coefficient_summary=pd.read_csv(output/"posterior_coefficient_summary.csv") if (output/"posterior_coefficient_summary.csv").exists() else pd.DataFrame()
    def identified_fraction():
        if coefficient_summary.empty or "identifiability_class" not in coefficient_summary:return np.nan
        return float((coefficient_summary["identifiability_class"]=="identified").mean())
    area_fractions={}
    for name,path in [("Ske_identified_area_fraction",output/"Ske_identifiability.tif"),
                      ("combined_identified_area_fraction",output/"combined_deformation_identifiability.tif")]:
        try:
            import rasterio
            with rasterio.open(path) as src:
                arr=src.read(1,masked=True).compressed()
                area_fractions[name]=float(np.mean(arr>=2)) if arr.size else np.nan
        except Exception:
            area_fractions[name]=np.nan
    coeff_path=output/"map_coefficients.npz"
    if not coeff_path.exists():
        raise FileNotFoundError(f"Model comparison requires MAP coefficients: {coeff_path}")
    coeff=np.load(coeff_path,allow_pickle=True)["coefficients"]
    cache_path=Path(diagnostics.get("cache_path",""))
    if not cache_path.exists():
        raise FileNotFoundError(f"Model comparison requires recorded Phase 4 cache: {cache_path}")
    factory=_phase4_cached_factory(cache_path)
    observation_sigma=float(config["phase4"]["observation_sigma_mm"])
    period_days=float(config["temporal"]["annual_period_days"])
    p=len(coeff)//4 if len(coeff)%4==0 else max(1,len(coeff)//2)

    def predict_variant(theta,variant,hc,hu,Z0):
        Z=np.column_stack([np.ones(len(Z0)),Z0])
        if variant=="confined_only":
            ske,lagc,_,_=decode_fields(theta[:2*p],Z,"confined_only")
            return 1000*ske[:,None]*rotate_coefficients(hc,lagc,period_days)
        ske,lagc,cu,lagu=decode_fields(theta[:4*p],Z,"two_aquifer")
        if variant=="two_aquifer_shared_unconfined_lag":
            shared=float(np.nanmedian(lagu)) if np.isfinite(lagu).any() else 0.0
            lagu=np.full_like(lagu,shared)
        return 1000*(ske[:,None]*rotate_coefficients(hc,lagc,period_days)+cu[:,None]*rotate_coefficients(hu,lagu,period_days))

    def evaluate_variant(variant,max_pixels_per_block=6000):
        sq=0.;n=0;loglike=0.;blocks=0;pixels=0
        for obs,hc,hu,Z0,weights,block_id in factory():
            if len(obs)==0:
                continue
            take=np.arange(len(obs))
            if len(take)>max_pixels_per_block:
                take=np.linspace(0,len(obs)-1,max_pixels_per_block,dtype=int)
            if isinstance(block_id,dict):
                fold_seed=int(block_id.get("row",0))+int(block_id.get("col",0))
            else:
                fold_seed=int(block_id)
            validation=((take + fold_seed) % 5)==0
            take=take[validation]
            if len(take)==0:
                continue
            pred=predict_variant(coeff,variant,hc[take],hu[take],Z0[take])
            residual=obs[take]-pred
            valid=np.isfinite(residual).all(1)
            if not valid.any():
                continue
            res=residual[valid]
            sq+=float(np.sum(res**2));n+=int(res.size)
            loglike+=float(-0.5*np.sum((res/observation_sigma)**2)-res.size*np.log(observation_sigma*np.sqrt(2*np.pi)))
            blocks+=1;pixels+=int(valid.sum())
        rmse=float(np.sqrt(sq/n)) if n else np.nan
        return {"spatial_block_rmse_mm":rmse,"validation_log_likelihood":loglike if n else np.nan,
                "validation_observations":n,"validation_pixels":pixels,"validation_blocks":blocks}

    spacings=config.get("geology",{}).get("spatial_basis",{}).get("candidate_spacing_km",[5,10,15])
    models=[("M0","confined_only",2*p),("M1","two_aquifer_shared_unconfined_lag",3*p+1),("M2","two_aquifer_independent_lag",4*p)]
    metrics={variant:evaluate_variant(variant) for _,variant,_ in models}
    current_variant=diagnostics.get("model_variant")
    current_spacing=float(spacings[0]) if spacings else np.nan
    rows=[]
    for model_id,variant,npar in models:
        for spacing in spacings:
            is_current=(variant=="two_aquifer_independent_lag" and current_variant=="two_aquifer" and float(spacing)==current_spacing)
            obj=diagnostics.get("objective",np.nan) if is_current else np.nan
            metric=metrics[variant]
            nobs=metric.get("validation_observations",0)
            rmse=metric["spatial_block_rmse_mm"]
            aic=(2*npar+nobs*np.log(max(rmse**2,1e-12))) if nobs and np.isfinite(rmse) else np.nan
            bic=(np.log(nobs)*npar+nobs*np.log(max(rmse**2,1e-12))) if nobs and np.isfinite(rmse) else np.nan
            rows.append({"model_id":model_id,"model_variant":variant,"rbf_spacing_km":float(spacing),
                         "n_parameters":npar,"training_objective":obj,"training_rmse_mm":rmse,
                         **metric,
                         "identified_coefficient_fraction":identified_fraction() if is_current else np.nan,
                         "Ske_identified_area_fraction":area_fractions["Ske_identified_area_fraction"] if is_current else np.nan,
                         "combined_identified_area_fraction":area_fractions["combined_identified_area_fraction"] if is_current else np.nan,
                         "aic_like":aic,"bic_like":bic,
                         "status":"real_cache_spatial_validation_current_spacing" if float(spacing)==current_spacing else "real_cache_spatial_validation_spacing_label"})
    write_table(pd.DataFrame(rows),output/"model_comparison.csv")
    return {"model_compare":"complete"}


def run_geology_model_compare(config):
    output=ensure_dir(ROOT/config["project"]["output_dir"])
    if not config.get("_execute_model_compare",False):
        return {"geology_model_compare":"not_started_requires_execute"}
    if not config.get("_smoke_test",False):
        from spatial_refit_validation import run_real_m1_validation, select_simpler_within_two_percent
        rows,folds=run_real_m1_validation(config,output,stage="geology",resume=bool(config.get("_resume_model_compare",False)))
        write_table(rows,output/"geology_model_comparison.csv")
        write_table(folds,output/"geology_model_fold_metrics.csv")
        valid=rows[rows["status"].isin(["complete_validated"])].copy()
        selected=select_simpler_within_two_percent(valid.rename(columns={"mean_validation_rmse_mm":"mean_spatial_rmse_mm","parameter_count":"n_parameters"}).to_dict("records"),"geology_model_id") if not valid.empty else None
        payload={"status":"complete_validated" if (rows["status"]=="complete_validated").all() else "partial",
                 "selected_geology_model":selected,"selection_basis":"real_data_spatial_refit","phase4_restart_allowed":False,
                 "reason":"G comparison complete; lag_c comparison and selected config still required." if selected else "G comparison is partial; no final geology model is selected until all real spatial folds complete."}
        write_json(payload,output/"geology_model_selection.json")
        return {"geology_model_compare":payload["status"]}
    from spatial_refit_validation import run_m1_smoke_test, select_simpler_within_two_percent
    rows,folds,gradient_errors=run_m1_smoke_test(output,stage="geology",maxiter=int(config.get("_smoke_maxiter",20)))
    write_table(rows,output/"geology_model_comparison.csv")
    write_table(folds,output/"geology_model_fold_metrics.csv")
    passed=bool((rows["status"]=="smoke_test_passed").all())
    selected=select_simpler_within_two_percent(rows.rename(columns={"mean_spatial_rmse_mm":"mean_spatial_rmse_mm"}).to_dict("records"),"geology_model_id") if passed else None
    payload={"status":"smoke_test_passed" if passed else "failed","validated_scope":"smoke_test_only_not_full_spatial_validation",
             "selected_geology_model_smoke":selected,"gradient_errors":gradient_errors,
             "phase4_restart_allowed":False,
             "reason":"Smoke test validates M1 optimizer plumbing only; full G0-G3 spatial refit is still required before Phase 4."}
    write_json(payload,output/"geology_model_selection.json")
    return {"geology_model_compare":payload["status"]}


def run_lag_c_model_compare(config):
    output=ensure_dir(ROOT/config["project"]["output_dir"])
    if not config.get("_execute_model_compare",False):
        return {"lag_c_model_compare":"not_started_requires_execute"}
    if not config.get("_smoke_test",False):
        from spatial_refit_validation import run_real_m1_validation, select_simpler_within_two_percent
        rows,folds=run_real_m1_validation(config,output,stage="lag_c",resume=bool(config.get("_resume_model_compare",False)))
        write_table(rows,output/"lag_c_model_comparison.csv")
        write_table(folds,output/"lag_c_model_fold_metrics.csv")
        valid=rows[rows["status"].isin(["complete_validated"])].copy()
        selected=select_simpler_within_two_percent(valid.rename(columns={"mean_validation_rmse_mm":"mean_spatial_rmse_mm","parameter_count":"n_parameters"}).to_dict("records"),"lag_c_model_id") if not valid.empty else None
        payload={"status":"complete_validated" if (rows["status"]=="complete_validated").all() else "partial",
                 "selected_lag_c_model":selected,"selection_basis":"real_data_spatial_refit","phase4_restart_allowed":False,
                 "reason":"Full Phase 4 remains blocked until selected_model_config.yaml is generated and reviewed." if selected else "lag_c comparison is partial; no final lag_c mode is selected until all real spatial folds complete."}
        write_json(payload,output/"lag_c_model_selection.json")
        return {"lag_c_model_compare":payload["status"]}
    from spatial_refit_validation import run_m1_smoke_test, select_simpler_within_two_percent
    rows,folds,gradient_errors=run_m1_smoke_test(output,stage="lag_c",maxiter=int(config.get("_smoke_maxiter",20)))
    write_table(rows,output/"lag_c_model_comparison.csv")
    write_table(folds,output/"lag_c_model_fold_metrics.csv")
    passed=bool((rows["status"]=="smoke_test_passed").all())
    selected=select_simpler_within_two_percent(rows.to_dict("records"),"lag_c_model_id") if passed else None
    payload={"status":"smoke_test_passed" if passed else "failed","validated_scope":"smoke_test_only_not_full_spatial_validation",
             "selected_lag_c_model_smoke":selected,"gradient_errors":gradient_errors,
             "phase4_restart_allowed":False,
             "reason":"Smoke test validates lag_c candidate plumbing only; full L0-L2 spatial refit is still required before Phase 4."}
    write_json(payload,output/"lag_c_model_selection.json")
    return {"lag_c_model_compare":payload["status"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--phase", default="1", choices=["1","2","3","4","5","model_compare","geology_model_compare","lag_c_model_compare","all"])
    parser.add_argument("--resume",action="store_true",help="Skip phases already marked complete")
    parser.add_argument("--force",action="store_true",help="Re-run a completed phase")
    parser.add_argument("--execute",action="store_true",help="Execute model-comparison diagnostics for --phase model_compare")
    parser.add_argument("--output-root",default=None,help="Override project.output_dir for revision reruns")
    parser.add_argument("--smoke-test",action="store_true",help="Run low-cost M1 model-comparison smoke test")
    parser.add_argument("--folds",type=int,default=None,help="Requested fold count for model comparison")
    parser.add_argument("--models",default=None,help="Comma-separated model ids for model comparison")
    parser.add_argument("--force-fold",default=None,help="Force one fold id for model comparison rerun")
    parser.add_argument("--maxiter",type=int,default=None,help="Override max iterations for model-comparison refits")
    args = parser.parse_args()
    config = load_config(args.config)
    default_output_dir=config["project"].get("output_dir","outputs")
    if args.output_root:
        config["project"]["output_dir"]=args.output_root
        config.setdefault("geology",{})["revision_output_dir"]=args.output_root
    def seed_revision_output():
        target=ensure_dir(ROOT/config["project"]["output_dir"])
        source=ROOT/default_output_dir
        if target.resolve()==source.resolve() or not source.exists():
            return
        needed=["phase_status.json","well_timeseries.csv","well_summary.csv","lag_summary.csv","tlcc_spectra.csv",
                "insar_at_wells.csv","well_harmonic_decomposition.csv","latent_head_leave_well_out_validation.csv",
                "latent_head_rank_selection.json","latent_head_models.pkl","insar_cube_manifest.json","insar_epochs.csv",
                "bulletin_standardized.csv","bulletin_constraints.csv","run_provenance.json"]
        for name in needed:
            src=source/name;dst=target/name
            if src.exists() and not dst.exists():
                dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        cache_src=source/"cache";cache_dst=target/"cache"
        if cache_src.exists():
            cache_dst.mkdir(parents=True,exist_ok=True)
            for item in cache_src.glob("vertical_timeseries_*.h5"):
                dst=cache_dst/item.name
                if not dst.exists():shutil.copy2(item,dst)
    seed_revision_output()
    light_model_compare=args.phase in {"geology_model_compare","lag_c_model_compare"}
    if light_model_compare:
        input_paths=[resolve_config_path(config,args.config)]
        revision_root=ROOT/config["project"]["output_dir"]
        for name in ("clay_thickness_semantics_check.json","geological_design_matrix_audit.json"):
            path=revision_root/name
            if path.exists():input_paths.append(path)
    else:
        insar_inputs=[Path(path) for path in glob.glob(str(resolve_config_path(config,config["insar"]["geotiff_glob"]))) if "velocity" not in Path(path).name.lower()]
        input_paths=[resolve_config_path(config,config["groundwater"]["file"]),resolve_config_path(config,config["bulletin"]["file"]),resolve_config_path(config,config["bulletin"]["verified_file"]),
                     resolve_config_path(config,config["insar"]["incidence_grid"]),*insar_inputs]
        geology_config=config.get("geology",{})
        if geology_config.get("input_mode")=="pre_rasterized":
            preprocessing=geology_config.get("preprocessing",{})
            for key in ("clay_group_1","clay_group_2","clay_group_3","clay_group_4","quaternary_thickness","extraction_layer_zone"):
                if key in preprocessing:
                    input_paths.append(resolve_config_path(config,preprocessing[key]["file"]))
            raw_stack=geology_config.get("raw_rasters",{}).get("stack")
            if raw_stack and resolve_config_path(config,raw_stack).exists():
                input_paths.append(resolve_config_path(config,raw_stack))
        else:
            for key in ("clay_group_1","clay_group_2","clay_group_3","clay_group_4","quaternary_thickness"):
                input_paths.append(resolve_config_path(config,config["geology"][key]))
            input_paths.append(resolve_config_path(config,config["geology"]["extraction_layer_zone"]["file"]))
    provenance=runtime_provenance(config,input_paths)
    config["_run_provenance"]=provenance
    config["_force_phase4_cache_rebuild"]=bool(args.force and args.phase in {"4","all"})
    config["_execute_model_compare"]=bool(args.execute)
    config["_smoke_test"]=bool(args.smoke_test)
    config["_resume_model_compare"]=bool(args.resume)
    config["_requested_folds"]=args.folds
    config["_requested_models"]=args.models
    config["_force_fold"]=args.force_fold
    if args.maxiter is not None:
        config["_model_compare_maxiter"]=args.maxiter
    print(f"code_version={provenance['code_version']} git_commit={provenance['git_commit']} config_sha256={provenance['config_sha256']}")
    write_json(provenance,ensure_dir(ROOT/config["project"]["output_dir"])/"run_provenance.json")
    if args.resume and args.force:raise ValueError("--resume and --force are mutually exclusive")
    phase_map={"1":run_phase1,"2":run_phase2,"3":run_phase3,"4":run_phase4,"5":run_phase5,"model_compare":run_model_compare,
               "geology_model_compare":run_geology_model_compare,"lag_c_model_compare":run_lag_c_model_compare}
    def execute(label):
        status_path=ROOT/config["project"]["output_dir"]/"phase_status.json"
        if args.resume and status_path.exists():
            import json
            if json.loads(status_path.read_text(encoding="utf-8")).get(f"phase_{label}")=="complete":return
        started=time.time();before={"phase":label,"started":pd.Timestamp.utcnow(),"config_sha256":config["_config_sha256"],
                                    "run_id":provenance.get("run_id"),"resume_from_run_id":None,
                                    "code_version":provenance.get("code_version"),"git_commit":provenance.get("git_commit"),
                                    "source_tree_sha256":provenance.get("source_tree_sha256")}
        write_json(before,ROOT/config["project"]["output_dir"]/f"phase_{label}_run_manifest.json")
        try:
            result=phase_map[label](config)
            phase_key=f"phase_{label}" if str(label).isdigit() else str(label)
            phase_state=result.get(phase_key) if isinstance(result,dict) else None
            final_state="complete" if phase_state=="complete" else (phase_state or "incomplete")
            before.update({"finished":pd.Timestamp.utcnow(),"elapsed_seconds":time.time()-started,"status":final_state})
            write_json(before,ROOT/config["project"]["output_dir"]/f"phase_{label}_run_manifest.json")
        except Exception as exc:
            import traceback
            before.update({"finished":pd.Timestamp.utcnow(),"elapsed_seconds":time.time()-started,"status":"failed",
                           "error_type":type(exc).__name__,"error_message":str(exc),"traceback":traceback.format_exc()})
            write_json(before,ROOT/config["project"]["output_dir"]/f"phase_{label}_run_manifest.json")
            raise
    if args.phase=="all":
        for label in ("1","2","3","4","5"):execute(label)
    else:
        execute(args.phase)


if __name__ == "__main__":
    main()

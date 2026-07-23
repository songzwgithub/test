"""Projected daily weighted-ALS latent-head fields and repeated spatial CV."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial.distance import cdist


def project_coordinates(lon_lat, target_crs="EPSG:32650"):
    points = np.asarray(lon_lat, float); transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    x, y = transformer.transform(points[:, 0], points[:, 1]); return np.column_stack([x, y])


def farthest_point_centers(points, n_centers):
    points = np.asarray(points, float); selected = [int(np.argmin(points[:, 0]+points[:, 1]))]
    distance = cdist(points, points[selected]).ravel()
    while len(selected) < min(n_centers, len(points)):
        selected.append(int(np.argmax(distance)))
        distance = np.minimum(distance, cdist(points, points[[selected[-1]]]).ravel())
    return points[selected]


def _rbf(points, centers, scale): return np.exp(-.5*(cdist(points, centers)/scale)**2)


def weighted_als_low_rank(matrix,rank,weights=None,smoothness=10.,ridge=1e-3,max_iter=100,tolerance=1e-5):
    """Weighted ALS; data loss is evaluated only where actual/short-gap records have weight."""
    X=np.asarray(matrix,float);observed=np.isfinite(X);W=observed.astype(float) if weights is None else np.nan_to_num(weights)*observed
    mean=np.divide(np.nansum(W*X,axis=1),W.sum(axis=1),out=np.zeros(X.shape[0]),where=W.sum(axis=1)>0);Y=X-mean[:,None]
    filled=np.where(observed,Y,0);u,s,vt=np.linalg.svd(filled,full_matrices=False);U=u[:,:rank]*s[:rank];V=vt[:rank].copy()
    from scipy.sparse import diags, eye
    from scipy.sparse.linalg import factorized
    n_time=X.shape[1]
    D=diags([np.ones(n_time-2),-2*np.ones(n_time-2),np.ones(n_time-2)],[0,1,2],shape=(n_time-2,n_time),format="csc")
    smooth_solver=factorized((eye(n_time,format="csc")+smoothness*(D.T@D)).tocsc())
    previous=np.inf
    for _ in range(max_iter):
        for i in range(X.shape[0]):
            ok=W[i]>0;A=V[:,ok]*np.sqrt(W[i,ok]);b=Y[i,ok]*np.sqrt(W[i,ok]);U[i]=np.linalg.solve(A@A.T+ridge*np.eye(rank),A@b)
        for t in range(X.shape[1]):
            ok=W[:,t]>0;A=U[ok]*np.sqrt(W[ok,t,None]);b=Y[ok,t]*np.sqrt(W[ok,t]);V[:,t]=np.linalg.solve(A.T@A+ridge*np.eye(rank),A.T@b)
        V=np.asarray([smooth_solver(row) for row in V]);prediction=mean[:,None]+U@V
        loss=np.sum(W*np.where(observed,(X-prediction)**2,0))/max(W.sum(),1)
        if abs(previous-loss)<=tolerance*max(1,previous):break
        previous=loss
    return prediction,V,U,mean,float(loss)


@dataclass
class LatentHeadModel:
    rank: int; dates: pd.DatetimeIndex; components: np.ndarray; centers: np.ndarray
    score_coefficients: np.ndarray; mean_coefficients: np.ndarray; scale_m: float
    aquifer_type: str; projected_crs: str; prediction_std_m: float

    def predict(self, lon_lat, dates=None, return_std=False):
        points = project_coordinates(lon_lat, self.projected_crs)
        B = np.column_stack([np.ones(len(points)), _rbf(points, self.centers, self.scale_m)])
        if dates is not None:
            requested = pd.DatetimeIndex(pd.to_datetime(dates))
            target = requested.view("i8");source = self.dates.view("i8")
            components = np.vstack([np.interp(target,source,row,left=np.nan,right=np.nan) for row in self.components])
            values = (B@self.mean_coefficients)[:, None]+(B@self.score_coefficients)@components
        else:
            values = (B@self.mean_coefficients)[:, None]+(B@self.score_coefficients)@self.components
        return (values, np.full_like(values, self.prediction_std_m)) if return_std else values


def _matrix(frame):
    f = frame.copy(); f["date"] = pd.to_datetime(f["date"]).dt.normalize()
    if "is_valid_for_model" in f: f = f[f["is_valid_for_model"].astype(str).str.lower().isin(["true", "1"])]
    value = "head_anomaly_m" if "head_anomaly_m" in f else "hydraulic_head_m"
    M = f.pivot_table(index="well_id", columns="date", values=value, aggfunc="median").sort_index(axis=1)
    full_dates=pd.date_range(M.columns.min(),M.columns.max(),freq="D");M=M.reindex(columns=full_dates)
    W=f.pivot_table(index="well_id",columns="date",values="observation_weight",aggfunc="max").reindex(index=M.index,columns=full_dates).fillna(0) if "observation_weight" in f else M.notna().astype(float)
    meta = f.drop_duplicates("well_id").set_index("well_id")[["lon", "lat", "aquifer_type"]].loc[M.index]
    return M,W,meta


def fit_low_rank(frame, rank=4, ridge=1e-3, projected_crs="EPSG:32650", max_centers=32,smoothness=10.):
    M,W,meta = _matrix(frame)
    if len(M) <= rank: raise ValueError("Too few wells for requested rank")
    reconstructed, components, scores,means,_ = weighted_als_low_rank(M.to_numpy(float),rank,W.to_numpy(float),smoothness,ridge)
    points = project_coordinates(meta[["lon", "lat"]], projected_crs)
    centers = farthest_point_centers(points, min(max_centers, max(4, len(points)//3)))
    distances = cdist(points, points); scale = np.median(distances[distances > 0])
    B = np.column_stack([np.ones(len(points)), _rbf(points, centers, scale)])
    penalty = ridge*np.eye(B.shape[1]); penalty[0, 0] = 0; solve = np.linalg.pinv(B.T@B+penalty)@B.T
    fitted = means[:, None]+scores@components
    residual = M.to_numpy(float)-fitted; std = float(np.nanstd(residual))
    return LatentHeadModel(rank, pd.DatetimeIndex(M.columns), components, centers, solve@scores,
        solve@means, float(scale), str(meta["aquifer_type"].iloc[0]), projected_crs, std)


def _metrics(truth, prediction):
    ok = np.isfinite(truth)&np.isfinite(prediction); e = prediction[ok]-truth[ok]
    ss = np.sum((truth[ok]-np.mean(truth[ok]))**2) if ok.any() else np.nan
    return {"RMSE": float(np.sqrt(np.mean(e**2))) if ok.any() else np.nan,
            "MAE": float(np.mean(abs(e))) if ok.any() else np.nan, "Bias": float(np.mean(e)) if ok.any() else np.nan,
            "R2": float(1-np.sum(e**2)/ss) if ok.any() and ss > 0 else np.nan, "n_predictions": int(ok.sum())}


def repeated_holdout_validation(frame, ranks=range(2,9), repeats=5, holdout_fraction=.2,
                                spatial_block_size_m=30000, random_seed=42,
                                validation_schemes=("random","spatial_block"), **fit_kwargs):
    if "is_valid_for_model" in frame:
        frame=frame[frame["is_valid_for_model"].astype(str).str.lower().isin(["true","1"])].copy()
    if "baseline_sufficient" in frame:
        frame=frame[frame["baseline_sufficient"].astype(str).str.lower().isin(["true","1"])].copy()
    M,W,meta = _matrix(frame); wells = M.index.to_numpy(); points = project_coordinates(meta[["lon","lat"]])
    blocks = np.floor((points-points.min(axis=0))/spatial_block_size_m).astype(int); block_id = blocks[:,0]*10000+blocks[:,1]
    rng = np.random.default_rng(random_seed); rows=[]
    all_dates=pd.DatetimeIndex(M.columns)
    for rank in ranks:
        for repeat in range(repeats):
            for scheme in validation_schemes:
                print(f"latent_cv rank={rank} repeat={repeat+1}/{repeats} scheme={scheme}",flush=True)
                if scheme == "random":
                    held = rng.choice(wells, max(1,int(np.ceil(len(wells)*holdout_fraction))), replace=False)
                    train = frame[~frame["well_id"].isin(held)]; test = frame[frame["well_id"].isin(held)].copy()
                elif scheme == "spatial_block":
                    chosen = rng.choice(np.unique(block_id)); held = wells[block_id == chosen]
                    train = frame[~frame["well_id"].isin(held)]; test = frame[frame["well_id"].isin(held)].copy()
                else:
                    width=max(30,int(len(all_dates)*holdout_fraction));start=int(rng.integers(0,max(1,len(all_dates)-width+1)))
                    held_dates=all_dates[start:start+width];held=wells
                    mask=pd.to_datetime(frame["date"]).dt.normalize().isin(held_dates)
                    train = frame[~mask]; test = frame[mask].copy()
                if train["well_id"].nunique() <= rank or test.empty: continue
                model = fit_low_rank(train, rank=rank, **fit_kwargs); dates = pd.DatetimeIndex(pd.to_datetime(test["date"]).unique())
                held=np.asarray(pd.unique(test["well_id"]))
                coords = meta.loc[held][["lon","lat"]].to_numpy(); pred = model.predict(coords, dates)
                lookup={(w,pd.Timestamp(d).normalize()):pred[i,j] for i,w in enumerate(held) for j,d in enumerate(dates)}
                value="head_anomaly_m" if "head_anomaly_m" in test else "hydraulic_head_m"
                predicted=np.array([lookup.get((w,pd.Timestamp(d).normalize()),np.nan) for w,d in zip(test.well_id,test.date)])
                rows.append({"rank":rank,"repeat":repeat,"scheme":scheme,"n_holdout_wells":len(held),**_metrics(test[value].to_numpy(float),predicted)})
    metrics=pd.DataFrame(rows); spatial=metrics[metrics.scheme=="spatial_block"].groupby("rank")["RMSE"].mean()
    best=int(spatial.idxmin()); return metrics,best,fit_low_rank(frame,rank=best,**fit_kwargs)


leave_well_out_validation = repeated_holdout_validation


def fit_aquifer_models(frame, **kwargs):
    output={}
    for aquifer in ("unconfined","confined"):
        metrics,rank,model=repeated_holdout_validation(frame[frame.aquifer_type==aquifer],**kwargs)
        output[aquifer]={"metrics":metrics,"best_rank":rank,"model":model}
    return output

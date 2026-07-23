from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm, TwoSlopeNorm, ListedColormap, BoundaryNorm

from plotting.common import (ROOT,OUTPUT,FIGROOT,MAIN,SUPP,SOURCE,WIDTH_DOUBLE_MM,WIDTH_SINGLE_MM,
                             CMAP_SEQ,CMAP_DIV,CMAP_LAG,IDENT_CMAP,IDENT_NORM,ensure_dirs,figure_size,
                             read_table,raster_summary,map_panel,panel_label,save_figure,not_generated,
                             write_source,caption_record,add_validation,CMAP_PHASE)


def _exists(*paths):
    return all(Path(p).exists() for p in paths)


def _safe_stats(path):
    if not Path(path).exists():
        return {"valid_pixels": 0}
    return raster_summary(path)


def _setup_two_year_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelrotation=0)


def _signed_lag(value):
    x=((float(value)+182.62125) % 365.2425) - 182.62125
    return x


def _ci_short_arc(center, low, high, period=365.2425):
    if not all(np.isfinite([center, low, high])):
        return None
    c=_signed_lag(center)
    rel_low=((float(low)-float(center)+period/2) % period) - period/2
    rel_high=((float(high)-float(center)+period/2) % period) - period/2
    if rel_low > rel_high:
        rel_low, rel_high = rel_high, rel_low
    return c + rel_low, c + rel_high, c


def _select_representative_wells(lag):
    reliable=lag[lag.get("lag_reliable",False).astype(bool)].copy()
    reliable["joint_snr"]=reliable["groundwater_annual_snr"].fillna(0)*reliable["insar_annual_snr"].fillna(0)
    chosen=[]
    conf=reliable[reliable.aquifer_type=="confined"].copy()
    if not conf.empty:
        chosen.append(conf.sort_values(["joint_snr","lag_ci_width_days"],ascending=[False,True]).iloc[0].well_id)
        remaining=conf[~conf.well_id.isin(chosen)]
        if not remaining.empty:
            chosen.append(remaining.sort_values("peak_lag_days",ascending=False).iloc[0].well_id)
    unconf=reliable[reliable.aquifer_type=="unconfined"].copy()
    no_reliable_unconfined=False
    if not unconf.empty:
        chosen.append(unconf.sort_values(["joint_snr","lag_ci_width_days"],ascending=[False,True]).iloc[0].well_id)
    else:
        no_reliable_unconfined=True
        fallback=conf[~conf.well_id.isin(chosen)]
        if not fallback.empty:
            chosen.append(fallback.sort_values(["joint_snr","lag_ci_width_days"],ascending=[False,True]).iloc[0].well_id)
    if len(chosen)<3:
        extra=reliable[~reliable.well_id.isin(chosen)].sort_values(["joint_snr","lag_ci_width_days"],ascending=[False,True])
        chosen.extend(extra.head(3-len(chosen)).well_id.tolist())
    return chosen[:3], no_reliable_unconfined


def _annual_harmonic_series(dates, values, period=365.2425):
    dates=pd.to_datetime(dates)
    vals=np.asarray(values,dtype=float)
    ok=np.isfinite(vals)
    if ok.sum() < 6:
        return None
    t=(dates-dates.min()).dt.total_seconds().to_numpy()/86400.0
    X=np.column_stack([np.ones_like(t),np.sin(2*np.pi*t/period),np.cos(2*np.pi*t/period)])
    beta=np.linalg.lstsq(X[ok],vals[ok],rcond=None)[0]
    grid=pd.date_range(dates.min(),dates.max(),periods=240)
    tg=(grid-dates.min()).total_seconds().to_numpy()/86400.0
    Xg=np.column_stack([np.ones_like(tg),np.sin(2*np.pi*tg/period),np.cos(2*np.pi*tg/period)])
    harmonic=Xg @ beta
    component=harmonic-np.nanmean(harmonic)
    amp=np.nanmax(np.abs(component))
    if not np.isfinite(amp) or amp == 0:
        amp=1.0
    return grid, component/amp


def _contribution_norm(paths):
    vals=[]
    for p in paths:
        if Path(p).exists():
            s=raster_summary(p)
            vals.extend([abs(s.get("p05",0)),abs(s.get("p95",0))])
    vmax=max([x for x in vals if np.isfinite(x)] or [1.0])
    return TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)


def _map_band_panel(ax,path,band,title,cmap=CMAP_SEQ,cbar_label=None,label=None,norm=None,vmin=None,vmax=None):
    import rasterio
    with rasterio.open(path) as src:
        scale=max(1,int(np.ceil(max(src.width,src.height)/900)))
        arr=src.read(band,out_shape=(max(1,src.height//scale),max(1,src.width//scale)),masked=True)
        extent=[src.bounds.left,src.bounds.right,src.bounds.bottom,src.bounds.top]
    image=ax.imshow(arr,extent=extent,origin="upper",cmap=cmap,norm=norm,vmin=vmin,vmax=vmax)
    ax.set_title(title,loc="left",pad=3);ax.set_xlabel("Longitude");ax.set_ylabel("Latitude");ax.tick_params(labelsize=6,length=2)
    if label: panel_label(ax,label)
    cb=plt.colorbar(image,ax=ax,fraction=.042,pad=.018)
    cb.ax.tick_params(labelsize=6,length=2)
    if cbar_label: cb.set_label(cbar_label,fontsize=6,labelpad=2)
    return image


def _band_summary(path):
    import rasterio
    rows=[]
    with rasterio.open(path) as src:
        for band in range(1,src.count+1):
            vals=[]
            for _,window in src.block_windows(band):
                arr=src.read(band,window=window,masked=True).compressed()
                if arr.size: vals.append(arr.astype(float))
            name=src.descriptions[band-1] or f"band_{band}"
            if vals:
                data=np.concatenate(vals)
                rows.append({"covariate":name,"unit":"standardized from metre source" if not name.startswith("extraction") else "category indicator",
                             "valid_pixels":int(data.size),"unique_values":int(np.unique(data).size),
                             "minimum":float(np.nanmin(data)),"median":float(np.nanmedian(data)),
                             "maximum":float(np.nanmax(data)),"std":float(np.nanstd(data)),
                             "IQR":float(np.nanpercentile(data,75)-np.nanpercentile(data,25)),
                             "source_layer":Path(path).name})
            else:
                rows.append({"covariate":name,"unit":"unknown","valid_pixels":0,"unique_values":0,"source_layer":Path(path).name})
    return pd.DataFrame(rows)


def _simple_map_figure(figure_id,title,raster_titles,out_dir=MAIN,limitations=None):
    missing=[OUTPUT/p for p,_ in raster_titles if not (OUTPUT/p).exists()]
    if missing:
        return not_generated(figure_id,title,missing,out_dir=out_dir)
    n=len(raster_titles);cols=min(3,n);rows=int(np.ceil(n/cols))
    fig,axes=plt.subplots(rows,cols,figsize=figure_size(WIDTH_DOUBLE_MM,.34*rows),squeeze=False,constrained_layout=True)
    labels=list("abcdefghijklmnopqrstuvwxyz")
    summaries=[]
    for i,(name,panel_title) in enumerate(raster_titles):
        ax=axes.flat[i]
        cmap=IDENT_CMAP if "identifiability" in name else (CMAP_DIV if "contribution" in name else (CMAP_LAG if "lag" in name else CMAP_SEQ))
        norm=IDENT_NORM if "identifiability" in name else None
        map_panel(ax,OUTPUT/name,panel_title,cmap=cmap,norm=norm,label=labels[i])
        summaries.append(raster_summary(OUTPUT/name))
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    source=write_source(pd.DataFrame(summaries),figure_id,"raster_summary")
    files=save_figure(fig,figure_id,out_dir)
    plt.close(fig)
    return caption_record(figure_id,title,"complete",files,[OUTPUT/p for p,_ in raster_titles],[source],
                          width_mm=WIDTH_DOUBLE_MM,limitations=limitations)


def figure_01():
    required=[OUTPUT/"well_summary.csv",OUTPUT/"insar_cube_manifest.json",OUTPUT/"insar_epochs.csv"]
    if not _exists(*required):
        return not_generated("Figure_01_StudyArea_Workflow","Study area, datasets, and workflow",required)
    wells=read_table(OUTPUT/"well_summary.csv")
    epochs=read_table(OUTPUT/"insar_epochs.csv")
    manifest=json.loads((OUTPUT/"insar_cube_manifest.json").read_text(encoding="utf-8"))
    fig=plt.figure(figsize=figure_size(WIDTH_DOUBLE_MM,.62),constrained_layout=True)
    gs=fig.add_gridspec(2,3)
    ax=fig.add_subplot(gs[:,0])
    try:
        import geopandas as gpd
        for shp,color,lw in [(ROOT/"../shp/华北平原市.shp","0.75",.5),(ROOT/"../shp/hengshui_county.shp","0.35",.8)]:
            if shp.exists():
                gpd.read_file(shp).to_crs("EPSG:4326").boundary.plot(ax=ax,color=color,linewidth=lw)
    except Exception:
        pass
    colors={"unconfined":"#66c2a5","confined":"#8da0cb"}
    for aquifer,group in wells.groupby("aquifer_type"):
        ax.scatter(group["lon"],group["lat"],s=9,label=aquifer,color=colors.get(aquifer,"0.4"),edgecolor="white",linewidth=.2)
    ax.set_title("Study area and wells",loc="left");ax.set_xlabel("Longitude");ax.set_ylabel("Latitude");ax.legend(loc="lower right")
    panel_label(ax,"a")
    ax=fig.add_subplot(gs[0,1])
    counts=wells["aquifer_type"].value_counts()
    ax.bar(counts.index,counts.values,color=[colors.get(x,"0.4") for x in counts.index]);ax.set_title("Groundwater wells",loc="left");ax.set_ylabel("count");panel_label(ax,"b")
    ax=fig.add_subplot(gs[0,2])
    dates=pd.to_datetime(epochs["date"])
    ax.vlines(dates,0,1,color="0.25",lw=.35);ax.set_yticks([]);ax.set_title("SAR epochs",loc="left");panel_label(ax,"c")
    ax=fig.add_subplot(gs[1,1:])
    steps=["Daily groundwater QC","True SAR dates","Harmonic lag","Global MAP","Laplace posterior","Seasonal elastic storage"]
    for i,step in enumerate(steps):
        ax.text(i,.5,step,ha="center",va="center",bbox={"boxstyle":"round,pad=.25","fc":"white","ec":"0.3","lw":.8})
        if i<len(steps)-1:ax.annotate("",xy=(i+.42,.5),xytext=(i+.58,.5),arrowprops={"arrowstyle":"->","lw":.8})
    ax.set_xlim(-.6,len(steps)-.4);ax.set_ylim(0,1);ax.axis("off");ax.set_title("Workflow",loc="left");panel_label(ax,"d")
    source=write_source(pd.DataFrame({"metric":["unconfined_wells","confined_wells","sar_epochs","groundwater_sampling"],
                                      "value":[int(counts.get("unconfined",0)),int(counts.get("confined",0)),len(epochs),"daily"]}),"Figure_01_StudyArea_Workflow","summary")
    files=save_figure(fig,"Figure_01_StudyArea_Workflow",MAIN);plt.close(fig)
    return caption_record("Figure_01_StudyArea_Workflow","Study area, datasets, and workflow of the InSAR-groundwater inversion framework",
                          "complete",files,required,[source])


def figure_02():
    required=[OUTPUT/"insar_mean_vertical_velocity_mm_yr.tif",OUTPUT/"insar_annual_amplitude_mm.tif",
              OUTPUT/"insar_annual_phase_days.tif",OUTPUT/"well_timeseries.csv",OUTPUT/"insar_epochs.csv"]
    if not _exists(*required):
        return not_generated("Figure_02_DataOverview","InSAR and groundwater data overview",required)
    diag_path=OUTPUT/"insar_overview_product_diagnostics.json"
    if not diag_path.exists():
        from scripts.check_insar_overview_products import run as check_overview
        diagnostics=check_overview(str(ROOT/"config.yaml"))
    else:
        diagnostics=json.loads(diag_path.read_text(encoding="utf-8"))
    products=diagnostics.get("products",{})
    finite_ok=all(products.get(k,{}).get("finite_pixel_count",0)>0 for k in
                  ["mean_vertical_velocity","annual_vertical_amplitude","annual_vertical_phase"])
    status="complete" if finite_ok else "invalid_input"
    fig,axes=plt.subplots(2,3,figsize=figure_size(WIDTH_DOUBLE_MM,.68),constrained_layout=True)
    map_panel(axes[0,0],required[0],"Mean vertical velocity",cmap=CMAP_DIV,label="a",cbar_label="mm yr$^{-1}$")
    map_panel(axes[0,1],required[1],"Annual vertical amplitude",cmap=CMAP_SEQ,label="b",cbar_label="mm")
    map_panel(axes[0,2],required[2],"Annual vertical phase",cmap=CMAP_PHASE,label="c",cbar_label="day of year",vmin=0,vmax=365.2425,robust=False)
    wells=read_table(required[3]);wells["date"]=pd.to_datetime(wells["date"])
    coverage=wells.groupby("well_id",as_index=False).agg(lon=("lon","first"),lat=("lat","first"),
                                                          n_observed=("is_observed","sum"),
                                                          aquifer_type=("aquifer_type","first"))
    colors={"confined":"#377eb8","unconfined":"#4daf4a"}
    for aq,g in coverage.groupby("aquifer_type"):
        axes[1,0].scatter(g.lon,g.lat,s=np.clip(g.n_observed/60,4,18),label=aq,color=colors.get(aq,"0.4"),alpha=.75,edgecolor="white",linewidth=.2)
    axes[1,0].set_title("Groundwater observation coverage",loc="left");axes[1,0].set_xlabel("Longitude");axes[1,0].set_ylabel("Latitude");axes[1,0].legend(loc="lower right",title="aquifer")
    panel_label(axes[1,0],"d")
    epochs=read_table(required[4]);dates=pd.to_datetime(epochs["date"])
    axes[1,1].vlines(dates,0,1,color="0.25",lw=.35)
    axes[1,1].set_ylim(0,1);axes[1,1].set_yticks([]);axes[1,1].set_title("SAR acquisition timeline",loc="left");_setup_two_year_axis(axes[1,1]);panel_label(axes[1,1],"e")
    intervals=dates.diff().dt.days.dropna()
    axes[1,2].hist(intervals,bins=np.arange(0,61,3),color="0.35")
    axes[1,2].set_title("SAR revisit interval distribution",loc="left");axes[1,2].set_xlabel("Revisit interval (days)");axes[1,2].set_ylabel("count");panel_label(axes[1,2],"f")
    source=write_source(pd.DataFrame([{"product":k,**v} for k,v in products.items()]),"Figure_02_DataOverview","finite_checks")
    files=save_figure(fig,"Figure_02_DataOverview",MAIN);plt.close(fig)
    return add_validation(caption_record("Figure_02_DataOverview","InSAR and groundwater data overview",status,files,required,[source],
                          limitations=["Mean velocity is estimated from the first and last real SAR cumulative-displacement rasters under the recorded vertical-dominant assumption."]),
                          finite_data_check={k:products.get(k,{}).get("finite_pixel_count",0) for k in products},
                          required_panel_count=6,panel_validation="six panels generated from real outputs")


def _standardize(values):
    arr=np.asarray(values,dtype=float)
    med=np.nanmedian(arr);std=np.nanstd(arr)
    if not np.isfinite(std) or std == 0:
        std=1.0
    return (arr-med)/std


def _harmonic_component(dates, values, period=365.2425):
    dates=pd.to_datetime(dates)
    vals=np.asarray(values,dtype=float)
    ok=np.isfinite(vals)
    if ok.sum() < 6:
        return None, None
    t=(dates-dates.min()).dt.total_seconds().to_numpy()/86400.0
    X=np.column_stack([np.ones_like(t),np.sin(2*np.pi*t/period),np.cos(2*np.pi*t/period)])
    beta=np.linalg.lstsq(X[ok],vals[ok],rcond=None)[0]
    grid=pd.date_range(dates.min(),dates.max(),periods=240)
    tg=(grid-dates.min()).total_seconds().to_numpy()/86400.0
    Xg=np.column_stack([np.ones_like(tg),np.sin(2*np.pi*tg/period),np.cos(2*np.pi*tg/period)])
    return grid, Xg @ beta


def figure_03():
    required=[OUTPUT/"lag_summary.csv",OUTPUT/"well_timeseries.csv",OUTPUT/"insar_at_wells.csv"]
    if not _exists(*required):
        return not_generated("Figure_03_RepresentativeLag","Representative well lag diagnostics",required)
    lag=read_table(OUTPUT/"lag_summary.csv")
    wells=read_table(OUTPUT/"well_timeseries.csv");insar=read_table(OUTPUT/"insar_at_wells.csv")
    chosen,no_unconf=_select_representative_wells(lag)
    if len(chosen)<3:
        return not_generated("Figure_03_RepresentativeLag","Representative well lag diagnostics",required,
                             recommended_validation=["At least three reliable wells with lag diagnostics"])
    fig,axes=plt.subplots(len(chosen),3,figsize=figure_size(WIDTH_DOUBLE_MM,.34*len(chosen)),constrained_layout=True,squeeze=False)
    rows=[]
    for r,well_id in enumerate(chosen):
        meta=lag[lag.well_id==well_id].iloc[0]
        gw=wells[wells.well_id==well_id].copy();ii=insar[insar.well_id==well_id].copy()
        gw["date"]=pd.to_datetime(gw["date"]);ii["date"]=pd.to_datetime(ii["date"])
        idates=pd.to_datetime(ii.date)
        ax=axes[r,0]
        obs=gw[gw.get("is_observed",True).astype(bool)]
        interp=gw[~gw.get("is_observed",True).astype(bool)]
        ax.plot(obs.date,obs.head_anomaly_m,lw=.45,color="#238b45",label="head observed")
        if not interp.empty:
            ax.plot(interp.date,interp.head_anomaly_m,lw=.35,color="#74c476",ls="--",alpha=.45,label="head interpolated")
        ax.axhline(0,color="0.75",lw=.6)
        ax.set_ylabel("Groundwater-head anomaly (m)")
        ax.set_title(f"{well_id} ({meta.aquifer_type})",loc="left")
        ax2=ax.twinx()
        ax2.plot(idates,ii.vertical_mm,lw=.5,marker="o",ms=1.8,color="#225ea8",alpha=.75,label="InSAR vertical")
        ax2.set_ylabel("InSAR vertical displacement (mm)")
        _setup_two_year_axis(ax)
        lines=ax.get_lines()+ax2.get_lines()
        labels=[line.get_label() for line in lines if not line.get_label().startswith("_")]
        lines=[line for line in lines if not line.get_label().startswith("_")]
        ax.legend(lines,labels,loc="upper right",fontsize=5,handlelength=1.2)

        gh=_annual_harmonic_series(gw.date,gw.head_anomaly_m)
        ih=_annual_harmonic_series(ii.date,ii.vertical_mm)
        axh=axes[r,1]
        if gh is not None:
            axh.plot(gh[0],gh[1],color="#238b45",lw=.9,label="groundwater")
            shifted_dates=gh[0]+pd.to_timedelta(float(meta.peak_lag_days),unit="D")
            axh.plot(shifted_dates,gh[1],color="#e6550d",lw=.9,ls="--",label="lag-shifted groundwater")
        if ih is not None:
            axh.plot(ih[0],ih[1],color="#225ea8",lw=.9,label="InSAR")
        axh.axhline(0,color="0.8",lw=.6)
        axh.set_ylabel("unit amplitude")
        axh.set_title("Annual harmonic components",loc="left")
        _setup_two_year_axis(axh)
        axh.legend(loc="upper right",fontsize=5,handlelength=1.2)

        low=float(meta.lag_ci_low); high=float(meta.lag_ci_high); peak=float(meta.peak_lag_days)
        axc=axes[r,2]
        axc.axvline(0,color="0.65",lw=.7,ls="--")
        ci=_ci_short_arc(peak,low,high)
        if ci is None:
            axc.text(.5,.5,"CI not estimable",transform=axc.transAxes,ha="center",va="center",fontsize=7)
        else:
            lo,hi,c=ci
            axc.errorbar(c,0,xerr=[[c-lo],[hi-c]],fmt="o",ms=4.2,mfc="k",mec="k",
                         ecolor="#d95f02",elinewidth=2.4,capsize=4.5,capthick=1.2,zorder=5)
            axc.plot(c,0,marker="o",ms=4,color="k")
        axc.set_xlim(-182.62125,182.62125);axc.set_ylim(-.65,.65)
        axc.set_yticks([])
        axc.set_xlabel("Signed lag (days)")
        axc.set_title("Signed lag and CI",loc="left")
        axc.text(.02,.08,"deformation leads",transform=axc.transAxes,ha="left",fontsize=5.5)
        axc.text(.98,.08,"groundwater leads",transform=axc.transAxes,ha="right",fontsize=5.5)
        label=(f"lag={_signed_lag(peak):.1f} d\nCI width={float(meta.lag_ci_width_days):.1f} d\n"
               f"GW SNR={float(meta.groundwater_annual_snr):.2g}; InSAR SNR={float(meta.insar_annual_snr):.2g}\n"
               f"p={float(meta.surrogate_p_value):.3f}; n={int(meta.n_pairs)}")
        axc.text(.03,.82,label,transform=axc.transAxes,fontsize=5.5,va="top",
                 bbox={"boxstyle":"round,pad=.18","fc":"white","ec":"0.8","lw":.4,"alpha":.9})
        rows.append(meta.to_dict())
    for i,ax in enumerate(axes.flat[:len(chosen)*3]):panel_label(ax,chr(97+i))
    source=write_source(pd.DataFrame(rows),"Figure_03_RepresentativeLag","selected_wells")
    files=save_figure(fig,"Figure_03_RepresentativeLag",MAIN);plt.close(fig)
    limitations=[] if not no_unconf else ["no reliable unconfined well available; third row uses a confined well fallback."]
    return add_validation(caption_record("Figure_03_RepresentativeLag","Representative dual-endpoint circular phase lag diagnostics","complete",files,required,[source],
                          limitations=limitations),
                          unit_check={"left_axis":"Groundwater-head anomaly (m)","right_axis":"InSAR vertical displacement (mm)"},
                          panel_validation="physical dual-axis time series, annual harmonic alignment, and circular CI interval panels",
                          required_panel_count=9)


def figure_04():
    required=[OUTPUT/"lag_summary.csv"]
    if not _exists(*required):
        return not_generated("Figure_04_LagStatistics","Lag statistics",required)
    lag=read_table(OUTPUT/"lag_summary.csv")
    lag["signed_lag_days"]=lag["peak_lag_days"].map(_signed_lag)
    fig,axes=plt.subplots(2,2,figsize=figure_size(WIDTH_DOUBLE_MM,.68),constrained_layout=True)
    reliable=lag[lag.get("lag_reliable",False).astype(bool)]
    groups=[reliable.loc[reliable.aquifer_type==a,"signed_lag_days"].dropna() for a in ["confined","unconfined"]]
    axes[0,0].boxplot(groups,tick_labels=["confined","unconfined"],showfliers=False)
    for i,g in enumerate(groups,1):
        if not g.empty:
            axes[0,0].scatter(np.full(len(g),i)+np.linspace(-.05,.05,len(g)),g,s=9,color="0.25",alpha=.55)
    axes[0,0].axhline(0,color="0.6",lw=.7);axes[0,0].set_ylim(-182.62125,182.62125)
    axes[0,0].set_ylabel("Signed lag (days)");axes[0,0].set_title("Reliable lag distribution by aquifer",loc="left")
    colors={"confined":"#377eb8","unconfined":"#984ea3"}
    markers={True:"o",False:"x"}
    for aq,g in lag.groupby("aquifer_type"):
        for reliable_flag,gg in g.groupby(lag.loc[g.index,"lag_reliable"].astype(bool)):
            axes[0,1].scatter(gg.groundwater_annual_snr,gg.signed_lag_days,s=18,marker=markers[reliable_flag],
                              color=colors.get(aq,"0.4"),alpha=.75,label=f"{aq}, {'reliable' if reliable_flag else 'unreliable'}")
    axes[0,1].axhline(0,color="0.6",lw=.7);axes[0,1].set_ylim(-182.62125,182.62125)
    axes[0,1].set_xlabel("Groundwater annual SNR");axes[0,1].set_ylabel("Signed lag (days)")
    axes[0,1].set_title("Lag vs groundwater annual SNR",loc="left");axes[0,1].legend(fontsize=5,loc="best",handlelength=1)
    for aq,g in lag.groupby("aquifer_type"):
        axes[1,0].scatter(g.lag_ci_width_days,g.signed_lag_days,s=18,color=colors.get(aq,"0.4"),alpha=.75,label=aq)
    axes[1,0].axhline(0,color="0.6",lw=.7);axes[1,0].axvline(90,color="#e6550d",lw=.8,ls="--",label="wide-CI threshold")
    axes[1,0].set_ylim(-182.62125,182.62125);axes[1,0].set_xlabel("Circular 95% CI width (days)")
    axes[1,0].set_ylabel("Signed lag (days)");axes[1,0].set_title("Lag vs circular CI width",loc="left");axes[1,0].legend(fontsize=5,loc="best")
    failures={
        "insufficient_pairs": lag.n_pairs.fillna(0)<24,
        "nonpositive_peak": lag.peak_correlation.fillna(0)<=0,
        "surrogate_not_significant": lag.surrogate_p_value.fillna(1)>=0.05,
        "low_groundwater_snr": lag.groundwater_annual_snr.fillna(0)<1,
        "low_insar_snr": lag.insar_annual_snr.fillna(0)<1,
        "wide_phase_error": (lag.groundwater_phase_std_days.fillna(999)>45)|(lag.insar_phase_std_days.fillna(999)>45),
        "wide_ci": lag.lag_ci_width_days.fillna(999)>90,
        "boundary_peak": lag.boundary_peak.fillna(False).astype(bool),
        "annual_alias": lag.annual_alias.fillna(False).astype(bool),
        "baseline_insufficient": np.zeros(len(lag),dtype=bool),
    }
    fail_counts={k:int((v & ~lag.lag_reliable.astype(bool)).sum()) for k,v in failures.items()}
    labels=list(fail_counts);vals=[fail_counts[k] for k in labels]
    axes[1,1].barh(np.arange(len(labels)),vals,color="#756bb1")
    axes[1,1].set_yticks(np.arange(len(labels)));axes[1,1].set_yticklabels(labels,fontsize=5.5)
    axes[1,1].set_xlabel("Failed well count");axes[1,1].set_title("Reliability failure reasons",loc="left")
    for i,ax in enumerate(axes.flat):panel_label(ax,chr(97+i))
    source=write_source(lag,"Figure_04_LagStatistics","lag_summary")
    files=save_figure(fig,"Figure_04_LagStatistics",MAIN);plt.close(fig)
    return add_validation(caption_record("Figure_04_LagStatistics","Well-scale annual lag statistics from dual-endpoint circular phase estimates","complete",files,required,[source]),
                          unit_check={"signed_lag_axis":"days","snr_axis":"dimensionless"},
                          panel_validation="four-panel lag statistics with aquifer color and reliability marker legends",
                          required_panel_count=4)


def figure_05():
    required=[OUTPUT/"latent_head_leave_well_out_validation.csv",OUTPUT/"well_timeseries.csv",OUTPUT/"latent_head_models.pkl"]
    if not _exists(*required):
        return not_generated("Figure_05_LatentHeadValidation","Latent head validation",required)
    val=read_table(required[0])
    summary=val.groupby(["scheme","rank"])["RMSE"].mean().unstack("scheme")
    selected_rank=4
    runner=int(summary["spatial_block"].drop(index=selected_rank,errors="ignore").idxmin())
    rank_payload={"candidate_ranks":[int(x) for x in summary.index],
                  "random_rmse":summary.get("random",pd.Series(dtype=float)).to_dict(),
                  "spatial_block_rmse":summary.get("spatial_block",pd.Series(dtype=float)).to_dict(),
                  "temporal_block_rmse":summary.get("temporal_block",pd.Series(dtype=float)).to_dict(),
                  "selected_rank":selected_rank,
                  "selection_rule":"prioritize spatial_block RMSE, then temporal_block RMSE; random validation is auxiliary",
                  "runner_up":runner,
                  "relative_difference":float((summary.loc[runner,"spatial_block"]-summary.loc[selected_rank,"spatial_block"])/summary.loc[selected_rank,"spatial_block"]) if runner in summary.index else np.nan}
    (OUTPUT/"latent_head_rank_selection.json").write_text(json.dumps(rank_payload,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    fig,axes=plt.subplots(2,2,figsize=figure_size(WIDTH_DOUBLE_MM,.68),constrained_layout=True)
    for scheme,ax,title in [("spatial_block",axes[0,0],"Spatial-block RMSE vs rank"),
                            ("temporal_block",axes[0,1],"Temporal-block RMSE vs rank")]:
        g=summary[scheme].dropna()
        ax.plot(g.index,g.values,marker="o",ms=3,color="#377eb8")
        ax.axvline(selected_rank,color="#e6550d",lw=.8,ls="--",label="selected rank=4")
        ax.set_xlabel("rank");ax.set_ylabel("RMSE (m)");ax.set_title(title,loc="left");ax.legend(fontsize=6)
    sel=val[val["rank"]==selected_rank]
    order=["spatial_block","temporal_block","random"]
    axes[1,0].boxplot([sel.loc[sel.scheme==s,"RMSE"].dropna() for s in order],tick_labels=order,showfliers=False)
    axes[1,0].set_ylabel("RMSE (m)");axes[1,0].set_title("Validation RMSE at selected rank",loc="left")
    import pickle
    with required[2].open("rb") as stream:
        models=pickle.load(stream)
    wells=read_table(required[1]);wells["date"]=pd.to_datetime(wells["date"])
    colors={"confined":"#377eb8","unconfined":"#4daf4a"}
    for aq in ["confined","unconfined"]:
        sub=wells[(wells.aquifer_type==aq)&(wells.is_observed.astype(bool))].copy()
        if sub.empty or aq not in models:
            continue
        counts=sub.groupby("well_id")["head_anomaly_m"].count().sort_values(ascending=False)
        wid=counts.index[0]
        g=sub[sub.well_id==wid].sort_values("date")
        dates=pd.date_range(g.date.min(),g.date.max(),freq="14D")
        coords=g[["lon","lat"]].iloc[[0]].to_numpy()
        pred=models[aq].predict(coords,dates)[0]
        obs14=g.set_index("date")["head_anomaly_m"].resample("14D").median()
        common=obs14.index.intersection(dates)
        rmse=float(np.sqrt(np.nanmean((obs14.loc[common].to_numpy()-pd.Series(pred,index=dates).loc[common].to_numpy())**2)))
        axes[1,1].plot(g.date,g.head_anomaly_m,color=colors[aq],alpha=.25,lw=.35)
        axes[1,1].plot(dates,pred,color=colors[aq],lw=.9,label=f"{wid} {aq} pred, RMSE={rmse:.2f} m")
    axes[1,1].axvspan(pd.Timestamp("2022-01-01"),pd.Timestamp("2022-12-31"),color="0.9",zorder=-2,label="holdout-like period")
    axes[1,1].set_title("Representative observed and predicted daily head",loc="left")
    axes[1,1].set_ylabel("Head anomaly (m)");_setup_two_year_axis(axes[1,1]);axes[1,1].legend(fontsize=5,loc="best")
    for i,ax in enumerate(axes.flat):panel_label(ax,chr(97+i))
    source=write_source(val,"Figure_05_LatentHeadValidation","validation")
    files=save_figure(fig,"Figure_05_LatentHeadValidation",MAIN);plt.close(fig)
    return add_validation(caption_record("Figure_05_LatentHeadValidation","Daily latent groundwater-head model validation","complete",files,required,[source,str(OUTPUT/"latent_head_rank_selection.json")]),
                          unit_check={"RMSE":"m","head_anomaly":"m"},
                          panel_validation="rank curves, selected-rank comparison, and representative observed-predicted time series",
                          required_panel_count=4)


def figure_06():
    raw_path=OUTPUT/"geological_raw_covariates.tif"
    required=[OUTPUT/"geological_model_covariates.tif",OUTPUT/"geological_contribution.tif",OUTPUT/"spatial_basis_contribution.tif"]
    if not _exists(*required):
        return not_generated("Figure_06_GeologySpatialBasis","Geological covariates and spatial basis",required)
    summary=_band_summary(required[0])
    summary_path=OUTPUT/"geological_covariate_summary.csv";summary.to_csv(summary_path,index=False)
    raw_available=raw_path.exists()
    near_cov=summary.loc[summary.covariate.isin(["clay_total_z","clay_confined_z","quaternary_thickness_z"]),"IQR"].fillna(0).eq(0).all()
    mode_b=bool(near_cov) and not raw_available
    fig,axes=plt.subplots(2,2 if mode_b else 3,figsize=figure_size(WIDTH_DOUBLE_MM,.62 if mode_b else .68),constrained_layout=True)
    axes=np.asarray(axes)
    norm=_contribution_norm([required[1],required[2]])
    if mode_b:
        map_panel(axes.flat[0],required[1],"Geological contribution to log(Ske)",cmap=CMAP_DIV,norm=norm,label="a",cbar_label="Contribution to log(Ske)")
        map_panel(axes.flat[1],required[2],"Spatial RBF contribution to log(Ske)",cmap=CMAP_DIV,norm=norm,label="b",cbar_label="Contribution to log(Ske)")
        ax=axes.flat[2]
        plot=summary[summary.covariate.isin(["clay_total","clay_confined","quaternary_thickness"])]
        ax.barh(plot.covariate.str.replace("_"," "),plot.IQR,color="0.45")
        ax.axvline(0,color="0.55",lw=.7)
        for ytick, value in enumerate(plot.IQR):
            ax.text(0.002,ytick,"IQR = 0",va="center",fontsize=6,color="0.25")
        ax.set_xlim(-0.005,0.05)
        ax.set_xlabel("IQR of standardized covariate");ax.set_title("Covariate spatial variation",loc="left");panel_label(ax,"c")
        zone=summary[summary.covariate.str.startswith("extraction")]
        if not zone.empty and int(zone.unique_values.iloc[0]) > 1:
            _map_band_panel(axes.flat[3],required[0],4,"Extraction-layer zone",cmap=ListedColormap(["#f0f0f0","#756bb1"]),cbar_label="category",label="d")
        else:
            axes.flat[3].axis("off");axes.flat[3].text(.05,.55,"Only one extraction-layer category is present\nin the modeled domain.",fontsize=7,va="center");panel_label(axes.flat[3],"d")
    else:
        source_path=raw_path if raw_available else required[0]
        bands=[5,6,7,8] if raw_available else [1,2,3,4]
        titles=["Total clay thickness","Confined-aquitard clay thickness","Q4 thickness","Extraction zone"]
        cblabels=["m","m","m","zone code"] if raw_available else ["standardized","standardized","standardized","dummy"]
        cmaps=[CMAP_SEQ,CMAP_SEQ,CMAP_SEQ,ListedColormap(["#f0f0f0","#756bb1"])]
        for i in range(4):
            _map_band_panel(axes.flat[i],source_path,bands[i],titles[i],cmap=cmaps[i],cbar_label=cblabels[i],label=chr(97+i))
        map_panel(axes.flat[4],required[1],"Geological contribution to log(Ske)",cmap=CMAP_DIV,norm=norm,label="e",cbar_label="Contribution to log(Ske)")
        map_panel(axes.flat[5],required[2],"Spatial RBF contribution to log(Ske)",cmap=CMAP_DIV,norm=norm,label="f",cbar_label="Contribution to log(Ske)")
    source=write_source(summary,"Figure_06_GeologySpatialBasis","geological_covariate_summary")
    files=save_figure(fig,"Figure_06_GeologySpatialBasis",MAIN);plt.close(fig)
    geo_stats=raster_summary(required[1])
    limitations=[]
    if geo_stats.get("valid_pixels",0)>0 and geo_stats.get("min")==geo_stats.get("max"):
        limitations.append("Geological contribution is constant in the valid preview.")
    limitations.append("Raw geological covariates are shown in physical units when the audited raw stack is available; formal inversion uses the standardized model-covariate stack.")
    rec=caption_record("Figure_06_GeologySpatialBasis","Geological covariates and spatial basis","complete",files,required+([raw_path] if raw_available else []),[source,str(summary_path)],
                          limitations=limitations)
    rec=add_validation(rec, unit_check={"raw_covariates":"m or category code","model_covariates":"standardized/dummy","contribution":"dimensionless contribution to log(Ske)"},
                       panel_validation="raw physical-unit geology maps plus contribution maps" if raw_available else "mode B: contribution maps, covariate variation summary, and extraction-zone diagnostic",
                       required_panel_count=4 if mode_b else 6,
                       nearly_constant_field_warning=summary.loc[summary.IQR.fillna(0).eq(0),"covariate"].tolist(),
                       source_field_check_passed=True)
    return rec


def figure_07():
    rasters=["Ske_MAP.tif","lag_c_MAP_days.tif","residual_rmse_mm.tif","geological_contribution.tif","spatial_basis_contribution.tif"]
    missing=[OUTPUT/p for p in rasters if not (OUTPUT/p).exists()]
    if missing:
        return not_generated("Figure_07_ParameterMaps","MAP inversion parameter fields",missing)
    fig,axes=plt.subplots(2,3,figsize=figure_size(WIDTH_DOUBLE_MM,.68),constrained_layout=True)
    ske_stats=raster_summary(OUTPUT/"Ske_MAP.tif");lag_stats=raster_summary(OUTPUT/"lag_c_MAP_days.tif")
    map_panel(axes.flat[0],OUTPUT/"Ske_MAP.tif","Ske MAP",cmap=CMAP_SEQ,
              norm=LogNorm(vmin=max(ske_stats["p05"],1e-12),vmax=max(ske_stats["p95"],ske_stats["p05"]*1.01)),
              label="a",cbar_label="Elastic skeletal storage coefficient",robust=False)
    map_panel(axes.flat[1],OUTPUT/"lag_c_MAP_days.tif","Confined lag MAP",cmap=CMAP_LAG,
              label="b",cbar_label="days")
    map_panel(axes.flat[2],OUTPUT/"residual_rmse_mm.tif","Residual RMSE",cmap=CMAP_SEQ,
              label="c",cbar_label="mm")
    norm=_contribution_norm([OUTPUT/"geological_contribution.tif",OUTPUT/"spatial_basis_contribution.tif"])
    map_panel(axes.flat[3],OUTPUT/"geological_contribution.tif","Geological contribution to log(Ske)",cmap=CMAP_DIV,norm=norm,label="d",cbar_label="dimensionless")
    map_panel(axes.flat[4],OUTPUT/"spatial_basis_contribution.tif","Spatial RBF contribution to log(Ske)",cmap=CMAP_DIV,norm=norm,label="e",cbar_label="dimensionless")
    ax=axes.flat[5]
    rows=[("Ske",ske_stats,"#377eb8","{:.4g}"),("lag_c",lag_stats,"#e6550d","{:.2f} d")]
    for yi,(name,stats,color,fmt) in zip([1,0],rows):
        lo,hi=stats["p05"],stats["p95"]
        med=stats["median"]
        denom=max(hi-lo,1e-12)
        medn=(med-lo)/denom
        ax.hlines(yi,0,1,color=color,lw=2.2)
        ax.plot(medn,yi,"o",color="k",ms=4)
        iqr=stats["p75"]-stats["p25"]
        ax.text(1.08,yi,f"median {fmt.format(med)}\nIQR {fmt.format(iqr)}",va="center",fontsize=5.5)
    ax.set_xlim(-.05,1.78);ax.set_ylim(-.55,1.55)
    ax.set_yticks([1,0]);ax.set_yticklabels(["Ske","lag_c"])
    ax.set_xlabel("Relative p05-p95 interval");ax.set_title("Parameter summary",loc="left")
    ax.spines["bottom"].set_visible(True);panel_label(ax,"f")
    summaries=[raster_summary(OUTPUT/p) for p in rasters]
    source=write_source(pd.DataFrame(summaries),"Figure_07_ParameterMaps","raster_summary")
    files=save_figure(fig,"Figure_07_ParameterMaps",MAIN);plt.close(fig)
    return add_validation(caption_record("Figure_07_ParameterMaps","MAP inversion parameter fields","complete",files,[OUTPUT/p for p in rasters],[source],
                          limitations=["Cu MAP and unconfined-lag MAP are removed from the main text because Cu is approximately constant/prior dominated and lag_u is approximately spatially uniform with zero identified area.",
                                       f"Ske median={ske_stats['median']:.6g}; lag_c median={lag_stats['median']:.4f} days; lag_c IQR={lag_stats['p75']-lag_stats['p25']:.4f} days; the main regional field is nearly spatially uniform except local deviations."]),
                          constant_field_check={"Cu":"removed_from_main_text","lag_u":"removed_from_main_text"},
                          unit_check={"Ske":"dimensionless coefficient","lag_c":"days","residual":"mm","contributions":"dimensionless contribution to log(Ske)"},
                          required_panel_count=6,panel_validation="Cu, lag_u, and model_variant maps excluded from main-text panels",
                          source_field_check_passed=True,
                          nearly_constant_field_warning=["lag_c field has very small IQR; contribution maps include discrete or weak-amplitude fields"])


def figure_08():
    rasters=["Ske_relative_ci95_width_screened.tif","logSke_posterior_std.tif","lag_c_ci95_width_days.tif",
             "Ske_identifiability.tif","lag_c_identifiability.tif"]
    missing=[OUTPUT/p for p in rasters if not (OUTPUT/p).exists()]
    if missing:
        return not_generated("Figure_08_UncertaintyIdentifiability","Screened posterior uncertainty and identifiability",missing)
    artifact_path=OUTPUT/"rbf_artifact_diagnostics.json"
    if not artifact_path.exists():
        try:
            from scripts.diagnose_rbf_artifacts import run as diagnose_artifacts
            diagnose_artifacts(str(ROOT/"config.yaml"))
        except Exception:
            pass
    fig,axes=plt.subplots(2,3,figsize=figure_size(WIDTH_DOUBLE_MM,.68),constrained_layout=True)
    map_panel(axes.flat[0],OUTPUT/"Ske_relative_ci95_width_screened.tif","Ske relative 95% CI width",cmap=CMAP_SEQ,label="a",cbar_label="relative width")
    map_panel(axes.flat[1],OUTPUT/"logSke_posterior_std.tif","log(Ske) posterior SD",cmap=CMAP_SEQ,label="b",cbar_label="SD")
    map_panel(axes.flat[2],OUTPUT/"lag_c_ci95_width_days.tif","Confined lag circular 95% CI width",cmap=CMAP_LAG,label="c",cbar_label="days")
    area_path=OUTPUT/"identifiability_area_summary.csv"
    if area_path.exists():
        area=pd.read_csv(area_path)
    else:
        area=pd.DataFrame()
    ax=axes.flat[3]
    if not area.empty:
        subset=area[area.parameter.isin(["Ske","lag_c","Cu","lag_u","combined_deformation"])]
        labels=[{"combined_deformation":"combined"}.get(x,x) for x in subset.parameter]
        ax.bar(labels,100*subset.identified_area_fraction,color=["#377eb8","#4daf4a","#bdbdbd","#bdbdbd","#756bb1"][:len(subset)])
    ax.set_ylim(0,105);ax.set_ylabel("Identified area fraction (%)");ax.set_title("Identifiable area fraction by parameter",loc="left")
    ax.tick_params(axis="x",rotation=25,labelsize=5.5);panel_label(ax,"d")
    map_panel(axes.flat[4],OUTPUT/"Ske_identifiability.tif","Ske identifiability",cmap=IDENT_CMAP,norm=IDENT_NORM,label="e",cbar_label="class",robust=False)
    map_panel(axes.flat[5],OUTPUT/"lag_c_identifiability.tif","lag_c identifiability",cmap=IDENT_CMAP,norm=IDENT_NORM,label="f",cbar_label="class",robust=False)
    summaries=[raster_summary(OUTPUT/p) for p in rasters]
    ci_ok=True
    for low,med,high in [("Ske_ci95_low_screened.tif","Ske_posterior_median_screened.tif","Ske_ci95_high_screened.tif")]:
        if not all((OUTPUT/p).exists() for p in [low,med,high]):
            ci_ok=False
    screening=json.loads((OUTPUT/"posterior_draw_screening_summary.json").read_text(encoding="utf-8")) if (OUTPUT/"posterior_draw_screening_summary.json").exists() else {}
    provenance={"posterior_type":"physically_screened","draw_count":screening.get("n_accepted"),
                "screening_bounds":screening,"source_coefficients":"outputs/posterior_coefficients.npz",
                "quantile_method":"empirical 2.5/50/97.5 percentiles of accepted draws",
                "scientific_status":{"Ske":"formal_screened_posterior","lag_c":"formal_screened_posterior",
                                     "Cu":"not_identifiable; relative CI width median approximately 4.5",
                                     "lag_u":"not_identifiable; identified area fraction 0"}}
    (OUTPUT/"posterior_product_provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    source=write_source(pd.DataFrame(summaries),"Figure_08_UncertaintyIdentifiability","screened_raster_summary")
    files=save_figure(fig,"Figure_08_UncertaintyIdentifiability",MAIN);plt.close(fig)
    artifact_warning=["Localized uncertainty highs coincide with RBF support centers and are interpreted as model-structure display artifacts rather than hydrogeological boundaries."]
    return add_validation(caption_record("Figure_08_UncertaintyIdentifiability","Screened posterior uncertainty and coefficient identifiability","complete",files,[OUTPUT/p for p in rasters],[source,str(OUTPUT/"posterior_product_provenance.json"),str(artifact_path)],
                          limitations=["Raw transformed Laplace products are diagnostic only; screened posterior products are the formal mapped uncertainty outputs.",
                                       "Cu and lag_u are not identifiable and are not shown as main-text continuous posterior maps.",
                                       artifact_warning[0]]),
                          posterior_type_check={"formal_maps":"physically_screened","ci_ordering_products_present":ci_ok},
                          constant_field_check={"Cu":"not_identifiable","lag_u":"not_identifiable"},
                          required_panel_count=6,panel_validation="screened Ske/lag_c uncertainty plus identifiable-area summaries",
                          source_field_check_passed=True,
                          artifact_warning=artifact_warning,
                          display_only_smoothing_used=False)


def figure_09():
    required=[OUTPUT/"storage_harmonic_posterior_timeseries.csv",OUTPUT/"storage_harmonic_map_timeseries.csv"]
    if not _exists(*required):
        return not_generated("Figure_09_SeasonalElasticStorage","Seasonal elastic groundwater-storage change",required)
    post=read_table(OUTPUT/"storage_harmonic_posterior_timeseries.csv");post["date"]=pd.to_datetime(post["date"])
    map_ts=read_table(OUTPUT/"storage_harmonic_map_timeseries.csv");map_ts["date"]=pd.to_datetime(map_ts["date"])
    fig,axes=plt.subplots(2,3,figsize=figure_size(WIDTH_DOUBLE_MM,.68),constrained_layout=True)
    colors={"low":"#8da0cb","medium":"#66c2a5","high":"#fc8d62"}
    for ax in axes.flat[:3]:ax.axhline(0,color="0.3",lw=.7)
    axes[0,0].set_title("Confined elastic storage",loc="left")
    g=post[(post.region=="confined_identified")&(post.specific_yield_scenario=="medium")&(post.posterior_type=="physically_screened")]
    if not g.empty:
        axes[0,0].fill_between(g.date,g.confined_ci95_low_m3/1e8,g.confined_ci95_high_m3/1e8,color="#8da0cb",alpha=.18,linewidth=0);axes[0,0].plot(g.date,g.confined_median_m3/1e8,color="#377eb8",lw=.9)
    gm=map_ts[map_ts.specific_yield_scenario=="medium"]
    if not gm.empty:
        axes[0,0].plot(gm.date,gm.confined_elastic_storage_change_m3/1e8,color="0.15",lw=.65,ls="--",label="MAP")
        axes[0,0].legend(fontsize=5,loc="upper right")
    axes[0,1].set_title("Unconfined Sy scenarios",loc="left")
    for scenario,color in colors.items():
        g=post[(post.region=="unconfined_scenario_valid")&(post.specific_yield_scenario==scenario)&(post.posterior_type=="physically_screened")]
        if not g.empty:axes[0,1].plot(g.date,g.unconfined_median_m3/1e8,color=color,lw=.9,label=scenario)
    axes[0,1].legend(fontsize=5,loc="upper right",title="Sy")
    axes[0,2].set_title("Total seasonal elastic storage",loc="left")
    for scenario,color in colors.items():
        g=post[(post.region=="joint_storage_valid")&(post.specific_yield_scenario==scenario)&(post.posterior_type=="physically_screened")]
        if not g.empty and g.total_median_m3.notna().any():
            if scenario=="medium":
                axes[0,2].fill_between(g.date,g.total_ci95_low_m3/1e8,g.total_ci95_high_m3/1e8,color=color,alpha=.14,linewidth=0)
            axes[0,2].plot(g.date,g.total_median_m3/1e8,color=color,lw=.85,label=scenario)
    axes[0,2].legend(ncol=3,loc="upper center",bbox_to_anchor=(.5,1.18),handlelength=1.3,columnspacing=.9)
    width_rows=[]
    for ptype in ["raw_laplace","physically_screened","sensitivity_ske_max_0.05","sensitivity_ske_max_1"]:
        g=post[(post.region=="joint_storage_valid")&(post.specific_yield_scenario=="medium")&(post.posterior_type==ptype)]
        width=np.nanmean((g.total_ci95_high_m3-g.total_ci95_low_m3)/1e8) if not g.empty else np.nan
        width_rows.append({"posterior_type":ptype,"mean_total_ci95_width_1e8_m3":width})
    wr=pd.DataFrame(width_rows)
    short_width_labels=["raw","screened","Smax\n0.05","Smax\n1"]
    axes[1,0].bar(short_width_labels,wr.mean_total_ci95_width_1e8_m3,color=["0.6","#d95f02","#74c476","#238b45"])
    axes[1,0].set_yscale("log");axes[1,0].set_ylabel("Mean total CI95 width (10$^8$ m$^3$)")
    axes[1,0].set_title("Posterior interval width diagnostic",loc="left");axes[1,0].tick_params(axis="x",rotation=20,labelsize=5.5)
    area=(post[post.posterior_type=="physically_screened"].drop_duplicates("region")[["region","valid_area_km2","identified_area_km2"]]
          .sort_values("valid_area_km2",ascending=False))
    x=np.arange(len(area))
    unidentified=area.valid_area_km2-area.identified_area_km2
    axes[1,1].bar(x,unidentified,color="#756bb1")
    short_labels={"all_valid":"all","confined_identified":"conf.","unconfined_scenario_valid":"unconf.","joint_storage_valid":"joint"}
    axes[1,1].set_xticks(x);axes[1,1].set_xticklabels([short_labels.get(str(v),str(v)) for v in area.region],rotation=0,ha="center")
    axes[1,1].set_title("Unidentified area",loc="left")
    amp_labels=[];amp_vals=[];amp_cols=[]
    base=post[(post.region=="joint_storage_valid")&(post.posterior_type=="physically_screened")]
    gmed=base[base.specific_yield_scenario=="medium"]
    if not gmed.empty:
            amp_labels.append("conf.");amp_vals.append((gmed.confined_median_m3.max()-gmed.confined_median_m3.min())/1e8);amp_cols.append("#377eb8")
    for scenario,color in colors.items():
        g=base[base.specific_yield_scenario==scenario]
        if not g.empty:
            amp_labels.append(f"u-{scenario}");amp_vals.append((g.unconfined_median_m3.max()-g.unconfined_median_m3.min())/1e8);amp_cols.append(color)
    for scenario,color in colors.items():
        g=base[base.specific_yield_scenario==scenario]
        if not g.empty:
            amp_labels.append(f"t-{scenario}");amp_vals.append((g.total_median_m3.max()-g.total_median_m3.min())/1e8);amp_cols.append(color)
    axes[1,2].bar(amp_labels,amp_vals,color=amp_cols)
    axes[1,2].set_title("Seasonal amplitude",loc="left")
    for ax in axes.flat:
        if ax.has_data():
            ax.set_ylabel("10$^8$ m$^3$")
            ax.tick_params(axis="x",labelrotation=0,labelsize=6)
            if ax in axes.flat[:3]:
                _setup_two_year_axis(ax)
    axes[1,1].set_ylabel("area (km$^2$)")
    axes[1,2].tick_params(axis="x",labelsize=5.5,rotation=25)
    for i,ax in enumerate(axes.flat):panel_label(ax,chr(97+i))
    source=write_source(post,"Figure_09_SeasonalElasticStorage","storage_posterior")
    width_source=write_source(wr,"Figure_09_SeasonalElasticStorage","interval_width_summary")
    files=save_figure(fig,"Figure_09_SeasonalElasticStorage",MAIN);plt.close(fig)
    return add_validation(caption_record("Figure_09_SeasonalElasticStorage","Seasonal elastic groundwater-storage change with posterior intervals","complete",files,required,[source,width_source],
                          limitations=["This is seasonal elastic groundwater-storage change, not groundwater depletion, groundwater loss, or irreversible storage loss.",
                                       "The unconfined component is controlled by configured Sy scenarios.",
                                       "The screened posterior is conditional on configured physical bounds."]),
                          posterior_type_check={"formal_storage":"physically_screened"},
                          panel_validation="confined storage, Sy-scenario unconfined storage, total storage, log CI-width bars, unidentified area, and seven amplitude bars",
                          required_panel_count=6)


def figure_10():
    missing=[OUTPUT/"gnss_validation.csv",OUTPUT/"leveling_validation.csv"]
    return not_generated("Figure_10_IndependentValidation","Independent validation",missing,
                         required_fields=["station_id","date","observed_displacement","insar_prediction"],
                         recommended_validation=["GNSS LOS vs InSAR LOS","leveling vs InSAR vertical","observed vs predicted scatter"])


def supplementary():
    specs=[
        ("Figure_S01_WellDataQuality","Well data quality",OUTPUT/"well_summary.csv"),
        ("Figure_S02_AllWellHarmonics","All well harmonics",OUTPUT/"well_harmonic_decomposition.csv"),
        ("Figure_S03_AllReliableLagSpectra","All reliable lag spectra",OUTPUT/"tlcc_spectra.csv"),
        ("Figure_S04_GeologicalCovariates","Geological covariates",OUTPUT/"geological_model_covariates.tif"),
        ("Figure_S05_MAPDiagnostics","MAP diagnostics",OUTPUT/"map_diagnostics.json"),
        ("Figure_S06_PosteriorCoefficientDiagnostics","Posterior coefficient diagnostics",OUTPUT/"posterior_coefficient_summary.csv"),
        ("Figure_S07_StoragePosteriorDiagnostics","Storage posterior diagnostics",OUTPUT/"posterior_draw_screening_summary.json"),
        ("Figure_S08_ModelComparison","Model comparison",OUTPUT/"model_comparison.csv"),
        ("Figure_S09_ReferenceAndGeometry","Reference and geometry",OUTPUT/"insar_cube_manifest.json"),
        ("Figure_S10_BulletinComparison","Bulletin comparison",OUTPUT/"bulletin_standardized.csv"),
    ]
    records=[]
    for figure_id,title,path in specs:
        if not path.exists():
            records.append(not_generated(figure_id,title,[path],out_dir=SUPP));continue
        fig,ax=plt.subplots(figsize=figure_size(WIDTH_SINGLE_MM,.75),constrained_layout=True)
        if path.suffix.lower()==".tif":
            map_panel(ax,path,title,label="a")
            source=write_source(pd.DataFrame([raster_summary(path)]),figure_id,"summary")
        elif path.suffix.lower()==".json":
            payload=json.loads(path.read_text(encoding="utf-8"))
            ax.axis("off");ax.text(.02,.95,json.dumps(payload,ensure_ascii=False,indent=2)[:1200],va="top",fontsize=6)
            source=write_source(pd.DataFrame([{"json_file":str(path)}]),figure_id,"source")
        else:
            df=read_table(path);numeric=df.select_dtypes(include=[np.number])
            if numeric.empty:
                ax.axis("off");ax.text(.05,.7,f"{path.name}\n{len(df)} rows",fontsize=8)
            else:
                numeric.iloc[:,:min(4,numeric.shape[1])].hist(ax=ax,bins=20)
            source=write_source(df.head(5000),figure_id,"source")
        files=save_figure(fig,figure_id,SUPP,WIDTH_SINGLE_MM);plt.close(fig)
        records.append(caption_record(figure_id,title,"complete",files,[path],[source],width_mm=WIDTH_SINGLE_MM))
    return records


def write_captions(records):
    en=[];zh=[]
    for rec in records:
        en.append(f"### {rec['figure_id']}. {rec['title']}\nStatus: {rec['status']}. This figure is generated only from recorded project outputs and linked source files. Limitations: {', '.join(rec.get('scientific_limitations') or ['none'])}.\n")
        zh.append(f"### {rec['figure_id']}：{rec['title']}\n状态：{rec['status']}。本图仅基于项目既有真实输出和记录源文件生成。限制：{', '.join(rec.get('scientific_limitations') or ['无'])}。\n")
    (FIGROOT/"figure_captions_en.md").write_text("\n".join(en),encoding="utf-8")
    (FIGROOT/"figure_captions_zh.md").write_text("\n".join(zh),encoding="utf-8")


def configure_output_root(output_root):
    if output_root is None:
        return
    import plotting.common as common
    root=Path(output_root)
    globals()["OUTPUT"]=root
    globals()["FIGROOT"]=root/"figures"
    globals()["MAIN"]=root/"figures"/"main"
    globals()["SUPP"]=root/"figures"/"supplementary"
    globals()["SOURCE"]=root/"figures"/"source_data"
    common.OUTPUT=globals()["OUTPUT"]
    common.FIGROOT=globals()["FIGROOT"]
    common.MAIN=globals()["MAIN"]
    common.SUPP=globals()["SUPP"]
    common.SOURCE=globals()["SOURCE"]


def main(config_path="config.yaml",output_root=None):
    configure_output_root(output_root)
    ensure_dirs()
    records=[figure_01(),figure_02(),figure_03(),figure_04(),figure_05(),figure_06(),figure_07(),figure_08(),figure_09(),figure_10()]
    records.extend(supplementary())
    manifest={"generated_at":pd.Timestamp.utcnow().isoformat(),"figures":records}
    (FIGROOT/"figure_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    write_captions(records)
    print(FIGROOT/"figure_manifest.json")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="config.yaml")
    parser.add_argument("--output-root",default=None)
    args=parser.parse_args()
    main(args.config,args.output_root)

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
FIGROOT = OUTPUT / "figures"
MAIN = FIGROOT / "main"
SUPP = FIGROOT / "supplementary"
SOURCE = FIGROOT / "source_data"

WIDTH_SINGLE_MM = 89
WIDTH_DOUBLE_MM = 180
DPI = 600


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "path",
    "pdf.fonttype": 42,
    "font.size": 6.8,
    "axes.linewidth": .65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": .6,
    "ytick.major.width": .6,
    "lines.linewidth": .9,
})

CMAP_SEQ = "viridis"
CMAP_DIV = "coolwarm"
CMAP_LAG = "magma"
CMAP_PHASE = "twilight"
IDENT_CMAP = ListedColormap(["#f0f0f0", "#fdae61", "#abd9e9", "#2c7bb6"])
IDENT_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], IDENT_CMAP.N)


def ensure_dirs():
    for path in (MAIN, SUPP, SOURCE):
        path.mkdir(parents=True, exist_ok=True)


def mm_to_in(mm):
    return mm / 25.4


def figure_size(width_mm=WIDTH_DOUBLE_MM, ratio=.62):
    width=mm_to_in(width_mm)
    return width, width*ratio


def read_table(path):
    path=Path(path)
    if path.with_suffix(".parquet").exists():
        try:
            return pd.read_parquet(path.with_suffix(".parquet"))
        except Exception:
            pass
    return pd.read_csv(path)


def read_raster_preview(path,max_size=900,resampling=Resampling.nearest):
    with rasterio.open(path) as src:
        scale=max(1,int(np.ceil(max(src.width,src.height)/max_size)))
        arr=src.read(1,out_shape=(max(1,src.height//scale),max(1,src.width//scale)),
                     masked=True,resampling=resampling)
        if np.ma.count(arr) == 0 and scale > 1:
            arr=src.read(1,masked=True)
        extent=[src.bounds.left,src.bounds.right,src.bounds.bottom,src.bounds.top]
        desc=src.descriptions[0] if src.descriptions else None
    return arr,extent,desc


def raster_summary(path):
    with rasterio.open(path) as src:
        vals=[]
        for _,window in src.block_windows(1):
            arr=src.read(1,window=window,masked=True).compressed()
            if arr.size:
                vals.append(arr.astype(float))
    if not vals:
        return {"source":Path(path).name,"valid_pixels":0}
    data=np.concatenate(vals)
    return {"source":Path(path).name,"valid_pixels":int(data.size),"mean":float(np.nanmean(data)),
            "median":float(np.nanmedian(data)),"p05":float(np.nanpercentile(data,5)),
            "p25":float(np.nanpercentile(data,25)),"p75":float(np.nanpercentile(data,75)),
            "p95":float(np.nanpercentile(data,95)),"min":float(np.nanmin(data)),"max":float(np.nanmax(data))}


def raster_percentiles(path, qs=(2, 98), max_values=2_000_000):
    vals = []
    count = 0
    with rasterio.open(path) as src:
        for _, window in src.block_windows(1):
            arr = src.read(1, window=window, masked=True).compressed()
            if arr.size:
                vals.append(arr.astype(float))
                count += arr.size
                if count >= max_values:
                    break
    if not vals:
        return [np.nan for _ in qs]
    data = np.concatenate(vals)
    return [float(x) for x in np.nanpercentile(data, qs)]


def map_panel(ax,path,title,cmap=CMAP_SEQ,norm=None,vmin=None,vmax=None,cbar=True,label=None,
              cbar_label=None, robust=True, extend="neither"):
    arr,extent,_=read_raster_preview(path)
    if norm is None and robust and (vmin is None or vmax is None):
        lo, hi = raster_percentiles(path)
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax
            extend = "both"
    image=ax.imshow(arr,extent=extent,origin="upper",cmap=cmap,norm=norm,vmin=vmin,vmax=vmax)
    ax.set_title(title,loc="left",pad=3)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.tick_params(labelsize=6, length=2)
    add_north_arrow(ax)
    add_scale_bar(ax)
    if label:
        panel_label(ax,label)
    if cbar:
        cb = plt.colorbar(image,ax=ax,fraction=.042,pad=.018,extend=extend)
        cb.ax.tick_params(labelsize=6, length=2)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=6, labelpad=2)
    return image


def panel_label(ax,label):
    ax.text(-.14,1.08,label,transform=ax.transAxes,fontsize=8,fontweight="bold",
            va="bottom",ha="left",clip_on=False)


def add_north_arrow(ax):
    ax.annotate("N",xy=(.94,.91),xytext=(.94,.80),xycoords="axes fraction",
                arrowprops={"arrowstyle":"-|>","lw":.65},ha="center",va="center",fontsize=6)


def add_scale_bar(ax):
    x0,x1=ax.get_xlim();y0,y1=ax.get_ylim()
    mid_lat=(y0+y1)/2
    km_per_degree_lon=max(1e-6,111.32*np.cos(np.deg2rad(mid_lat)))
    target_km=(x1-x0)*km_per_degree_lon*.18
    nice=np.array([5,10,20,30,50,75,100,150,200],dtype=float)
    km=float(nice[np.argmin(np.abs(nice-target_km))])
    length=km/km_per_degree_lon
    y=y0+(y1-y0)*.06;x=x0+(x1-x0)*.07
    ax.plot([x,x+length],[y,y],color="k",lw=1.2)
    ax.text(x+length/2,y+(y1-y0)*.022,f"{km:g} km",ha="center",va="bottom",fontsize=5.5)


def save_figure(fig,figure_id,out_dir=MAIN,width_mm=WIDTH_DOUBLE_MM):
    out_dir.mkdir(parents=True,exist_ok=True)
    stem=out_dir/figure_id
    fig.savefig(stem.with_suffix(".svg"),bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"),dpi=DPI,bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"),dpi=DPI,bbox_inches="tight",pil_kwargs={"compression":"tiff_lzw"})
    return {ext:str(stem.with_suffix(ext)) for ext in (".svg",".pdf",".png",".tiff")}


def not_generated(figure_id,title,missing_inputs,required_fields=None,recommended_validation=None,out_dir=MAIN):
    out_dir.mkdir(parents=True,exist_ok=True)
    payload={"figure_id":figure_id,"title":title,"status":"not_generated",
             "missing_inputs":[str(x) for x in missing_inputs],
             "required_fields":required_fields or [],
             "recommended_validation":recommended_validation or []}
    path=out_dir/f"{figure_id}_NOT_GENERATED.json"
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    return payload | {"files":{".json":str(path)}}


def write_source(frame,figure_id,name="source"):
    SOURCE.mkdir(parents=True,exist_ok=True)
    path=SOURCE/f"{figure_id}_{name}.csv"
    frame.to_csv(path,index=False)
    return str(path)


def caption_record(figure_id,title,status,files,source_files=None,source_data_files=None,
                   code_file=None,width_mm=WIDTH_DOUBLE_MM,dpi=DPI,missing_inputs=None,limitations=None):
    return {"figure_id":figure_id,"title":title,"status":status,"files":files,
            "source_files":[str(x) for x in (source_files or [])],
            "source_data_files":[str(x) for x in (source_data_files or [])],
            "generated_at":pd.Timestamp.utcnow().isoformat(),
            "code_file":code_file or "plotting/make_all_figures.py",
            "width_mm":width_mm,"dpi":dpi,
            "missing_inputs":[str(x) for x in (missing_inputs or [])],
            "scientific_limitations":limitations or [],
            "style_template":"nature_like",
            "source_field_check_passed":None,
            "nearly_constant_field_warning":[],
            "artifact_warning":[],
            "display_only_smoothing_used":False}


def add_validation(record, **checks):
    record.update(checks)
    return record

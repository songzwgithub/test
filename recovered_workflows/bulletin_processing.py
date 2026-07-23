"""Verified bulletin tables; free-text parsing is audit-only and never inversion input."""
from __future__ import annotations
from pathlib import Path
import re,numpy as np,pandas as pd

def load_verified_bulletin(path):
    frame=pd.read_csv(path,skipinitialspace=True);required={"year","metric","value","unit","observation_system","source_status"}
    if required-set(frame):raise ValueError(f"Verified bulletin columns missing: {sorted(required-set(frame))}")
    if not frame.source_status.eq("verified").all():raise ValueError("Only manually verified bulletin rows may constrain inversion")
    return frame

def parse_source_for_audit(path):
    text=Path(path).read_text(encoding="utf-8");rows=[]
    for year in range(2010,2021):
        match=re.search(rf"{year}年(.*?)(?={year+1}年|$)",text,re.S);section=match.group(1) if match else ""
        line=next((x for x in section.splitlines() if "衡水" in x and "深层地下水平均埋深" in x),"")
        if year==2010:m=re.search(r"平均埋深\s*(\d+(?:\.\d+)?)",line);value=float(m.group(1)) if m else np.nan
        elif year in (2011,2012):m=re.search(r"衡水\s*(\d+(?:\.\d+)?)",line);value=float(m.group(1)) if m else np.nan
        else:
            before=line.split("与上年")[0];numbers=[float(x) for x in re.findall(r"\d+(?:\.\d+)?",before)];value=numbers[-1] if year in (2016,2017,2018,2019,2020) else (numbers[-2] if len(numbers)>=2 else np.nan)
        rows.append({"year":year,"metric":"city_deep_mean_depth","value":value})
        center=next((x for x in section.splitlines() if "中心" in x and "埋深" in x and ("漏斗" in x)),"")
        m=re.search(r"中心埋深\s*(\d+(?:\.\d+)?)",center);rows.append({"year":year,"metric":"funnel_center_depth","value":float(m.group(1)) if m else np.nan})
        if year==2020:
            auto=next((x for x in section.splitlines() if "自动监测站" in x),"");m=re.search(r"中心水位\s*(-?\d+(?:\.\d+)?)",auto)
            rows.append({"year":2020,"metric":"funnel_center_head","value":float(m.group(1)) if m else np.nan})
    return pd.DataFrame(rows)

def validate_parser(source_path,verified_path,tolerance=1e-6):
    parsed=parse_source_for_audit(source_path);verified=load_verified_bulletin(verified_path)
    expected=verified[verified.year.le(2020)][["year","metric","value"]];comparison=expected.merge(parsed,on=["year","metric"],how="left",suffixes=("_verified","_parsed"))
    comparison["difference"]=comparison.value_parsed-comparison.value_verified
    conflicts=comparison[comparison.difference.abs().gt(tolerance)|comparison.value_parsed.isna()].copy()
    return comparison,conflicts

def prepare_bulletin_constraints(source_path,verified_path,output_dir):
    from io_utils import ensure_dir,write_table
    output=ensure_dir(output_dir);verified=load_verified_bulletin(verified_path);comparison,parser_conflicts=validate_parser(source_path,verified_path)
    write_table(verified,output/"bulletin_standardized.csv");write_table(parser_conflicts,output/"bulletin_parser_conflicts.csv")
    conflicts=[]
    for metric,group in verified.sort_values("year").groupby("metric"):
        if any(token in metric for token in ("change","delta","变化")):
            continue
        previous=None
        for row in group.itertuples():
            if previous is not None:
                calculated=float(row.value)-float(previous.value)
                change_rows=verified[(verified.year.eq(row.year))&(verified.metric.isin([metric+"_change",metric+"_delta",metric+"_annual_change"]))]
                for change in change_rows.itertuples():
                    conflicts.append({"year":int(row.year),"metric":metric,"reported_change":float(change.value),
                                      "calculated_change":calculated,"difference":float(change.value)-calculated,
                                      "source_metric":change.metric})
            previous=row
    conflict_frame=pd.DataFrame(conflicts,columns=["year","metric","reported_change","calculated_change","difference","source_metric"])
    write_table(conflict_frame,output/"bulletin_source_conflicts.csv")
    if not parser_conflicts.empty:raise RuntimeError("Bulletin parser disagrees with manually verified table")
    city=verified[verified.metric.eq("city_deep_mean_depth")].copy();baseline=city.iloc[0].value;city["delta_h_m"]=-(city.value-baseline)
    return verified,city

parse_bulletin=lambda path: {"verified_required":pd.DataFrame(),"city_deep":parse_source_for_audit(path).query("metric == 'city_deep_mean_depth'")}

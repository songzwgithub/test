"""Small, dependency-light I/O utilities for the inversion project."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import subprocess
import uuid
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def load_config(path="config.yaml"):
    """Load the JSON-compatible YAML configuration and resolve no paths implicitly."""
    config_path = resolve_path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Config is not JSON-compatible YAML and PyYAML is unavailable") from exc
        config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    config["_config_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return config


def resolve_path(value, base=ROOT):
    path = Path(str(value))
    if path.is_absolute():
        return path
    combined = Path(base) / path
    if any(character in str(combined) for character in "*?[]"):
        return Path(os.path.abspath(str(combined)))
    return combined.resolve()


def resolve_config_path(config, value):
    return resolve_path(value, Path(config["_config_dir"]))


def ensure_dir(path):
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(value, path):
    target = Path(path)
    ensure_dir(target.parent)
    atomic_write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), target, "utf-8")


def write_table(frame, path):
    target = Path(path)
    ensure_dir(target.parent)
    with tempfile.NamedTemporaryFile("w",suffix=target.suffix,dir=target.parent,delete=False,encoding="utf-8-sig",newline="") as stream:
        temporary=Path(stream.name);frame.to_csv(stream,index=False)
    os.replace(temporary,target)
    parquet = target.with_suffix(".parquet")
    try:
        temporary=parquet.with_suffix(parquet.suffix+".tmp");frame.to_parquet(temporary,index=False);os.replace(temporary,parquet)
    except Exception:
        pickle=parquet.with_suffix(".pkl");temporary=pickle.with_suffix(".pkl.tmp");frame.to_pickle(temporary);os.replace(temporary,pickle)


def atomic_write_text(text,path,encoding="utf-8"):
    target=Path(path);ensure_dir(target.parent)
    with tempfile.NamedTemporaryFile("w",dir=target.parent,delete=False,encoding=encoding,newline="") as stream:
        temporary=Path(stream.name);stream.write(text);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,target)


def runtime_provenance(config,input_paths=None,code_version="2026.07.12-harmonic-map-v2"):
    try:
        git=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,check=False).stdout.strip() or None
    except OSError: git=None
    cache_path=ROOT/config.get("project",{}).get("output_dir","outputs")/"input_sha256_cache.json"
    try:
        cache=json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    except Exception:
        cache={}
    fingerprints=[];updated_cache=dict(cache)
    sidecars={".shp":(".shp",".shx",".dbf",".prj",".cpg",".qpj")}
    paths=[]
    for path in list(input_paths or []):
        path=Path(path)
        if path.suffix.lower() in sidecars:
            for suffix in sidecars[path.suffix.lower()]:
                candidate=path.with_suffix(suffix)
                if candidate.exists():
                    paths.append(candidate)
        else:
            paths.append(path)
    for index,path in enumerate(paths,1):
        path=Path(path)
        if path.exists() and path.is_file():
            stat=path.stat();key=str(path.resolve());cached=cache.get(key,{})
            if cached.get("size_bytes")==stat.st_size and cached.get("mtime_ns")==stat.st_mtime_ns and cached.get("sha256"):
                fingerprint={k:cached[k] for k in ("path","size_bytes","sha256")}
            else:
                print(f"hashing_input {index}/{len(paths)} {path}",flush=True)
                fingerprint=file_fingerprint(path);fingerprint["mtime_ns"]=stat.st_mtime_ns
            updated_cache[key]={**fingerprint,"mtime_ns":stat.st_mtime_ns}
            fingerprints.append({k:fingerprint[k] for k in ("path","size_bytes","sha256")})
    try:
        write_json(updated_cache,cache_path)
    except Exception:
        pass
    dependencies={}
    for name in ("numpy","pandas","scipy","rasterio","h5py","geopandas","pyproj","yaml"):
        try:
            module=__import__(name)
            dependencies[name]=getattr(module,"__version__","installed")
        except Exception:
            dependencies[name]=None
    source_digest=hashlib.sha256()
    for source in sorted(ROOT.glob("*.py")):
        source_digest.update(source.name.encode("utf-8"))
        source_digest.update(file_fingerprint(source)["sha256"].encode("utf-8"))
    return {"code_version":code_version,"git_commit":git,"config_sha256":config.get("_config_sha256"),
            "input_file_sha256":fingerprints,"inputs":fingerprints,"run_id":uuid.uuid4().hex,
            "source_tree_sha256":source_digest.hexdigest(),
            "run_start_time":pd.Timestamp.utcnow().isoformat(),"python_version":platform.python_version(),
            "dependency_versions":dependencies}


def file_fingerprint(path, block_size=1024 * 1024):
    """Return a reproducible SHA256 without loading the file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
    stat = Path(path).stat()
    return {
        "path": str(Path(path).resolve()),
        "size_bytes": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def read_table(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")

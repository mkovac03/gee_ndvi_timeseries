#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from pprint import pformat

import ee
import numpy as np
import pandas as pd
from tqdm import tqdm
from dateutil.relativedelta import relativedelta

import geopandas as gpd
from shapely.geometry import mapping

# Optional YAML support
try:
    import yaml
except ImportError:
    yaml = None


# -------------------------
# Earth Engine init (configurable project)
# -------------------------
def init_ee(gee_project: str | None = None):
    gee_project = (gee_project or "").strip() or None
    try:
        if gee_project:
            ee.Initialize(project=gee_project)
        else:
            ee.Initialize()
    except Exception:
        ee.Authenticate()
        if gee_project:
            ee.Initialize(project=gee_project)
        else:
            ee.Initialize()


# -------------------------
# Defaults
# -------------------------
DEFAULT_CONFIG = {
    "workdir": "./ndvi_run",
    "gee_project": "",

    "locations_path": "locations.csv",
    "locations_type": "auto",   # auto|csv|vector
    "input_epsg": 4326,

    "id_col": "SITE_ID",
    "csv_x_col": "X",
    "csv_y_col": "Y",
    "csv_wkt_col": "",
    "vector_layer": "",

    "save_locations_gpkg": False,
    "locations_gpkg_out": "locations_loaded.gpkg",
    "locations_gpkg_layer": "sites",

    "tutorial": False,
    "dummy_out": "dummy_locations.csv",

    "start": "2016-01-01",
    "end": "2019-01-01",
    "unit": "month",   # day|week|month|year
    "step": 1,

    # temporal aggregation across images in the window
    "agg": "median",  # mean|median

    "scale": 10,
    "cloud_pct": 60,
    "nodata": -9999.0,

    "out_csv": "ndvi_timeseries.csv",

    # Unique runs
    "use_run_dir": True,
    "run_root": "runs",
    "run_id": "auto",

    # Plotting
    "plot": True,
    "plot_mode": "save",   # save|show
    "plot_dir": "plots",
    "plot_format": "png",

    # Plot type switch
    "plot_type": "line",   # line|box|both

    # Combined plot (the "allplot" is ALWAYS this combined LINE plot)
    "plot_combined": True,
    "plot_combined_filename": "ndvi_all_sites.png",

    # Ribbon (for line plots)
    "plot_ribbon": True,
    "ribbon_multiplier": 1.0,
    "ribbon_alpha_site": 0.18,
    "ribbon_alpha_combined": 0.08,
    "ribbon_source": "auto",  # auto|temporal|spatial|none

    # Boxplot settings (interval-based)
    "compute_box_stats": "auto",     # auto|true|false (auto => enabled when plot_type includes box)
    "boxplot_alpha": 0.35,
    "box_xtick_every": 3,            # show every N intervals on x axis in per-site boxplots

    # Optional additional combined "boxstyle" plot (NOT the allplot; default off)
    "save_combined_boxstyle": False,
    "combined_boxstyle_filename": "",  # if empty => auto-name as boxstyle_<stem>.<plot_format>
    "box_band_alpha_inner": 0.10,       # P25-75 band alpha (combined boxstyle)
    "box_band_alpha_outer": 0.05,       # P05-95 band alpha (combined boxstyle)

    # Optional spatial SD for non-point geometries (polygons)
    "compute_spatial_sd": False,

    # Nodata/invalid masking
    "mask_nodata_in_df": True,
    "mask_invalid_ndvi": True,
}


# -------------------------
# Tee logger
# -------------------------
class Tee:
    """Duplicate writes to multiple file-like streams."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def start_run_logging(run_dir: Path, cfg: dict, cfg_path: Path):
    """
    Redirect stdout+stderr to both console and a log file in run_dir.
    Also print (and thus log) the run header + config snapshot.
    Returns: (log_file_handle, run_datetime_str)
    """
    run_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_fp = run_dir / "run.log"

    log_fh = open(log_fp, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_fh)
    sys.stderr = Tee(sys.__stderr__, log_fh)

    print("=" * 80)
    print(f"RUN START: {run_dt}")
    print(f"RUN DIR  : {run_dir}")
    print(f"CONFIG   : {cfg_path.resolve()}")
    print("=" * 80)
    print("CONFIG (merged defaults + user overrides):")
    print(pformat(cfg, width=120, sort_dicts=True))
    print("=" * 80)

    # Save exact config used for reproducibility
    try:
        used_cfg_fp = run_dir / "config_used.json"
        with open(used_cfg_fp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
        print(f"Saved merged config to: {used_cfg_fp}")
    except Exception as e:
        print(f"WARNING: failed to save config_used.json: {e}", file=sys.stderr)

    return log_fh, run_dt


# -------------------------
# Config I/O
# -------------------------
def write_default_config(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in [".yml", ".yaml"]:
        if yaml is None:
            raise ImportError("pyyaml not installed. pip install pyyaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(DEFAULT_CONFIG, f, sort_keys=False)
    elif path.suffix.lower() == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
    else:
        raise ValueError("Config extension must be .yaml/.yml or .json")
    print(f"Wrote template config: {path}")


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    ext = path.suffix.lower()
    if ext in [".yml", ".yaml"]:
        if yaml is None:
            raise ImportError("pyyaml not installed. pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        raise ValueError("Config extension must be .yaml/.yml or .json")

    merged = dict(DEFAULT_CONFIG)
    merged.update(cfg)
    return merged


def validate_config(cfg: dict):
    if cfg["unit"] not in ["day", "week", "month", "year"]:
        raise ValueError("unit must be one of: day, week, month, year")
    if cfg["agg"] not in ["mean", "median"]:
        raise ValueError("agg must be one of: mean, median")
    if int(cfg["step"]) < 1:
        raise ValueError("step must be >= 1")
    if int(cfg["scale"]) <= 0:
        raise ValueError("scale must be > 0")

    s = datetime.strptime(cfg["start"], "%Y-%m-%d")
    e = datetime.strptime(cfg["end"], "%Y-%m-%d")
    if e <= s:
        raise ValueError("end must be after start")

    if not cfg.get("input_epsg"):
        raise ValueError("input_epsg must be set (EPSG code).")

    if cfg.get("plot_mode", "save") not in ["save", "show"]:
        raise ValueError("plot_mode must be 'save' or 'show'")

    rs = str(cfg.get("ribbon_source", "auto")).lower()
    if rs not in ["auto", "temporal", "spatial", "none"]:
        raise ValueError("ribbon_source must be one of: auto, temporal, spatial, none")

    pt = str(cfg.get("plot_type", "line")).lower()
    if pt not in ["line", "box", "both"]:
        raise ValueError("plot_type must be one of: line, box, both")

    cbs = str(cfg.get("compute_box_stats", "auto")).lower()
    if cbs not in ["auto", "true", "false"]:
        raise ValueError("compute_box_stats must be one of: auto, true, false")


def resolve_path(base_dir: Path, p: str) -> Path:
    pp = Path(p).expanduser()
    return pp if pp.is_absolute() else (base_dir / pp)


# -------------------------
# Run directory helper
# -------------------------
def make_run_dir(workdir: Path, cfg: dict) -> Path:
    if not bool(cfg.get("use_run_dir", True)):
        return workdir

    run_root = str(cfg.get("run_root", "runs")).strip() or "runs"
    run_id = str(cfg.get("run_id", "auto")).strip()

    if run_id.lower() == "auto" or run_id == "":
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = workdir / run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# -------------------------
# Dummy locations (CSV)
# -------------------------
def write_dummy_locations_csv(out_csv: Path):
    df = pd.DataFrame(
        [
            {"SITE_ID": "forest_gribskov", "X": 12.33333, "Y": 56.00000, "WKT": ""},
            {"SITE_ID": "cropland_lammefjorden", "X": 11.4661757267, "Y": 55.7687384862, "WKT": ""},
            {"SITE_ID": "wetland_lille_vildmose", "X": 10.2207437, "Y": 56.88767998, "WKT": ""},
        ]
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote dummy locations CSV: {out_csv}")


def prompt_tutorial_fallback() -> bool:
    if not sys.stdin.isatty():
        return False
    ans = input(
        "Locations not found. Would you like to run the script in tutorial mode with dummy locations? [y/N] "
    ).strip().lower()
    return ans in ("y", "yes")


# -------------------------
# Load locations: CSV or vector
# -------------------------
def infer_locations_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".csv":
        return "csv"
    if ext in [".shp", ".gpkg", ".geojson", ".json"]:
        return "vector"
    return "vector"


def load_locations_gdf(loc_path: Path, cfg: dict) -> gpd.GeoDataFrame:
    loc_type = cfg.get("locations_type", "auto").lower()
    if loc_type == "auto":
        loc_type = infer_locations_type(loc_path)

    input_epsg = int(cfg["input_epsg"])
    id_col = cfg.get("id_col", "SITE_ID")

    if loc_type == "csv":
        df = pd.read_csv(loc_path)

        if id_col not in df.columns:
            df[id_col] = [f"site_{i+1}" for i in range(len(df))]

        wkt_col = str(cfg.get("csv_wkt_col", "")).strip()
        if wkt_col and wkt_col in df.columns and df[wkt_col].astype(str).str.strip().ne("").any():
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.GeoSeries.from_wkt(df[wkt_col]),
                crs=f"EPSG:{input_epsg}",
            )
        else:
            x_col = cfg.get("csv_x_col", "X")
            y_col = cfg.get("csv_y_col", "Y")
            if x_col not in df.columns or y_col not in df.columns:
                raise ValueError(
                    f"CSV must contain '{x_col}' and '{y_col}' columns for coordinates, "
                    f"or provide a WKT column via csv_wkt_col."
                )
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df[x_col], df[y_col]),
                crs=f"EPSG:{input_epsg}",
            )

    elif loc_type == "vector":
        layer = str(cfg.get("vector_layer", "")).strip() or None
        gdf = gpd.read_file(loc_path, layer=layer)

        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=input_epsg)

        if id_col not in gdf.columns:
            gdf[id_col] = [f"site_{i+1}" for i in range(len(gdf))]

    else:
        raise ValueError("locations_type must be one of: auto, csv, vector")

    if id_col != "SITE_ID":
        gdf = gdf.rename(columns={id_col: "SITE_ID"})

    # GEE expects lon/lat
    gdf = gdf.to_crs(epsg=4326)

    if bool(cfg.get("save_locations_gpkg", False)):
        out_gpkg = Path(cfg.get("locations_gpkg_out", "locations_loaded.gpkg"))
        out_layer = str(cfg.get("locations_gpkg_layer", "sites"))
        gdf.attrs["_save_gpkg"] = (out_gpkg, out_layer)

    return gdf


def maybe_save_locations_gpkg(gdf: gpd.GeoDataFrame, out_base: Path):
    tpl = gdf.attrs.get("_save_gpkg", None)
    if not tpl:
        return
    out_gpkg_rel, out_layer = tpl
    out_gpkg = resolve_path(out_base, str(out_gpkg_rel))
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_gpkg, layer=out_layer, driver="GPKG")
    print(f"Saved locations to GPKG: {out_gpkg} (layer='{out_layer}')")


def has_nonpoint_geometries(gdf: gpd.GeoDataFrame) -> bool:
    return not (set(gdf.geometry.geom_type.unique()) <= {"Point"})


# -------------------------
# Time windows
# -------------------------
def make_windows(start: str, end: str, unit: str, step: int):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")

    windows = []
    cur = s
    while cur < e:
        if unit == "day":
            nxt = cur + timedelta(days=step)
        elif unit == "week":
            nxt = cur + timedelta(weeks=step)
        elif unit == "month":
            nxt = cur + relativedelta(months=step)
        elif unit == "year":
            nxt = cur + relativedelta(years=step)
        else:
            raise ValueError("unit must be one of: day, week, month, year")

        if nxt > e:
            nxt = e

        windows.append((cur, nxt))
        cur = nxt

    return windows


# -------------------------
# EE FeatureCollection from GeoDataFrame
# -------------------------
def gdf_to_ee_featurecollection(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    feats = []
    for _, row in gdf.iterrows():
        ee_geom = ee.Geometry(mapping(row.geometry))
        feats.append(ee.Feature(ee_geom, {"SITE_ID": str(row["SITE_ID"])}))
    return ee.FeatureCollection(feats)


# -------------------------
# Sentinel-2 NDVI + temporal SD or MAD + optional temporal percentiles for boxplots
# -------------------------
def mask_s2_sr(image):
    qa = image.select("QA60")
    cloud = 1 << 10
    cirrus = 1 << 11
    qa_mask = qa.bitwiseAnd(cloud).eq(0).And(qa.bitwiseAnd(cirrus).eq(0))

    scl = image.select("SCL")
    scl_mask = (
        scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    )
    return image.updateMask(qa_mask).updateMask(scl_mask)


def build_ndvi_collection(fc_geom, start_ee, end_ee, cloud_pct: float):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_ee, end_ee)
        .filterBounds(fc_geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(mask_s2_sr)
        .map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI").float())
    )


def build_ndvi_var_pct_images(
    fc_geom,
    start_ee,
    end_ee,
    agg: str,
    cloud_pct: float,
    nodata: float,
    need_box_stats: bool,
):
    """
    Returns:
      ndvi_img (band NDVI): temporal mean/median NDVI in window
      var_img  : temporal SD (if mean) or temporal MAD (if median)
      pct_img  : (optional) temporal percentiles across images in window (NDVI_P05..NDVI_P95)
      n_images : image count
      var_kind : 'sd' or 'mad'
      var_key  : 'NDVI_SD_TEMP' or 'NDVI_MAD_TEMP'
    """
    col = build_ndvi_collection(fc_geom, start_ee, end_ee, cloud_pct)
    n_images = col.size()

    if agg == "mean":
        ndvi = col.select("NDVI").reduce(ee.Reducer.mean()).rename("NDVI")
        var = col.select("NDVI").reduce(ee.Reducer.stdDev()).rename("NDVI_SD_TEMP")
        var_kind = "sd"
        var_key = "NDVI_SD_TEMP"

        var = ee.Image(
            ee.Algorithms.If(
                n_images.gt(1),
                var,
                ee.Image.constant(nodata).rename(var_key).toFloat(),
            )
        )
    else:
        ndvi = col.select("NDVI").reduce(ee.Reducer.median()).rename("NDVI")
        absdev = col.select("NDVI").map(lambda img: img.subtract(ndvi).abs().rename("DEV"))
        mad = absdev.select("DEV").reduce(ee.Reducer.median()).rename("NDVI_MAD_TEMP")
        var_kind = "mad"
        var_key = "NDVI_MAD_TEMP"

        var = ee.Image(
            ee.Algorithms.If(
                n_images.gt(1),
                mad,
                ee.Image.constant(nodata).rename(var_key).toFloat(),
            )
        )

    ndvi = ee.Image(
        ee.Algorithms.If(
            n_images.gt(0),
            ndvi,
            ee.Image.constant(nodata).rename("NDVI").toFloat(),
        )
    )

    pct_img = None
    if need_box_stats:
        pct = col.select("NDVI").reduce(ee.Reducer.percentile([5, 25, 50, 75, 95]))
        pct = pct.rename(["NDVI_P05", "NDVI_P25", "NDVI_P50", "NDVI_P75", "NDVI_P95"])
        pct_img = ee.Image(
            ee.Algorithms.If(
                n_images.gt(0),
                pct,
                ee.Image.constant([nodata, nodata, nodata, nodata, nodata]).rename(
                    ["NDVI_P05", "NDVI_P25", "NDVI_P50", "NDVI_P75", "NDVI_P95"]
                ).toFloat(),
            )
        )

    return ndvi, var, pct_img, n_images, var_kind, var_key


def extract_timeseries(
    sites_fc: ee.FeatureCollection,
    cfg: dict,
    nonpoint: bool,
    need_box_stats: bool,
) -> pd.DataFrame:
    fc_geom = sites_fc.geometry()
    windows = make_windows(cfg["start"], cfg["end"], cfg["unit"], int(cfg["step"]))

    nodata = float(cfg["nodata"])
    scale = int(cfg["scale"])
    agg = str(cfg["agg"]).lower()
    compute_spatial_sd = bool(cfg.get("compute_spatial_sd", False)) and nonpoint

    records = []
    for (ws, we) in tqdm(windows, desc="Intervals"):
        ws_str = ws.strftime("%Y-%m-%d")
        we_str = we.strftime("%Y-%m-%d")

        ndvi_img, var_img, pct_img, n_images, var_kind, var_key = build_ndvi_var_pct_images(
            fc_geom=fc_geom,
            start_ee=ee.Date(ws_str),
            end_ee=ee.Date(we_str),
            agg=agg,
            cloud_pct=float(cfg["cloud_pct"]),
            nodata=nodata,
            need_box_stats=need_box_stats,
        )

        # Stable NDVI extraction
        ndvi_spatial_reducer = (ee.Reducer.mean() if agg == "mean" else ee.Reducer.median()).setOutputs(["NDVI"])
        ndvi_fc = ndvi_img.reduceRegions(
            collection=sites_fc,
            reducer=ndvi_spatial_reducer,
            scale=scale,
        ).map(lambda f: f.set({"N_IMAGES": n_images}))

        ndvi_out = ndvi_fc.getInfo()

        # Stable temporal variability extraction (SD or MAD)
        var_fc = var_img.reduceRegions(
            collection=sites_fc,
            reducer=ee.Reducer.mean().setOutputs([var_key]),
            scale=scale,
        ).map(lambda f: f.set({"N_IMAGES": n_images}))

        var_out = var_fc.getInfo()

        var_by_site = {}
        for feat in var_out["features"]:
            p = feat["properties"]
            sid = p.get("SITE_ID")
            v = p.get(var_key, None)
            try:
                v = float(v) if v is not None else None
            except Exception:
                v = None
            if v is not None and abs(v - nodata) < 1e-6:
                v = None
            var_by_site[sid] = v

        # Optional temporal percentiles (for interval boxplots)
        pct_by_site = {}
        if need_box_stats and pct_img is not None:
            pct_fc = pct_img.reduceRegions(
                collection=sites_fc,
                reducer=ee.Reducer.mean(),
                scale=scale,
            ).map(lambda f: f.set({"N_IMAGES": n_images}))
            pct_out = pct_fc.getInfo()

            for feat in pct_out["features"]:
                p = feat["properties"]
                sid = p.get("SITE_ID")
                vals = {}
                for k in ["NDVI_P05", "NDVI_P25", "NDVI_P50", "NDVI_P75", "NDVI_P95"]:
                    v = p.get(k, None)
                    try:
                        v = float(v) if v is not None else None
                    except Exception:
                        v = None
                    if v is not None and abs(v - nodata) < 1e-6:
                        v = None
                    vals[k] = v
                pct_by_site[sid] = vals

        # Optional spatial SD of window NDVI inside polygons
        spat_by_site = {}
        if compute_spatial_sd:
            spat_fc = ndvi_img.reduceRegions(
                collection=sites_fc,
                reducer=ee.Reducer.stdDev().setOutputs(["NDVI_SD_SPAT"]),
                scale=scale,
            ).map(lambda f: f.set({"N_IMAGES": n_images}))
            spat_out = spat_fc.getInfo()
            for feat in spat_out["features"]:
                p = feat["properties"]
                sid = p.get("SITE_ID")
                v = p.get("NDVI_SD_SPAT", None)
                try:
                    v = float(v) if v is not None else None
                except Exception:
                    v = None
                if v is not None and abs(v - nodata) < 1e-6:
                    v = None
                spat_by_site[sid] = v

        # Merge per-site
        for feat in ndvi_out["features"]:
            p = feat["properties"]
            sid = p.get("SITE_ID")

            ndvi = p.get("NDVI", None)
            try:
                ndvi = float(ndvi) if ndvi is not None else None
            except Exception:
                ndvi = None
            if ndvi is not None and abs(ndvi - nodata) < 1e-6:
                ndvi = None

            nimg = p.get("N_IMAGES", None)
            try:
                nimg = int(nimg) if nimg is not None else None
            except Exception:
                nimg = None

            vtemp = var_by_site.get(sid, None)

            rec = {
                "SITE_ID": sid,
                "window_start": ws_str,
                "window_end": we_str,
                "N_IMAGES": nimg,
                "NDVI": ndvi,
                "TEMP_VAR_KIND": var_kind,  # sd or mad
                "NDVI_SD_TEMP": vtemp if var_kind == "sd" else None,
                "NDVI_MAD_TEMP": vtemp if var_kind == "mad" else None,
                "NDVI_SD_SPAT": spat_by_site.get(sid, None) if compute_spatial_sd else None,
            }

            if need_box_stats:
                vals = pct_by_site.get(sid, {})
                rec.update({
                    "NDVI_P05": vals.get("NDVI_P05"),
                    "NDVI_P25": vals.get("NDVI_P25"),
                    "NDVI_P50": vals.get("NDVI_P50"),
                    "NDVI_P75": vals.get("NDVI_P75"),
                    "NDVI_P95": vals.get("NDVI_P95"),
                })

            records.append(rec)

    return pd.DataFrame(records)


def clean_df(df: pd.DataFrame, cfg: dict, need_box_stats: bool) -> pd.DataFrame:
    out = df.copy()
    out["NDVI"] = pd.to_numeric(out["NDVI"], errors="coerce")
    out["NDVI_SD_TEMP"] = pd.to_numeric(out.get("NDVI_SD_TEMP"), errors="coerce")
    out["NDVI_MAD_TEMP"] = pd.to_numeric(out.get("NDVI_MAD_TEMP"), errors="coerce")
    out["NDVI_SD_SPAT"] = pd.to_numeric(out.get("NDVI_SD_SPAT"), errors="coerce")
    out["N_IMAGES"] = pd.to_numeric(out.get("N_IMAGES"), errors="coerce")

    if need_box_stats:
        for k in ["NDVI_P05", "NDVI_P25", "NDVI_P50", "NDVI_P75", "NDVI_P95"]:
            out[k] = pd.to_numeric(out.get(k), errors="coerce")

    if bool(cfg.get("mask_invalid_ndvi", True)):
        out.loc[(out["NDVI"] < -1.0001) | (out["NDVI"] > 1.0001), "NDVI"] = np.nan

    if bool(cfg.get("mask_nodata_in_df", True)):
        nodata = float(cfg["nodata"])
        for k in ["NDVI", "NDVI_SD_TEMP", "NDVI_MAD_TEMP", "NDVI_SD_SPAT"]:
            if k in out.columns:
                out.loc[(out[k] - nodata).abs() < 1e-6, k] = np.nan
        if need_box_stats:
            for k in ["NDVI_P05", "NDVI_P25", "NDVI_P50", "NDVI_P75", "NDVI_P95"]:
                out.loc[(out[k] - nodata).abs() < 1e-6, k] = np.nan

    # sanity
    out.loc[out["NDVI_SD_TEMP"] < 0, "NDVI_SD_TEMP"] = np.nan
    out.loc[out["NDVI_MAD_TEMP"] < 0, "NDVI_MAD_TEMP"] = np.nan
    out.loc[out["NDVI_SD_SPAT"] < 0, "NDVI_SD_SPAT"] = np.nan

    return out


def choose_ribbon(df: pd.DataFrame, cfg: dict, nonpoint: bool) -> pd.DataFrame:
    """
    Create:
      - NDVI_RIBBON  : numeric ribbon width
      - RIBBON_KIND  : 'sd'/'mad'/'spatial_sd'/'none'
    Rule:
      - agg=mean   => temporal SD
      - agg=median => temporal MAD
    """
    out = df.copy()

    unit = str(cfg.get("unit", "")).lower()
    agg = str(cfg.get("agg", "")).lower()
    src = str(cfg.get("ribbon_source", "auto")).lower()
    allow_spat = bool(cfg.get("compute_spatial_sd", False)) and nonpoint

    if agg == "mean":
        temporal_col = "NDVI_SD_TEMP"
        temporal_kind = "sd"
    else:
        temporal_col = "NDVI_MAD_TEMP"
        temporal_kind = "mad"

    if src == "auto":
        if unit == "day":
            if allow_spat:
                out["NDVI_RIBBON"] = out["NDVI_SD_SPAT"]
                out["RIBBON_KIND"] = "spatial_sd"
            else:
                out["NDVI_RIBBON"] = np.nan
                out["RIBBON_KIND"] = "none"
        else:
            out["NDVI_RIBBON"] = out[temporal_col]
            out["RIBBON_KIND"] = temporal_kind
    elif src == "temporal":
        out["NDVI_RIBBON"] = out[temporal_col]
        out["RIBBON_KIND"] = temporal_kind
    elif src == "spatial":
        if allow_spat:
            out["NDVI_RIBBON"] = out["NDVI_SD_SPAT"]
            out["RIBBON_KIND"] = "spatial_sd"
        else:
            out["NDVI_RIBBON"] = np.nan
            out["RIBBON_KIND"] = "none"
    else:
        out["NDVI_RIBBON"] = np.nan
        out["RIBBON_KIND"] = "none"

    return out


# -------------------------
# Plotting
# -------------------------
def _ensure_backend(plot_mode: str):
    if plot_mode == "save":
        import matplotlib
        matplotlib.use("Agg")


def plot_line_per_site(df: pd.DataFrame, base_title: str, plot_mode: str, plot_dir: Path, plot_format: str,
                       show_ribbon: bool, ribbon_mult: float, ribbon_alpha: float):
    if df.empty:
        print("No records to plot.")
        return

    _ensure_backend(plot_mode)
    import matplotlib.pyplot as plt

    dfp = df.copy()
    dfp["window_start"] = pd.to_datetime(dfp["window_start"])
    dfp = dfp.sort_values(["SITE_ID", "window_start"])

    plot_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for site_id, g in dfp.groupby("SITE_ID"):
        x = g["window_start"].to_numpy()
        y = g["NDVI"].to_numpy(dtype=float, na_value=np.nan)

        fig, ax = plt.subplots()
        ax.plot(x, y, marker="o", linestyle="-")

        kind = str(g["RIBBON_KIND"].iloc[0]) if "RIBBON_KIND" in g.columns else "none"
        ribbon_label = f"ribbon: ±{ribbon_mult:g} × {kind}" if kind != "none" else "ribbon: none"

        if show_ribbon and "NDVI_RIBBON" in g.columns and kind != "none":
            r = g["NDVI_RIBBON"].to_numpy(dtype=float, na_value=np.nan)
            lower = y - ribbon_mult * r
            upper = y + ribbon_mult * r
            ax.fill_between(x, lower, upper, alpha=ribbon_alpha)

        ax.set_title(f"{base_title} — {site_id}\n{ribbon_label}")
        ax.set_xlabel("Time (window start)")
        ax.set_ylabel("NDVI")
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        fig.tight_layout()

        if plot_mode == "save":
            fp = plot_dir / f"ndvi_line_{site_id}.{plot_format}"
            fig.savefig(fp, dpi=150, bbox_inches="tight")
            saved.append(fp)
            plt.close(fig)
        else:
            plt.show()

    if saved:
        print("Saved per-site LINE plots:")
        for fp in saved:
            print(f"  - {fp}")


def plot_line_combined(df: pd.DataFrame, base_title: str, plot_mode: str, plot_dir: Path, filename: str,
                       show_ribbon: bool, ribbon_mult: float, ribbon_alpha: float):
    if df.empty:
        return

    _ensure_backend(plot_mode)
    import matplotlib.pyplot as plt

    dfp = df.copy()
    dfp["window_start"] = pd.to_datetime(dfp["window_start"])
    dfp = dfp.sort_values(["SITE_ID", "window_start"])

    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    kinds = set(str(k) for k in dfp["RIBBON_KIND"].dropna().unique()) if "RIBBON_KIND" in dfp.columns else {"none"}
    kind_label = list(kinds)[0] if len(kinds) == 1 else "mixed"
    ribbon_label = f"ribbon: ±{ribbon_mult:g} × {kind_label}" if kind_label != "none" else "ribbon: none"

    for site_id, g in dfp.groupby("SITE_ID"):
        x = g["window_start"].to_numpy()
        y = g["NDVI"].to_numpy(dtype=float, na_value=np.nan)
        ax.plot(x, y, marker="o", linestyle="-", label=str(site_id))

        if show_ribbon and "NDVI_RIBBON" in g.columns:
            k = str(g["RIBBON_KIND"].iloc[0])
            if k != "none":
                r = g["NDVI_RIBBON"].to_numpy(dtype=float, na_value=np.nan)
                lower = y - ribbon_mult * r
                upper = y + ribbon_mult * r
                ax.fill_between(x, lower, upper, alpha=ribbon_alpha)

    ax.set_title(f"{base_title}\n{ribbon_label}")
    ax.set_xlabel("Time (window start)")
    ax.set_ylabel("NDVI")
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    n_sites = dfp["SITE_ID"].nunique()
    ncol = min(4, max(1, n_sites))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=ncol,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.5,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)

    if plot_mode == "save":
        fp = plot_dir / filename
        fig.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved ALLPLOT (combined LINE) plot: {fp}")
    else:
        plt.show()


def plot_box_per_site_intervals(df: pd.DataFrame, base_title: str, plot_mode: str, plot_dir: Path, plot_format: str,
                                box_alpha: float, xtick_every: int):
    """
    Interval-based boxplots: one box per interval (window_start), per site.
    Uses precomputed percentiles (P05/P25/P50/P75/P95) across images within each interval.
    """
    needed = {"NDVI_P05", "NDVI_P25", "NDVI_P50", "NDVI_P75", "NDVI_P95"}
    if not needed.issubset(set(df.columns)):
        print("Boxplots requested but percentile columns are missing. (Enable compute_box_stats.)")
        return

    _ensure_backend(plot_mode)
    import matplotlib.pyplot as plt

    dfp = df.copy()
    dfp["window_start"] = pd.to_datetime(dfp["window_start"])
    dfp = dfp.sort_values(["SITE_ID", "window_start"])

    plot_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for site_id, g in dfp.groupby("SITE_ID"):
        g = g.sort_values("window_start").reset_index(drop=True)

        stats = []
        labels = []
        for _, row in g.iterrows():
            p05, p25, p50, p75, p95 = row["NDVI_P05"], row["NDVI_P25"], row["NDVI_P50"], row["NDVI_P75"], row["NDVI_P95"]
            if not (np.isfinite(p05) and np.isfinite(p25) and np.isfinite(p50) and np.isfinite(p75) and np.isfinite(p95)):
                continue

            lbl = row["window_start"].strftime("%Y-%m-%d")
            stats.append({
                "label": lbl,
                "whislo": p05,
                "q1": p25,
                "med": p50,
                "q3": p75,
                "whishi": p95,
                "fliers": [],
            })
            labels.append(lbl)

        if len(stats) == 0:
            continue

        fig, ax = plt.subplots()
        bp = ax.bxp(stats, showfliers=False, patch_artist=True)

        for patch in bp["boxes"]:
            patch.set_alpha(box_alpha)

        n = len(stats)
        xtick_every = max(1, int(xtick_every))
        ticks = np.arange(1, n + 1)
        keep = (ticks - 1) % xtick_every == 0
        ax.set_xticks(ticks[keep])
        ax.set_xticklabels([labels[i] for i in np.where(keep)[0]], rotation=45, ha="right")

        ax.set_title(f"{base_title} — {site_id}\nbox: temporal percentiles within each interval (P05/P25/P50/P75/P95)")
        ax.set_xlabel("Interval (window_start)")
        ax.set_ylabel("NDVI")
        fig.tight_layout()

        if plot_mode == "save":
            fp = plot_dir / f"ndvi_box_{site_id}.{plot_format}"
            fig.savefig(fp, dpi=150, bbox_inches="tight")
            saved.append(fp)
            plt.close(fig)
        else:
            plt.show()

    if saved:
        print("Saved per-site BOX plots (interval-based):")
        for fp in saved:
            print(f"  - {fp}")


def plot_box_combined_quantile_bands(df: pd.DataFrame, base_title: str, plot_mode: str, plot_dir: Path, filename: str,
                                    alpha_inner: float, alpha_outer: float):
    """
    Optional extra combined 'boxplot-style' plot: quantile bands per site over time
      - median line: P50
      - inner band: P25..P75 (the 'box')
      - outer band: P05..P95 (the 'whiskers')
    This is NOT the main allplot; the allplot is always the combined LINE plot.
    """
    needed = {"NDVI_P05", "NDVI_P25", "NDVI_P50", "NDVI_P75", "NDVI_P95"}
    if not needed.issubset(set(df.columns)):
        print("Combined boxstyle requested but percentile columns are missing.")
        return

    _ensure_backend(plot_mode)
    import matplotlib.pyplot as plt

    dfp = df.copy()
    dfp["window_start"] = pd.to_datetime(dfp["window_start"])
    dfp = dfp.sort_values(["SITE_ID", "window_start"])

    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()

    for site_id, g in dfp.groupby("SITE_ID"):
        g = g.sort_values("window_start")
        x = g["window_start"].to_numpy()
        p05 = g["NDVI_P05"].to_numpy(dtype=float, na_value=np.nan)
        p25 = g["NDVI_P25"].to_numpy(dtype=float, na_value=np.nan)
        p50 = g["NDVI_P50"].to_numpy(dtype=float, na_value=np.nan)
        p75 = g["NDVI_P75"].to_numpy(dtype=float, na_value=np.nan)
        p95 = g["NDVI_P95"].to_numpy(dtype=float, na_value=np.nan)

        ax.plot(x, p50, marker="o", linestyle="-", label=str(site_id))
        ax.fill_between(x, p05, p95, alpha=alpha_outer)
        ax.fill_between(x, p25, p75, alpha=alpha_inner)

    ax.set_title(f"{base_title}\nboxstyle: P25–P75 band + P05–P95 band (temporal percentiles within each interval)")
    ax.set_xlabel("Time (window start)")
    ax.set_ylabel("NDVI")
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    n_sites = dfp["SITE_ID"].nunique()
    ncol = min(4, max(1, n_sites))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=ncol,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.5,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)

    if plot_mode == "save":
        fp = plot_dir / filename
        fig.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved combined BOXSTYLE plot: {fp}")
    else:
        plt.show()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="GEE Sentinel-2 NDVI time series extractor (config-driven).")
    ap.add_argument("--config", type=str, default="config.yaml",
                    help="Path to config .yaml/.yml or .json (default: config.yaml)")
    ap.add_argument("--write-config", type=str, default="", help="Write a template config file and exit")
    args = ap.parse_args()

    if args.write_config:
        write_default_config(Path(args.write_config))
        return 0

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: Config not found: {cfg_path}", file=sys.stderr)
        print("Tip: create one with --write-config config.yaml", file=sys.stderr)
        return 2

    cfg = load_config(cfg_path)
    validate_config(cfg)

    workdir = Path(cfg["workdir"]).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    run_dir = make_run_dir(workdir, cfg)

    # Start capturing all prints/errors to run_dir/run.log
    log_fh, _run_dt = start_run_logging(run_dir, cfg, cfg_path)

    try:
        # Inputs from workdir, outputs to run_dir
        loc_path = resolve_path(workdir, cfg["locations_path"])
        dummy_out = resolve_path(run_dir, cfg["dummy_out"])
        out_csv = resolve_path(run_dir, cfg["out_csv"])
        plot_dir = resolve_path(run_dir, cfg.get("plot_dir", "plots"))
        plot_format = str(cfg.get("plot_format", "png")).lower()

        if not loc_path.exists():
            if bool(cfg.get("tutorial", False)):
                write_dummy_locations_csv(dummy_out)
                loc_path = dummy_out
            else:
                if prompt_tutorial_fallback():
                    write_dummy_locations_csv(dummy_out)
                    loc_path = dummy_out
                else:
                    print(f"ERROR: locations not found: {loc_path}", file=sys.stderr)
                    print("Set tutorial: true in config to auto-run with dummy locations.", file=sys.stderr)
                    return 2

        gdf = load_locations_gdf(loc_path, cfg)
        maybe_save_locations_gpkg(gdf, run_dir)
        nonpoint = has_nonpoint_geometries(gdf)

        init_ee(cfg.get("gee_project", ""))

        sites_fc = gdf_to_ee_featurecollection(gdf)

        plot_type = str(cfg.get("plot_type", "line")).lower()
        cbs = str(cfg.get("compute_box_stats", "auto")).lower()
        need_box_stats = (plot_type in ("box", "both")) if cbs == "auto" else (cbs == "true")

        df_raw = extract_timeseries(sites_fc, cfg, nonpoint=nonpoint, need_box_stats=need_box_stats)
        df = clean_df(df_raw, cfg, need_box_stats=need_box_stats)
        df = choose_ribbon(df, cfg, nonpoint=nonpoint)

        # Make the method explicit in CSV
        df["AGG_METHOD"] = str(cfg.get("agg", "")).lower()
        if need_box_stats:
            df["BOX_KIND"] = "temporal_percentiles_within_interval"

        print("\nPreview (first 10 rows):")
        print(df.head(10).to_string(index=False))
        print("\nNon-null counts:")
        cols = ["NDVI", "NDVI_SD_TEMP", "NDVI_MAD_TEMP", "NDVI_SD_SPAT", "NDVI_RIBBON"]
        if need_box_stats:
            cols += ["NDVI_P05", "NDVI_P25", "NDVI_P50", "NDVI_P75", "NDVI_P95"]
        print(df[cols].notna().sum())

        # ----------------------------
        # Plotting
        # ----------------------------
        if bool(cfg.get("plot", True)):
            base_title = f"S2 NDVI (agg={cfg['agg']}) [{cfg['unit']} step={cfg['step']}]"
            plot_mode = str(cfg.get("plot_mode", "save"))

            ribbon_mult = float(cfg.get("ribbon_multiplier", 1.0))
            alpha_site = float(cfg.get("ribbon_alpha_site", 0.18))
            alpha_all = float(cfg.get("ribbon_alpha_combined", 0.08))

            # Per-site LINE plots only if requested
            if plot_type in ("line", "both"):
                plot_line_per_site(
                    df,
                    base_title=base_title,
                    plot_mode=plot_mode,
                    plot_dir=plot_dir,
                    plot_format=plot_format,
                    show_ribbon=bool(cfg.get("plot_ribbon", True)),
                    ribbon_mult=ribbon_mult,
                    ribbon_alpha=alpha_site,
                )

            # ALLPLOT (combined) saved ONCE, always as LINE plot
            if bool(cfg.get("plot_combined", True)):
                fname = str(cfg.get("plot_combined_filename", "ndvi_all_sites.png")).strip() or "ndvi_all_sites.png"
                plot_line_combined(
                    df,
                    base_title=base_title,
                    plot_mode=plot_mode,
                    plot_dir=plot_dir,
                    filename=fname,
                    show_ribbon=bool(cfg.get("plot_ribbon", True)),
                    ribbon_mult=ribbon_mult,
                    ribbon_alpha=alpha_all,
                )

            # Boxplots per site only if box or both
            if plot_type in ("box", "both"):
                box_alpha = float(cfg.get("boxplot_alpha", 0.35))
                xtick_every = int(cfg.get("box_xtick_every", 3))

                plot_box_per_site_intervals(
                    df,
                    base_title=base_title,
                    plot_mode=plot_mode,
                    plot_dir=plot_dir,
                    plot_format=plot_format,
                    box_alpha=box_alpha,
                    xtick_every=xtick_every,
                )

                # Optional extra combined boxstyle plot (NOT the allplot)
                if bool(cfg.get("save_combined_boxstyle", False)) and bool(cfg.get("plot_combined", True)):
                    alpha_inner = float(cfg.get("box_band_alpha_inner", 0.10))
                    alpha_outer = float(cfg.get("box_band_alpha_outer", 0.05))
                    custom = str(cfg.get("combined_boxstyle_filename", "")).strip()
                    if custom:
                        box_fname = custom
                    else:
                        stem = Path(str(cfg.get("plot_combined_filename", "ndvi_all_sites.png"))).stem
                        box_fname = f"boxstyle_{stem}.{plot_format}"

                    plot_box_combined_quantile_bands(
                        df,
                        base_title=base_title,
                        plot_mode=plot_mode,
                        plot_dir=plot_dir,
                        filename=box_fname,
                        alpha_inner=alpha_inner,
                        alpha_outer=alpha_outer,
                    )

        # Save outputs
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}")
        print(f"Run directory: {run_dir}")
        print("=" * 80)
        print(f"RUN END: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  (SUCCESS)")
        print("=" * 80)
        return 0

    except Exception:
        print("\nFATAL ERROR (traceback follows):", file=sys.stderr)
        traceback.print_exc()
        print("=" * 80, file=sys.stderr)
        print(f"RUN END: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  (FAILED)", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        return 1

    finally:
        # Restore stdout/stderr
        try:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        except Exception:
            pass
        try:
            log_fh.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

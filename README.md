# GEE Sentinel-2 NDVI Time Series

This repository provides a config-driven Python script to extract Sentinel-2 NDVI time series from Google Earth Engine (GEE) for a set of point locations or polygons (CSV, SHP, GPKG, GeoJSON). The script can also generate plots and writes each run into a unique timestamped folder with logs for reproducibility.

---

## What the script computes

For each location and each time window (e.g., monthly), the script:

1. Loads Sentinel-2 Surface Reflectance imagery: `COPERNICUS/S2_SR_HARMONIZED`
2. Applies cloud masking using:

   * QA60 (cloud + cirrus bits)
   * SCL masking to remove problematic classes
3. Computes NDVI per image:

   * `NDVI = (B8 − B4) / (B8 + B4)`
4. Aggregates within each time window (temporal aggregation):

   * `agg: mean`   → NDVI mean + temporal **SD**
   * `agg: median` → NDVI median + temporal **MAD**
5. Aggregates over the geometry (spatial aggregation):

   * points: NDVI at the point (scale-dependent)
   * polygons: NDVI over the polygon (reduced over pixels within the polygon)
6. Optionally computes per-window temporal percentiles for interval “boxplots”:

   * P05/P25/P50/P75/P95

Output is a CSV with one row per `(SITE_ID, time window)` plus optional plots.

---

## Repository structure

Typical files:

* `gee_ndvi_timeseries.py` (main script)
* `config.yaml` (configuration controlling inputs/time/outputs/plotting)
* `environment.yml` (conda environment file)
* `runs/` (created automatically when `use_run_dir: true`), containing:

  * `ndvi_timeseries.csv` (output)
  * `plots/` (plots)
  * `run.log` (full log)
  * `config_used.json` (exact config used)

---

## Commands

Where to run commands:

* **Linux:** open a terminal (GNOME Terminal / Konsole), `cd` into the repo folder.
* **macOS:** open Terminal.app, `cd` into the repo folder.
* **Windows:** use one of:

  * **Anaconda Prompt** (recommended if using conda)
  * **Windows Terminal** (PowerShell)
  * **Git Bash**

Windows example (PowerShell):

```powershell
cd C:\Users\YourName\Documents\gee_ndvi_timeseries
```

---

## Installation (conda recommended)

From the repository root folder:

```bash
conda env create -f environment.yml
conda activate gee-ndvi
```

If you want a different environment name:

```bash
conda env create -f environment.yml -n gee-ndvi-timeseries
conda activate gee-ndvi-timeseries
```

---

## Google Earth Engine authentication

The script runs:

* `ee.Initialize()` and, if it fails,
* `ee.Authenticate()` then `ee.Initialize()`

On first run, you may be prompted to authenticate with Google in a browser.

Optional Earth Engine cloud project (GCP project id). Set this only if you need it:

```yaml
gee_project: "your-gcp-project-id"
```

If not needed, leave it empty:

```yaml
gee_project: ""
```

---

## Inputs: locations

The script supports:

* CSV (points)
* Vector files: `.shp`, `.gpkg`, `.geojson`

All inputs are reprojected internally to EPSG:4326 because GEE expects lon/lat.

### A) CSV points

A minimal CSV example (EPSG:4326 lon/lat):

```csv
SITE_ID,X,Y
forest_gribskov,12.33333,56.00000
cropland_lammefjorden,11.4661757267,55.7687384862
wetland_lille_vildmose,10.2207437,56.88767998
```

Default CSV config keys:

| Config key    | Default value | Meaning                                                   |
| ------------- | ------------- | --------------------------------------------------------- |
| `id_col`      | `"SITE_ID"`   | Column containing the unique location ID                  |
| `csv_x_col`   | `"X"`         | X coordinate column (longitude if EPSG:4326)              |
| `csv_y_col`   | `"Y"`         | Y coordinate column (latitude if EPSG:4326)               |
| `csv_wkt_col` | `""`          | Optional WKT geometry column (if provided, overrides X/Y) |
| `input_epsg`  | `4326`        | EPSG code of the coordinates/geometry in the CSV          |

Notes:

* If your CSV coordinates are not lon/lat, set `input_epsg` to the correct EPSG code.
* You can also provide geometries as WKT (points or polygons) if you set `csv_wkt_col: "WKT"`.

### B) Vector files (SHP / GPKG / GeoJSON)

Your vector file can contain points or polygons. It must include a unique ID field (default `SITE_ID`), or the script will create IDs automatically.

For GeoPackages with multiple layers:

```yaml
vector_layer: "sites"
```

---

## Configuration (config.yaml)

The script reads `config.yaml` by default when you run:

```bash
python gee_ndvi_timeseries.py
```

To use another config file:

```bash
python gee_ndvi_timeseries.py --config path/to/your_config.yaml
```

---

## Key configuration parameters

The full config is in `config.yaml`. The parameters below are the ones that most strongly affect runtime, output size, and correctness.

### 1) Paths and run folders

```yaml
workdir: "./"
locations_path: "locations.csv"
out_csv: "ndvi_timeseries.csv"

use_run_dir: true
run_root: "runs"
run_id: "auto"
```

* Relative paths are resolved relative to `workdir`.
* With `use_run_dir: true`, each run is written into `runs/YYYYMMDD_HHMMSS/`.
* The run folder always includes `run.log` and `config_used.json`.

### 2) Time windows (drives row count + runtime)

```yaml
start: "2016-01-01"
end:   "2026-01-01"
unit: "month"   # day | week | month | year
step: 1
```

Row count grows as:

* `rows ≈ N_locations × N_time_windows`

Example: 3 locations × ~120 monthly windows ≈ ~360 rows.

### 3) Temporal aggregation (drives NDVI + uncertainty metric)

```yaml
agg: "mean"    # mean | median
```

* `mean` → NDVI mean and temporal **SD** (`NDVI_SD_TEMP`)
* `median` → NDVI median and temporal **MAD** (`NDVI_MAD_TEMP`)

### 4) Cloud filtering and spatial scale

```yaml
cloud_pct: 60
scale: 10
```

* `cloud_pct` filters images using `CLOUDY_PIXEL_PERCENTAGE < cloud_pct`.
* `scale` controls the spatial sampling/reduction scale in meters.

### 5) Location chunking (helps avoid large EE payloads)

```yaml
location_chunk_size: 0
```

* `0` means all locations are processed together per time window.
* For large location counts, set to a value like 200–1000 to reduce per-request payload size.

---

## Plotting

```yaml
plot: true
plot_mode: "save"          # save | show
plot_type: "line"          # line | box | both

plot_combined: true
plot_combined_filename: "ndvi_all_sites.png"

plot_ribbon: true
ribbon_source: "auto"
ribbon_multiplier: 1.0
ribbon_alpha_site: 0.18
ribbon_alpha_combined: 0.08

compute_box_stats: "auto"
boxplot_alpha: 0.35
box_xtick_every: 3
```

* `plot_type: line` → time series
* `plot_type: box` → interval percentiles used for boxplots
* `plot_type: both` → both sets of plots

Important behavior:

* The combined “all sites” plot is saved once as a **line** plot even if `plot_type: box`.

Ribbons:

* If `agg=mean`, ribbon uses **SD**
* If `agg=median`, ribbon uses **MAD**

Boxplots:

* When box stats are enabled, the script computes per-window temporal percentiles: P05/P25/P50/P75/P95.

---

## Output CSV columns

Core columns:

* `SITE_ID`
* `window_start`, `window_end`
* `N_IMAGES`: number of Sentinel-2 images in the filtered collection for that window
* `NDVI`: aggregated NDVI value for that window/site

Variability / spread:

* If `agg=mean`:

  * `TEMP_VAR_KIND = sd`
  * `NDVI_SD_TEMP` = temporal SD in the window
* If `agg=median`:

  * `TEMP_VAR_KIND = mad`
  * `NDVI_MAD_TEMP` = temporal MAD in the window

Ribbon:

* `NDVI_RIBBON`: value used for ribbon width
* `RIBBON_KIND`: sd/mad/spatial_sd/none

Optional (when box stats enabled):

* `NDVI_P05`, `NDVI_P25`, `NDVI_P50`, `NDVI_P75`, `NDVI_P95`
* `BOX_KIND = temporal_percentiles_within_interval`

---

## Running an extraction

1. Put a locations file in the repo folder (e.g., `locations.csv`)
2. Edit `config.yaml` (time range, unit/step, agg, plot_type, etc.)
3. Run:

```bash
python gee_ndvi_timeseries.py
```

Outputs will be in:

* `runs/YYYYMMDD_HHMMSS/`

---

## Troubleshooting notes

* Large row counts are expected:

  * `rows ≈ N_locations × N_time_windows`
* If runs become slow or fail for large location counts:

  * set `location_chunk_size` to a value like 200–1000
  * reduce the time range or use larger intervals (`unit: year`)
  * disable box stats if not needed (`compute_box_stats: false`)

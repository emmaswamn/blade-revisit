# Judge / reproduce kit

Run this project **without our S3 bucket**. Images come from the public DTU dataset.

**Data:** [DTU – Drone inspection images of wind turbine](https://data.mendeley.com/datasets/hd96prn3nc/1)  
**Licence:** CC BY 4.0 · **DOI:** [10.17632/hd96prn3nc.1](https://doi.org/10.17632/hd96prn3nc.1)  
**Authors:** ASM Shihavuddin, Xiao Chen

---

## 1. Clone & Python env

```bash
git clone <this-repo-url> blade-revisit
cd blade-revisit
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Download images (Mendeley, file-by-file)

There is no single convenient zip; the script lists each year folder and downloads JPGs.

```bash
mkdir -p "$HOME/data/wind-turbine-dtu"
python scripts/download_dtu_mendeley.py --out "$HOME/data/wind-turbine-dtu"
```

- Re-run is safe: existing files with matching size are skipped.  
- Optional: `--year 2017` or `--year 2018`, `--dry-run` to only list counts.  
- Expect a long first run (hundreds of large JPGs).

Layout after download:

```text
~/data/wind-turbine-dtu/
  2017/*.JPG
  2018/*.JPG
```

Override later with `export BLADE_DATA_ROOT=/path/to/wind-turbine-dtu`.

## 3. Start the API (local)

```bash
export BLADE_DATA_ROOT="$HOME/data/wind-turbine-dtu"
export PYTHONPATH=src
uvicorn api.main:app --app-dir . --host 127.0.0.1 --port 8765
```

Open:

- Review UI (15 pairs): http://127.0.0.1:8765/review  
- Stage debug: http://127.0.0.1:8765/  
- Rust search: http://127.0.0.1:8765/match  

UI language: English by default (toggle in page).

## 4. Optional: prewarm review JPEGs

Speeds up `/review` step images (writes under `data/debug_preview/`).

```bash
export BLADE_DATA_ROOT="$HOME/data/wind-turbine-dtu"
export PYTHONPATH=src
python scripts/prewarm_review.py
```

## 5. What not to do

- Do **not** need `BLADE_S3_BUCKET` or AWS credentials for local reproduce.  
- `scripts/sync_*_s3.py` is for our cloud mirror only — skip it.  
- Basic Auth (`BLADE_BASIC_*`) is only for a public demo host; leave unset locally.

## Cite the data

If you use the images, cite the Mendeley dataset (DOI above) and the original paper as appropriate (Shihavuddin & Chen; see dataset page).

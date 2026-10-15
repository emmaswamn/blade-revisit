# blade-revisit

Year-over-year wind-turbine inspection revisit: OpenCV 5 → tiered decisions.

**Judges / external reproduce (no our S3):** follow [`judge/README.md`](judge/README.md)  
— download DTU images from [Mendeley](https://data.mendeley.com/datasets/hd96prn3nc/1) (CC BY 4.0), then run locally.

## Quick start (local)

```bash
cd blade-revisit
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Public DTU JPGs → ~/data/wind-turbine-dtu/{2017,2018}
python scripts/download_dtu_mendeley.py

export BLADE_DATA_ROOT="$HOME/data/wind-turbine-dtu"
export PYTHONPATH=src
uvicorn api.main:app --app-dir . --host 127.0.0.1 --port 8765
# http://127.0.0.1:8765/review
```

Optional smoke search:

```bash
python worker/run_local.py --query DJI_0004.JPG --top-k 5 --ref-limit 30
```

`BLADE_DATA_ROOT` / `BLADE_DB` override data and SQLite paths.

## Layout

| Path | Role |
|---|---|
| `judge/README.md` | Reproduce for judges (Mendeley download, no S3) |
| `scripts/download_dtu_mendeley.py` | File-by-file DTU download |
| `src/blade/` | Decision core |
| `worker/` | Local batch jobs |
| `api/` + `web/` | Review & debug UI |
| `configs/thresholds.yaml` | Thresholds |

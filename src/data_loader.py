import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

UCI_OULAD_ZIP = (
    "https://archive.ics.uci.edu/static/public/349/"
    "open+university+learning+analytics+dataset.zip"
)


def _csv_source(base: str | Path, filename: str) -> str | Path:
    base_str = str(base).rstrip("/")
    if base_str.startswith("http"):
        return f"{base_str}/{filename}"
    return Path(base_str) / filename


def ensure_data(
    raw_dir: Path | None = None,
    release_url: str | None = None,
) -> Path:
    raw_dir = Path(raw_dir or RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if (raw_dir / "studentInfo.csv").exists():
        return raw_dir

    zip_path = raw_dir / "oulad.zip"
    sources = [url for url in [release_url, UCI_OULAD_ZIP] if url]

    for source in sources:
        print(f"Downloading OULAD from {source} ...")
        try:
            urlretrieve(source, zip_path)
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(raw_dir)
            print(f"Data extracted to {raw_dir}")
            return raw_dir
        except Exception as exc:
            print(f"Download failed: {exc}")

    raise FileNotFoundError(
        "Could not download OULAD. Upload oulad.zip to a GitHub Release "
        "and set OULAD_RELEASE_URL in the notebook."
    )


def load_student_info(data_source: str | Path | None = None) -> pd.DataFrame:
    source = data_source or RAW_DIR
    path = _csv_source(source, "studentInfo.csv")
    if not str(path).startswith("http") and not Path(path).exists():
        raise FileNotFoundError(f"studentInfo.csv not found at {path}")
    return pd.read_csv(path)


def aggregate_vle_features(data_source: str | Path | None = None) -> pd.DataFrame:
    source = data_source or RAW_DIR
    path = _csv_source(source, "studentVle.csv")
    if not str(path).startswith("http") and not Path(path).exists():
        return pd.DataFrame()

    vle = pd.read_csv(path)
    return (
        vle.groupby(["code_module", "code_presentation", "id_student"])
        .agg(
            total_clicks=("sum_click", "sum"),
            active_days=("date", "nunique"),
            unique_activities=("id_site", "nunique"),
        )
        .reset_index()
    )


def aggregate_assessment_features(data_source: str | Path | None = None) -> pd.DataFrame:
    source = data_source or RAW_DIR
    sa_path = _csv_source(source, "studentAssessment.csv")
    ass_path = _csv_source(source, "assessments.csv")

    if not str(sa_path).startswith("http"):
        if not Path(sa_path).exists() or not Path(ass_path).exists():
            return pd.DataFrame()

    sa = pd.read_csv(sa_path)
    ass = pd.read_csv(ass_path)[["id_assessment", "code_module", "code_presentation"]]
    merged = sa.merge(ass, on="id_assessment", how="left")
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce")
    merged = merged[merged["score"].notna()]

    return (
        merged.groupby(["code_module", "code_presentation", "id_student"])
        .agg(
            mean_score=("score", "mean"),
            max_score=("score", "max"),
            min_score=("score", "min"),
            num_assessments=("score", "count"),
        )
        .reset_index()
    )


def build_modeling_dataset(
    data_source: str | Path | None = None,
    include_vle: bool = True,
) -> pd.DataFrame:
    source = data_source or RAW_DIR
    df = load_student_info(source)
    keys = ["code_module", "code_presentation", "id_student"]

    if include_vle:
        vle_feats = aggregate_vle_features(source)
        if not vle_feats.empty:
            df = df.merge(vle_feats, on=keys, how="left")

    assess_feats = aggregate_assessment_features(source)
    if not assess_feats.empty:
        df = df.merge(assess_feats, on=keys, how="left")

    return df

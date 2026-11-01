import argparse
import shutil
from pathlib import Path

from pig.config import load_config

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["M1_tfidf.joblib", "M2_semantic.joblib"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "space_build"))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "models").mkdir(parents=True)
    for f in ["app.py", "requirements.txt", "README.md"]:
        shutil.copy(ROOT / "app" / f, out / f)
    shutil.copytree(ROOT / "src" / "pig", out / "src" / "pig", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "configs", out / "configs")
    for m in MODELS:
        src = cfg.path("models") / m
        if not src.exists():
            raise FileNotFoundError(f"{src} not found: train the model first")
        shutil.copy(src, out / "models" / m)
    print(f"Space prepared in {out}")


if __name__ == "__main__":
    main()

import argparse
import logging

from pig.config import load_config
from pig.pipeline import build_cascade, load_data, load_models, train_classic, train_transformer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stage", choices=["classic", "transformer", "cascade", "all"], default="classic")
    ap.add_argument("--embedder", choices=["sentence", "hashing"], default="sentence")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    splits = load_data(cfg)
    if args.stage in ("classic", "all"):
        for name, m in train_classic(cfg, splits, args.embedder).items():
            print(f"{name}: threshold={m.threshold:.4f}")
    if args.stage in ("transformer", "all"):
        det, _ = train_transformer(cfg, splits)
        print(f"M3_transformer: threshold={det.threshold:.4f}")
    if args.stage in ("cascade", "all"):
        models = load_models(cfg, args.embedder)
        _, info = build_cascade(cfg, models["M2_semantic"], models["M3_transformer"], splits["val"])
        print(info)


if __name__ == "__main__":
    main()

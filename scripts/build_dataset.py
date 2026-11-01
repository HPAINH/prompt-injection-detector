import argparse
import json
import logging

from pig.config import load_config
from pig.data.build import build_dataset
from pig.eval.plots import plot_eda
from pig.features.embedder import make_embedder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-hf", action="store_true")
    ap.add_argument("--embedder", choices=["sentence", "hashing", "none"], default="sentence")
    ap.add_argument("--translated", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    emb = None if args.embedder == "none" else make_embedder(cfg, args.embedder, cfg.path("cache"))
    translated = None
    if args.translated:
        import pandas as pd

        translated = pd.read_parquet(args.translated)
    df, stats = build_dataset(cfg, use_hf=not args.no_hf, embedder=emb, translated=translated,
                              out_dir=cfg.path("processed"))
    plot_eda(df, cfg.path("reports"))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

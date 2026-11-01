import argparse
import logging

from pig.config import load_config
from pig.eval.metrics import summary_table
from pig.pipeline import ablation_table, evaluate_all, load_data, load_models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--embedder", choices=["sentence", "hashing"], default="sentence")
    ap.add_argument("--no-transformer", action="store_true")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--latency-n", type=int, default=200)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    splits = load_data(cfg)
    models = load_models(cfg, args.embedder, not args.no_transformer, args.reference)
    res = evaluate_all(cfg, models, splits, args.latency_n, splits["val"])
    print(summary_table(res["test"]).to_string(index=False))
    print()
    print(summary_table(res["heldout"]).to_string(index=False))
    print()
    print(ablation_table(res).to_string(index=False))


if __name__ == "__main__":
    main()

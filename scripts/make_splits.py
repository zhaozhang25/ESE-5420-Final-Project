"""Create the shared split file. Refuses to replace a different existing split."""

import argparse

from shoppers.config import load_config
from shoppers.data import load_dataset
from shoppers.splits import compute_splits, write_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true",
                        help="replace a different split; every result must then be rerun")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _, y = load_dataset(cfg.data_path, cfg.drop_duplicates)
    table = compute_splits(y, cfg.test_size, cfg.n_folds, cfg.seed)
    write_splits(table, cfg.splits_path, force=args.force)

    dev = table[table.role == "dev"]
    print(f"{cfg.name}: dev {len(dev)}, test {len(table) - len(dev)}, "
          f"fold sizes {dev.fold.value_counts().sort_index().tolist()}")


if __name__ == "__main__":
    main()

import argparse
import pig.evaluation

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", type=str, nargs="+")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    pig.evaluation.full_run(gpus=1, versions=args.versions)


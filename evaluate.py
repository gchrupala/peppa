import argparse
import evaluation

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", type=str, nargs="+")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    evaluation.full_run(args.gpus, args.versions)


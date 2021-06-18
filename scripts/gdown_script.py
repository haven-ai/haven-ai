import tqdm
import argparse
import os
import gdown

from haven import haven_examples as he
from haven import haven_wizard as hw
from haven import haven_results as hr


if __name__ == "__main__":
    # Specify arguments regarding save directory and job scheduler
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        required=False,
        help="Define the experiment group to run.",
    )
    parser.add_argument("--path", required=False, help="Define the base directory where the experiments will be saved.")

    args, others = parser.parse_known_args()
    args.url = "https://drive.google.com/u/0/uc?id=13HGW3_zCTKHGVtr_JDHD4Wv64PP5Z2mG&export=download"
    args.path = "/mnt/public/datasets2/kitti/cadnn.pth"
    gdown.cached_download(args.url, path=args.path, quiet=False, proxy=False)

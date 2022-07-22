import argparse
import json
import os
import sys

python_module_path = os.path.join(__file__, os.pardir, os.pardir, "python")
sys.path.append(os.path.abspath(python_module_path))

from livseg.data_management.s3utils import download_data

parser = argparse.ArgumentParser()

parser.add_argument("--target", dest="data_dir", type=str)
parser.add_argument("--prefix", dest="prefix", type=str)

args = parser.parse_args()

with open(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "bucket_config.json")), "r") as f:
    bucketconfig = json.load(f)
    all(bucketconfig[k] != "" for k in ["service_name", "region_name", "bucketname"])

download_data(bucketconfig, prefix=args.prefix, target=args.data_dir)

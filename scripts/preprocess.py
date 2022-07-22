"""Perform the preprocessing task"""
import argparse
import os
import shutil
import sys

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i", "--input_dir", dest="input_dir", type=str, help="Data directory"
)

parser.add_argument(
    "-t", "--target_dir", dest="target_dir", type=str, help="Target data directory"
)

parser.add_argument(
    "-f",
    "--force",
    dest="force",
    action="store_true",
    help="Force creation of temp dir",
)

python_module_path = os.path.join(__file__, os.pardir, os.pardir, "python")
sys.path.append(os.path.abspath(python_module_path))

from livseg.data_management.loader import load_and_merge_ct_segmentations_paths
from livseg.data_management.preprocessing import preprocess_from_df


args = parser.parse_args()
path = args.input_dir
target_dir = args.target_dir
force = args.force

df = load_and_merge_ct_segmentations_paths(path)

if force and os.path.exists(target_dir):
    shutil.rmtree(target_dir)

print(f'{"#"*5} Preprocess {"#"*5}')
preprocess_from_df(df, target_dir)

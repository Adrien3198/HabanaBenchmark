"""Creates train test directories"""
import argparse
import os
import sys

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i", "--input_dir", dest="input_dir", type=str, help="Data directory"
)

parser.add_argument(
    "-t", "--target_dir", dest="target_dir", type=str, help="Data directory"
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

from livseg.data_management.dataset_handler import DatasetHandler


args = parser.parse_args()

DatasetHandler.create_directory(args.input_dir, args.target_dir, args.force)

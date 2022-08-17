import argparse
import logging
import shlex
import subprocess
import time
from logging.handlers import RotatingFileHandler
from os import listdir
from os.path import isfile, join

logger = logging.getLogger("pyveg_bulk_donwload_job")
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
logger.setLevel(logging.INFO)

c_handler = logging.StreamHandler()
c_handler.setFormatter(formatter)
f_handler = RotatingFileHandler(
    "pyveg_bulk_download_job_{}.log".format(time.strftime("%Y-%m-%d_%H-%M-%S")),
    maxBytes=5 * 1024 * 1024,
    backupCount=10,
)
f_handler.setFormatter(formatter)

logger.addHandler(f_handler)
logger.addHandler(c_handler)


def run_pipeline(config_directory):
    # Directory containing input files

    # Get the full paths to the files in the input dir
    full_paths = [
        join(config_directory, f)
        for f in listdir(config_directory)
        if isfile(join(config_directory, f))
    ]

    # Build a list of commands to reproject each file individually
    cmds = []
    for input_fpath in full_paths[:5]:
        safe_input = shlex.quote(str(input_fpath))

        cmds.append(f"pyveg_run_pipeline --config_file {safe_input}")

    # Now run those commands (This might be expensive)

    failed = 0
    for cmd in cmds:
        logger.info(f"Running gee download using the command: {cmd}")
        try:
            subprocess.run(cmd, shell=True)

        except subprocess.SubprocessError as e:
            failed += 1
            logger.error(f"Download using the command: {cmd} failed")

    return failed


def main():

    # run pyveg pipeline on a loop by running all the config files in a given directory
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dir", help="Path to directory with config files", required=True
    )

    args = parser.parse_args()
    n = run_pipeline(args.config_dir)
    logger.info(f"Bulk download finished. Number of failed dowloads {n}")


if __name__ == "__main__":
    main()

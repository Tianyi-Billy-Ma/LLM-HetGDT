import sys

sys.dont_write_bytecode = True


import os
import logging
import shutil
import stat
import zipfile
import logging
import json
import pickle
from pytorch_lightning.utilities.rank_zero import rank_zero_only

logger = logging.getLogger(__name__)


def load_file(file_path):
    if file_path.endswith(".json"):
        return load_json(file_path)
    elif file_path.endswith(".pkl"):
        return load_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def load_json(file_path):
    with open(file_path, "rb") as f:
        data = json.load(f)
        logger.info(f"Loaded Json: {file_path}")
    return data


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        logger.info(f"Loaded Pickle: {file_path}")
    return data


@rank_zero_only
def reset_folders(dirs):
    for dir in dirs:
        try:
            delete_dir(dir)
        except Exception as e:
            print(e)


@rank_zero_only
def reset_wandb_runs(all_runs):
    for run in all_runs:
        logger.info(f"Deleting wandb run: {run}")
        run.delete()


def zipDir(dirpath, outFullName):
    """
    zip folder
    :param dirpath: target folder path
    :param outFullName: target file path (.zip)
    :return:
    """
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        fpath = path.replace(dirpath, "")
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return:
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        logging.getLogger("Dirs Creator").info(
            "Creating directories error: {0}".format(err)
        )
        exit(-1)


def delete_dir(filePath):
    if os.path.exists(filePath):
        for fileList in os.walk(filePath):
            for name in fileList[2]:
                os.chmod(os.path.join(fileList[0], name), stat.S_IWRITE)
                os.remove(os.path.join(fileList[0], name))
        shutil.rmtree(filePath)
        return "delete ok"
    else:
        return "no filepath"

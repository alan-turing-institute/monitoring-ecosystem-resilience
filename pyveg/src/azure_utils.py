import os
import io
import json

import arrow
import re
import tempfile
from PIL import Image

from pyveg.src.file_utils import split_filepath

# load the azure configuration if we have the azure_config.py file
try:
    from pyveg.azure_config import config
except:
    print(
        """
    azure_config.py not found - this is needed for using Azure storage or batch.
    Copy pyveg/azure_config_template.py to pyveg/azure_config.py then input your
    own values for Azure Storage account name and Access key, then redo `pip install .`
    """
    )

from azure.storage.blob import (
    BlockBlobService,
    PublicAccess,
    ContainerPermissions
)
from azure.common import AzureMissingResourceHttpError


def sanitize_container_name(orig_name):
    """
    only allowed alphanumeric characters and dashes.
    """
    sanitized_name = ""
    previous_character = None
    for character in orig_name:
        if not re.search("[-a-zA-Z\d]", character):
            if not previous_character == "-":
                sanitized_name += "-"
                previous_character = "-"
            else:
                continue
        else:
            sanitized_name += character.lower()
            previous_character = character
    if "\\" in sanitized_name:
        sanitized_name = sanitized_name.replace("\\","/")

    return sanitized_name


def check_container_exists(container_name, bbs=None):
    """
    See if a container already exists for this account name.
    """
    if not bbs:
        bbs = BlockBlobService(
            account_name=config["storage_account_name"],
            account_key=config["storage_account_key"],
        )
    return bbs.exists(container_name)


def create_container(container_name, bbs=None):
    if not bbs:
        bbs = BlockBlobService(
            account_name=config["storage_account_name"],
            account_key=config["storage_account_key"],
        )
    exists = check_container_exists(container_name, bbs)
    if not exists:
        bbs.create_container(container_name)


def check_blob_exists(blob_name, container_name, bbs=None):
    """
    See if a blob already exists for this account name.
    """
    if not bbs:
        bbs = BlockBlobService(
            account_name=config["storage_account_name"],
            account_key=config["storage_account_key"],
        )
    blob_names = bbs.list_blob_names(container_name)
    return blob_name in blob_names


def get_sas_token(container_name, token_duration=1, permissions="READ", bbs=None):
    if not bbs:
        bbs = BlockBlobService(
            account_name=config["storage_account_name"],
            account_key=config["storage_account_key"],
        )
    token_permission = (
        ContainerPermissions.WRITE
        if permissions == "WRITE"
        else ContainerPermissions.READ
    )
    duration = token_duration  # days
    token = bbs.generate_container_shared_access_signature(
        container_name=container_name,
        permission=token_permission,
        protocol="https",
        start=arrow.utcnow().shift(hours=-1).datetime,
        expiry=arrow.utcnow().shift(hours=token_duration).datetime,
    )
    return token


def retrieve_blob(blob_name, container_name, destination="/tmp/", bbs=None):
    """
    use the BlockBlobService to retrieve file from Azure, and place in destination folder.
    """
    if not bbs:
        bbs = BlockBlobService(
            account_name=config["storage_account_name"],
            account_key=config["storage_account_key"],
        )
    local_filename = blob_name.split("/")[-1]
    try:
        bbs.get_blob_to_path(
            container_name, blob_name, os.path.join(destination, local_filename)
        )
        return True, "retrieved script OK"
    except (AzureMissingResourceHttpError):
        return False, "failed to retrieve {} from {}".format(blob_name, container_name)
    return os.path.join(destination, local_filename)


def list_directory(path, container_name, bbs=None):
    if not bbs:
        bbs = BlockBlobService(
            account_name=config["storage_account_name"],
            account_key=config["storage_account_key"],
        )
        pass
    output_names = []
    prefix = remove_container_name_from_blob_path(path, container_name)
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    blob_names = bbs.list_blob_names(container_name, prefix=prefix, delimiter="/")
    blob_names = [bn[:-1] if bn.endswith("/") else bn for bn in blob_names]
    return [os.path.basename(bn) for bn in blob_names]


def remove_container_name_from_blob_path(blob_path, container_name):
    """
    Get the bit of the filepath after the container name.
    """
    # container name will often be part of filepath - we want
    # the blob name to be the bit after that
    if not container_name in blob_path:
        return blob_path
    blob_name_parts = []
    filepath_parts = split_filepath(blob_path)
    container_name_found = False
    for path_part in filepath_parts:
        if container_name_found:
            blob_name_parts.append(path_part)
        if path_part == container_name:
            container_name_found = True
    if len(blob_name_parts) == 0:
        return ""
    return "/".join(blob_name_parts)


def delete_blob(blob_name, container_name, bbs=None):
    if not bbs:
        bbs = BlockBlobService(
            account_name=config["storage_account_name"],
            account_key=config["storage_account_key"],
        )
    blob_exists = check_blob_exists(blob_name, container_name, bbs)
    if not blob_exists:
        return
    bbs.delete_blob(container_name, blob_name)


def write_file_to_blob(file_path, blob_name, container_name, bbs=None):
    if not bbs:
        bbs = BlockBlobService(
            account_name=config["storage_account_name"],
            account_key=config["storage_account_key"],
        )
    bbs.create_blob_from_path(container_name, blob_name, file_path)


def write_files_to_blob(
    path, container_name, blob_path=None, file_endings=[], bbs=None
):
    """
    Upload a whole directory structure to blob storage.
    If we are given 'blob_path' we use that - if not we preserve the given file path structure.
    In both cases we take care to remove the container name from the start of the blob path
    """

    if not bbs:
        bbs = BlockBlobService(
            account_name=config["storage_account_name"],
            account_key=config["storage_account_key"],
        )
    filepaths_to_upload = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            filepath = os.path.join(root, filename)
            if file_endings:
                for ending in file_endings:
                    if filename.endswith(ending):
                        filepaths_to_upload.append(filepath)
            else:
                filepaths_to_upload.append(filepath)
    for filepath in filepaths_to_upload:
        if blob_path:
            blob_fullpath = os.path.join(blob_path, os.path.split(filepath)[-1])
        else:
            blob_fullpath = filepath
        blob_name = remove_container_name_from_blob_path(blob_fullpath, container_name)

        write_file_to_blob(filepath, blob_name, container_name, bbs)


def save_image(
    image, output_location, output_filename, container_name, format="png", bbs=None
):
    """
    Given a PIL.Image (list of pixel values), save
    to requested filename - note that the file extension
    will determine the output file type, can be .png, .tif,
    probably others...
    """
    if not bbs:
        bbs = BlockBlobService(
            account_name=config["storage_account_name"],
            account_key=config["storage_account_key"],
        )
    output_path = os.path.join(output_location, output_filename)
    blob_name = remove_container_name_from_blob_path(output_path, container_name)
    im_bytes = io.BytesIO()
    image.save(im_bytes, format=format)
    bbs.create_blob_from_bytes(container_name, blob_name, im_bytes.getvalue())


def read_image(blob_name, container_name, bbs=None):
    if not bbs:
        bbs = BlockBlobService(
            account_name=config["storage_account_name"],
            account_key=config["storage_account_key"],
        )
    blob_name = remove_container_name_from_blob_path(blob_name, container_name)
    img_bytes = bbs.get_blob_to_bytes(container_name, blob_name)
    image = Image.open(io.BytesIO(img_bytes.content))
    return image


def save_json(data, blob_path, filename, container_name, bbs=None):
    if not bbs:
        bbs = BlockBlobService(
            account_name=config["storage_account_name"],
            account_key=config["storage_account_key"],
        )
    blob_name = os.path.join(blob_path, filename)
    blob_name = remove_container_name_from_blob_path(blob_name, container_name)
    bbs.create_blob_from_text(container_name, blob_name, json.dumps(data))


def read_json(blob_name, container_name, bbs=None):
    if not bbs:
        bbs = BlockBlobService(
            account_name=config["storage_account_name"],
            account_key=config["storage_account_key"],
        )
    blob_name = remove_container_name_from_blob_path(blob_name, container_name)
    data_blob = bbs.get_blob_to_text(container_name, blob_name)
    data = json.loads(data_blob.content)
    return data


def get_blob_to_tempfile(filename, container_name, bbs=None):
    if not bbs:
        bbs = BlockBlobService(
            account_name=config["storage_account_name"],
            account_key=config["storage_account_key"],
        )
    blob_name = remove_container_name_from_blob_path(filename, container_name)
    td = tempfile.mkdtemp()
    output_name = os.path.join(td, os.path.basename(filename))
    bbs.get_blob_to_path(container_name, blob_name, output_name)
    return output_name


def download_summary_json(container, json_dir):
    """
    Parameters
    ==========
    container: str, the container name
    json_dir: str, temporary directory into which to put json file.
    """

    print("Getting summary JSON file  to {}".format(json_dir))
    blob_dirs = list_directory(container, container)
    json_blob_dir = None
    for b in blob_dirs:
        if b.endswith("combine"):
            json_blob_dir = b
    json_blob_file = list_directory(json_blob_dir, container)[0]
    blob_path = "/".join([json_blob_dir, json_blob_file])
    print("Will retrieve blob {}".format(blob_path))
    retrieve_blob(blob_path, container, json_dir)


def download_rgb(container, rgb_dir):
    """
    Parameters
    ==========
    container: str, the container name
    rgb_dir: str, directory into which to put image files.
    """
    print("Getting RGB images to {}".format(rgb_dir))
    bbs = BlockBlobService(
        account_name=config["storage_account_name"],
        account_key=config["storage_account_key"],
    )
    blob_names = bbs.list_blob_names(container)
    rgb_names = [b for b in blob_names if "PROCESSED" in b and b.endswith("RGB.png")]
    print("Found {} images".format(len(rgb_names)))
    for blob in rgb_names:
        retrieve_blob(blob, container, rgb_dir)

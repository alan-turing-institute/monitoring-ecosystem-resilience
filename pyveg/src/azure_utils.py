import os
import arrow
import re

from pyveg.src.file_utils import split_filepath

# load the azure configuration if we have the azure_config.py file
try:
    from pyveg.azure_config import config
except:
    print("""
    azure_config.py not found - this is needed for using Azure storage or batch.
    Copy pyveg/azure_config_template.py to pyveg/azure_config.py then input your
    own values for Azure Storage account name and Access key, then redo `pip install .`
    """)

from azure.storage.blob import BlockBlobService, PublicAccess, ContainerPermissions
from azure.common import AzureMissingResourceHttpError


def sanitize_container_name(orig_name):
    """
    only allowed alphanumeric characters and dashes.
    """
    sanitized_name = ""
    previous_character = None
    for character in orig_name:
        if not re.search("[-a-zA-Z\d]",character):
           if not previous_character == "-":
               sanitized_name += "-"
               previous_character = "-"
           else:
               continue
        else:
            sanitized_name += character
            previous_character = character
    return sanitized_name


def check_container_exists(container_name, bbs=None):
    """
    See if a container already exists for this account name.
    """
    if not bbs:
        bbs = BlockBlobService(account_name=config["account_name"],
                               account_key=config["account_key"])
    return bbs.exists(container_name)


def create_container(container_name, bbs=None):
    if not bbs:
        bbs = BlockBlobService(account_name=config["account_name"],
                               account_key=config["account_key"])
    exists = check_container_exists(container_name, bbs)
    if not exists:
        bbs.create_container(container_name)


def get_sas_token(container_name, token_duration=1, permissions="READ", bbs=None):
    if not bbs:
        bbs = BlockBlobService(account_name=config["account_name"],
                               account_key=config["account_key"])
    token_permission = ContainerPermissions.WRITE if permissions=="WRITE" \
                           else ContainerPermissions.READ
    duration = token_duration # days
    token = bbs.generate_container_shared_access_signature(
        container_name=container_name,
        permission=token_permission,
        protocol='https',
        start=arrow.utcnow().shift(hours=-1).datetime,
        expiry=arrow.utcnow().shift(hours=token_duration).datetime)
    return token


def retrieve_blob(blob_name, container_name, destination="/tmp/", bbs=None):
    """
    use the BlockBlobService to retrieve file from Azure, and place in destination folder.
    """
    if not bbs:
        bbs = BlockBlobService(account_name=config["account_name"],
                                   account_key=config["account_key"])
    local_filename = blob_name.split("/")[-1]
    try:
        bbs.get_blob_to_path(container_name,
                             blob_name,
                             os.path.join(destination,
                                          local_filename))
        return True, 'retrieved script OK'
    except(AzureMissingResourceHttpError):
        return False, 'failed to retrieve {} from {}'.format(blob_name,
                                                             container_name)
    return os.path.join(destination, local_filename)


def list_directory(path, container_name, bbs=None):
    if not bbs:
        bbs = BlockBlobService(account_name=config["account_name"],
                               account_key=config["account_key"])
        pass
    output_names = []
    blob_names = bbs.list_blob_names(container_name)
    for blob_name in blob_names:
        blob_path = os.path.join(container_name, blob_name)
        if blob_path.startswith(path):
            blob_path_parts = split_filepath(blob_path)
            n_parts_in_path = len(split_filepath(path))
            output_name = blob_path_parts[n_parts_in_path]
            if output_name not in output_names:
                output_names.append(output_name)
    return output_names



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
        return None
    return os.path.join(*blob_name_parts)


def write_file_to_blob(file_path, blob_name, container_name, bbs=None):
    if not bbs:
        bbs = BlockBlobService(account_name=config["account_name"],
                               account_key=config["account_key"])
    bbs.create_blob_from_path(container_name, blob_name, file_path)


def write_files_to_blob(path, container_name, blob_path = None, file_endings = [], bbs=None):
    """
    Upload a whole directory structure to blob storage.
    If we are given 'blob_path' we use that - if not we preserve the given file path structure.
    In both cases we take care to remove the container name from the start of the blob path
    """

    if not bbs:
        bbs = BlockBlobService(account_name=config["account_name"],
                               account_key=config["account_key"])
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
        print("Will write {} to blob {}".format(filepath, blob_name))

        write_file_to_blob(filepath, blob_name, container_name, bbs)

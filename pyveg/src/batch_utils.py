"""
Functions for submitting batch jobs.  Currently only support Azure Batch.
Largely taken from https://github.com/Azure-Samples/batch-python-quickstart
"""
import os
import sys
import json
import tempfile
import datetime

import azure.storage.blob as azureblob
import azure.batch.batch_service_client as batch
import azure.batch.batch_auth as batch_auth
import azure.batch.models as batchmodels
from azure.storage.blob import BlockBlobService

try:
    from pyveg.azure_config import config
except:
    print("""
    azure_config.py not found - this is needed for using Azure storage or batch.
    Copy pyveg/azure_config_template.py to pyveg/azure_config.py then input your
    own values for Azure Storage account name and Access key, then redo `pip install .`
    """)


def submit_tasks(list_of_configs, job_name):
    """
    Submit batch jobs to Azure batch.

    Parameter
    =========
    list_of_configs: list of dict, as proveded by BaseModule.get_config()
    job_name: str, should identify the module generating the jobs
    """
    # check we have the azure config file
    azure_config_filename = os.path.join(os.path.dirname(__file__),"..","azure_config.py")
    if not os.path.exists(azure_config_filename):
        raise RuntimeError("azure_config.py not found - will not be able to run batch jobs.")
    # create the batch service client to perform batch operations
    batch_service_client = create_batch_client()
    # create block blob service to upload config files
    blob_client = BlockBlobService(account_name=config["storage_account_name"],
                                   account_key=config["storage_account_key"])
    # create a pool of worker nodes if it doesn't already exist
    try:
        create_pool(config["batch_pool_id"], batch_service_client)
    except:
        print("pool already exists")
        pass
    # create a job - module name plus timestamp
    try:
        create_job(job_name, config["batch_pool_id"], batch_service_client)
    except:
        print("job already exists")
        pass
    # create a container on the storage account for uploading config to
    config_container_name = 'input'
    blob_client.create_container(config_container_name, fail_on_exist=False)
    # upload the azure config to this, as the batch job will need it.
    input_azure_config = upload_file_to_container(blob_client,
                                                  config_container_name,
                                                  azure_config_filename)
    # upload the batch_commands.sh script
    script_filename = os.path.join(os.path.dirname(__file__),"..","scripts","batch_commands.sh")
    input_script = upload_file_to_container(blob_client,
                                            config_container_name,
                                            script_filename)
    # temporary dir to store json files before uploading to storage account
    tmp_json_dir = tempfile.mkdtemp()
    task_ids = []
    for i, c in enumerate(list_of_configs):
        config_filename = os.path.join(tmp_json_dir, "config_{}_{}.json".format(job_name, i))
        with open(config_filename,"w") as outfile:
            json.dump(c, outfile)
        input_config = upload_file_to_container(blob_client,
                                                config_container_name,
                                                config_filename)
        task_id = "task_{}".format(i)
        add_task(task_id,
                 job_name,
                 input_script,
                 input_config,
                 input_azure_config,
                 batch_service_client)
        task_ids.append(task_id)
    return task_ids


def add_task(task_id, job_name,
             input_script,
             input_config,
             input_azure_config,
             batch_service_client=None):
    """
    add the batch task to the job.
    """
    if not batch_service_client:
        batch_service_client = create_batch_client()
    command = "/bin/bash {} {} {}".format(
        input_script.file_path,
        input_config.file_path,
        input_azure_config.file_path)
    print("Adding task {} to job {}".format(task_id, job_name))
    user = batch.models.UserIdentity(
        auto_user=batch.models.AutoUserSpecification(
            elevation_level=batch.models.ElevationLevel.admin,
            scope=batch.models.AutoUserScope.task))
    task = batch.models.TaskAddParameter(
        id=task_id,
        command_line=command,
        user_identity=user,
        resource_files=[input_script, input_config, input_azure_config])
    batch_service_client.task.add(job_name, task)


def wait_for_tasks_to_complete(job_id, timeout=60, batch_service_client=None):
    """
    Returns when all tasks in the specified job reach the Completed state.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str job_id: The id of the job whose tasks should be to monitored.
    :param timedelta timeout: The duration to wait for task completion. If all
    tasks in the specified job do not reach Completed state within this time
    period, an exception will be raised.
    """
    if not batch_service_client:
        batch_service_client = create_batch_client()
    timeout_expiration = datetime.datetime.now() + datetime.timedelta(minutes=timeout)

    print("Monitoring all tasks for 'Completed' state, timeout in {}..."
          .format(timeout), end='')

    while datetime.datetime.now() < timeout_expiration:
        print('.', end='')
        sys.stdout.flush()
        tasks = batch_service_client.task.list(job_id)

        incomplete_tasks = [task for task in tasks if
                            task.state != batchmodels.TaskState.completed]
        if not incomplete_tasks:
            print()
            return True
        else:
            time.sleep(1)

    print()
    raise RuntimeError("ERROR: Tasks did not reach 'Completed' state within "
                       "timeout period of " + str(timeout))


def create_pool(pool_id, batch_service_client=None):
    """
    Creates a pool of compute nodes.

    Parameters
    ==========
    pool_id: str, identifier for the pool
    batch_service_client: azure.batch.BatchServiceClient, A Batch service client.
    """
    print('Creating pool [{}]...'.format(pool_id))
    if not batch_service_client:
        batch_service_client = create_batch_client()
    new_pool = batch.models.PoolAddParameter(
        id=pool_id,
        virtual_machine_configuration=batchmodels.VirtualMachineConfiguration(
            image_reference=batchmodels.ImageReference(
                publisher="Canonical",
                offer="UbuntuServer",
                sku="18.04-LTS",
                version="latest"
            ),
            node_agent_sku_id="batch.node.ubuntu 18.04"),
        vm_size=config["pool_vm_size"],
        target_low_priority_nodes=config["pool_low_priority_node_count"],
        target_dedicated_nodes=config["pool_dedicated_node_count"]
    )
    batch_service_client.pool.add(new_pool)


def create_job(job_id, pool_id, batch_service_client=None):
    """
    Creates a job with the specified ID, associated with the specified pool.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str job_id: The ID for the job.
    :param str pool_id: The ID for the pool.
    """
    print('Creating job [{}]...'.format(job_id))
    if not batch_service_client:
        batch_service_client = create_batch_client()
    job = batch.models.JobAddParameter(
        id=job_id,
        pool_info=batch.models.PoolInformation(pool_id=pool_id))

    batch_service_client.job.add(job)



def create_batch_client():
    credentials = batch_auth.SharedKeyCredentials(config["batch_account_name"],
                                                  config["batch_account_key"])

    batch_client = batch.BatchServiceClient(
        credentials,
        batch_url=config["batch_account_url"])
    return batch_client



def upload_file_to_container(block_blob_client, container_name, file_path):
    """
    Uploads a local file to an Azure Blob storage container.

    :param block_blob_client: A blob service client.
    :type block_blob_client: `azure.storage.blob.BlockBlobService`
    :param str container_name: The name of the Azure Blob storage container.
    :param str file_path: The local path to the file.
    :rtype: `azure.batch.models.ResourceFile`
    :return: A ResourceFile initialized with a SAS URL appropriate for Batch
    tasks.
    """
    blob_name = os.path.basename(file_path)

    print('Uploading file {} to container [{}]...'.format(file_path,
                                                          container_name))

    block_blob_client.create_blob_from_path(container_name,
                                            blob_name,
                                            file_path)

    sas_token = block_blob_client.generate_blob_shared_access_signature(
        container_name,
        blob_name,
        permission=azureblob.BlobPermissions.READ,
        expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=2))

    sas_url = block_blob_client.make_blob_url(container_name,
                                              blob_name,
                                              sas_token=sas_token)

    return batchmodels.ResourceFile(http_url=sas_url, file_path=blob_name)



def _read_stream_as_string(stream, encoding):
    """Read stream as string

    :param stream: input stream generator
    :param str encoding: The encoding of the file. The default is utf-8.
    :return: The file content.
    :rtype: str
    """
    output = io.BytesIO()
    try:
        for data in stream:
            output.write(data)
        if encoding is None:
            encoding = 'utf-8'
        return output.getvalue().decode(encoding)
    finally:
        output.close()
    raise RuntimeError('could not write data to stream or decode bytes')


def print_task_output(batch_service_client, job_id, encoding=None):
    """Prints the stdout.txt file for each task in the job.

    :param batch_client: The batch client to use.
    :type batch_client: `batchserviceclient.BatchServiceClient`
    :param str job_id: The id of the job with task output files to print.
    """

    print('Printing task output...')

    tasks = batch_service_client.task.list(job_id)

    for task in tasks:

        node_id = batch_service_client.task.get(
            job_id, task.id).node_info.node_id
        print("Task: {}".format(task.id))
        print("Node: {}".format(node_id))

        stream = batch_service_client.file.get_from_task(
            job_id, task.id, config._STANDARD_OUT_FILE_NAME)

        file_text = _read_stream_as_string(
            stream,
            encoding)
        print("Standard output:")
        print(file_text)

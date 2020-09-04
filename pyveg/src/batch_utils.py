"""
Functions for submitting batch jobs.  Currently only support Azure Batch.
Largely taken from https://github.com/Azure-Samples/batch-python-quickstart
"""
import os
import sys
import json
import tempfile
import datetime
import time

import azure.storage.blob as azureblob
import azure.batch.batch_service_client as batch
import azure.batch.batch_auth as batch_auth
import azure.batch.models as batchmodels
from azure.storage.blob import BlockBlobService

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


def prepare_for_task_submission(
    job_name, config_container_name, batch_service_client, blob_client
):
    """
    Create pool and job if not already existing, and upload the azure config file
    and the bash script used to run the batch job.

    Parameters
    ==========
    job_name: str, ID of the job
    batch_service_client: BatchServiceClient to interact with Azure batch.

    Returns
    =======
    input_azure_config, input_script: ResourceFiles corresponding to the azure_config.py
                     and batch_commands.sh scripts, uploaded to blob storage.

    """

    # check we have the azure config file
    azure_config_filename = os.path.join(
        os.path.dirname(__file__), "..", "azure_config.py"
    )
    if not os.path.exists(azure_config_filename):
        raise RuntimeError(
            "azure_config.py not found - will not be able to run batch jobs."
        )

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

    blob_client.create_container(config_container_name, fail_on_exist=False)
    # upload the azure config to this, as the batch job will need it.
    input_azure_config = upload_file_to_container(
        blob_client, config_container_name, azure_config_filename
    )
    # upload the batch_commands.sh script
    script_filename = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "batch_commands.sh"
    )
    input_script = upload_file_to_container(
        blob_client, config_container_name, script_filename
    )
    return input_azure_config, input_script


def submit_tasks(task_dicts, job_name):
    """
    Submit batch jobs to Azure batch.

    Parameter
    =========
    task_dicts: list of dicts, [ {
                       "task_id": <task_id>,
                       "config": <config_dict>,
                       "depends_on": [<task_ids>]
                                } ]
    job_name: str, should identify the sequence generating the jobs
    """
    # create the batch service client to perform batch operations
    batch_service_client = create_batch_client()
    # create block blob service to upload config files
    blob_client = BlockBlobService(
        account_name=config["storage_account_name"],
        account_key=config["storage_account_key"],
    )
    # create a container on the storage account for uploading config to
    config_container_name = "input"
    input_azure_config, input_script = prepare_for_task_submission(
        job_name, config_container_name, batch_service_client, blob_client
    )
    # temporary dir to store json files before uploading to storage account
    tmp_json_dir = tempfile.mkdtemp()

    for i, task_dict in enumerate(task_dicts):
        module_class = task_dict["config"]["class_name"]
        config_filename = os.path.join(
            tmp_json_dir, "config_{}_{}.json".format(module_class, i)
        )
        with open(config_filename, "w") as outfile:
            json.dump(task_dict["config"], outfile)
        input_config = upload_file_to_container(
            blob_client, config_container_name, config_filename
        )
        task_id = task_dict["task_id"]
        task_dependencies = task_dict["depends_on"]
        add_task(
            task_id,
            job_name,
            input_script,
            input_config,
            input_azure_config,
            task_dependencies,
            batch_service_client,
        )
    return True


def add_task(
    task_id,
    job_name,
    input_script,
    input_config,
    input_azure_config,
    task_dependencies,
    batch_service_client=None,
):
    """
    add the batch task to the job.

    Parameters
    ==========
    task_id: str, unique ID within this job for the task
    job_name: str, name for the job - usually Sequence name + timestamp
    input_script: ResourceFile corresponding to bash script uploaded to blob storage
    input_config: ResourceFile corresponding to json config for this task uploaded to blob storage
    input_azure_config: ResourceFile corresponding to azure config, uploaded to blob storage
    task_dependencies: list of str, task_ids of any tasks that this one depends on
    batch_service_client: BatchServiceClient
    """
    print("Adding task {} with dependency on {}".format(task_id, task_dependencies))

    if not batch_service_client:
        batch_service_client = create_batch_client()
    command = "/bin/bash {} {} {}".format(
        input_script.file_path, input_config.file_path, input_azure_config.file_path
    )
    print("Adding task {} to job {}".format(task_id, job_name))
    user = batch.models.UserIdentity(
        auto_user=batch.models.AutoUserSpecification(
            elevation_level=batch.models.ElevationLevel.admin,
            scope=batch.models.AutoUserScope.task,
        )
    )
    if len(task_dependencies) > 0:
        task = batch.models.TaskAddParameter(
            id=task_id,
            command_line=command,
            user_identity=user,
            resource_files=[input_script, input_config, input_azure_config],
            depends_on=batch.models.TaskDependencies(task_ids=task_dependencies),
        )
    else:
        task = batch.models.TaskAddParameter(
            id=task_id,
            command_line=command,
            user_identity=user,
            resource_files=[input_script, input_config, input_azure_config],
        )
    batch_service_client.task.add(job_name, task)
    return True


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

    print(
        "Monitoring all tasks for 'Completed' state, timeout in {}...".format(timeout),
        end="",
    )

    num_success, num_failed, num_incomplete = 0, 0, 0
    while datetime.datetime.now() < timeout_expiration:
        print(".", end="")
        sys.stdout.flush()
        num_incomplete, num_success, num_failed = check_tasks_status(
            job_id, batch_service_client
        )
        if num_incomplete == 0:
            print()
            return {
                "Succeeded": num_success,
                "Failed": num_failed,
                "Incomplete": num_incomplete,
            }
        else:
            time.sleep(1)

    print()
    print(
        "WARNING: {} Tasks did not reach 'Completed' state within "
        "timeout period of {}".format(num_incomplete, timeout)
    )
    return {
        "Succeeded": num_success,
        "Failed": num_failed,
        "Incomplete": num_incomplete,
    }


def check_task_failed_dependencies(task, job_id, batch_service_client=None):
    """
    If a task depends on other task(s), and those have failed, the job
    will not be able to run.

    Parameters
    ==========
    task: azure.batch.models.CloudTask, the task we will look at dependencies for
    job_id: str, the unique ID of the Job.
    batch_service_client: BatchServiceClient - will create if not provided.

    Returns
    =======
    True if the job depends on other tasks that have failed (or those
            tasks depend on failed tasks)
    False otherwise
    """
    if not batch_service_client:
        batch_service_client = create_batch_client()
    if task.state != batchmodels.TaskState.active:
        return False
    if not task.depends_on:
        return False
    dependencies = task.depends_on.task_ids
    if len(dependencies) == 0:
        return False
    for dependency in dependencies:
        dep_task = batch_service_client.task.get(job_id, dependency)
        if dep_task.state == batchmodels.TaskState.completed and \
           dep_task.execution_info.exit_code != 0:
            # return True if any of the dependencies failed
            return True
        # use this a recursive function
        dep_dep_failed = check_task_failed_dependencies(dep_task,
                                                        job_id,
                                                        batch_service_client)
        if dep_dep_failed:
            return True

    # got all the way through dependency tree with no failues - return False
    return False


def check_tasks_status(job_id, task_name_prefix="", batch_service_client=None):
    """
    For a given job, query the status of all the tasks.

    Returns
    =======
    task_status: dict, containing the following keys/values:
     num_success: int, successfully completed
     num_failed: int, completed but with non-zero exit code
     num_running: int, currently running
     num_waiting: int, in "active" state
     num_cannot_run: int, in "active" state, but with dependent tasks that failed.
    """
    if not batch_service_client:
        batch_service_client = create_batch_client()
    tasks = batch_service_client.task.list(job_id)
    # Filter by name if provided.  Most tasks will be named after the Module they run.
    if task_name_prefix:
        tasks = [task for task in tasks if task.id.startswith(task_name_prefix)]

    running_tasks = [
        task for task in tasks if (task.state == batchmodels.TaskState.running \
        or task.state == batchmodels.TaskState.preparing)
    ]
    num_running = len(running_tasks)

    incomplete_tasks = [
        task for task in tasks if task.state != batchmodels.TaskState.completed
    ]
    # create a list of 0s or 1s depending on whether tasks have failed dependencies
    cannot_run = [
        int(check_task_failed_dependencies(task, job_id, batch_service_client))
        for task in incomplete_tasks
        if task.state == batchmodels.TaskState.active
    ]
    num_cannot_run = sum(cannot_run)
    num_waiting = len(cannot_run) - num_cannot_run
    # create a list of 0s or 1s depending on whether tasks had status_code==0.
    task_success = [
        int(task.execution_info.exit_code == 0)
        for task in tasks
        if task.state == batchmodels.TaskState.completed
    ]
    num_success = sum(task_success)
    return {
        "num_success": num_success,
        "num_failed": len(task_success) - num_success,
        "num_running": num_running,
        "num_waiting": num_waiting,
        "num_cannot_run": num_cannot_run
    }


def create_pool(pool_id, batch_service_client=None):
    """
    Creates a pool of compute nodes.

    Parameters
    ==========
    pool_id: str, identifier for the pool
    batch_service_client: azure.batch.BatchServiceClient, A Batch service client.
    """
    print("Creating pool [{}]...".format(pool_id))
    if not batch_service_client:
        batch_service_client = create_batch_client()
    new_pool = batch.models.PoolAddParameter(
        id=pool_id,
        virtual_machine_configuration=batchmodels.VirtualMachineConfiguration(
            image_reference=batchmodels.ImageReference(
                publisher="Canonical",
                offer="UbuntuServer",
                sku="18.04-LTS",
                version="latest",
            ),
            node_agent_sku_id="batch.node.ubuntu 18.04",
        ),
        vm_size=config["pool_vm_size"],
        target_low_priority_nodes=config["pool_low_priority_node_count"],
        target_dedicated_nodes=config["pool_dedicated_node_count"],
    )
    batch_service_client.pool.add(new_pool)


def create_job(job_id, pool_id=None, batch_service_client=None):
    """
    Creates a job with the specified ID, associated with the specified pool.

    Parameters
    ==========
    job_id: str, ID for the job - will typically be module or sequence name +timestamp
    pool_id: str, ID for the pool.  If not provided, use the one from azure_config.py
    batch_service_client: BatchServiceClient instance.  Create one if not provided.
    """
    print("Creating job [{}]...".format(job_id))
    if not batch_service_client:
        batch_service_client = create_batch_client()
    if not pool_id:
        pool_id = config["batch_pool_id"]
    try:
        create_pool(pool_id, batch_service_client)
    except:
        print("pool {}  already exists".format(pool_id))
    job = batch.models.JobAddParameter(
        id=job_id,
        pool_info=batch.models.PoolInformation(pool_id=pool_id),
        uses_task_dependencies=True,
    )

    batch_service_client.job.add(job)


def delete_job(job_id, batch_service_client=None):
    """
    Removes a job, and associated tasks.
    """
    if not batch_service_client:
        batch_service_client = create_batch_client()
    batch_service_client.job.delete(job_id)


def delete_pool(pool_id=None, batch_service_client=None):
    """
    Removes a pool of batch nodes
    """
    if not pool_id:
        pool_id = config["batch_pool_id"]
    if not batch_service_client:
        batch_service_client = create_batch_client()
    batch_service_client.pool.delete(pool_id)


def create_batch_client():
    credentials = batch_auth.SharedKeyCredentials(
        config["batch_account_name"], config["batch_account_key"]
    )

    batch_client = batch.BatchServiceClient(
        credentials, batch_url=config["batch_account_url"]
    )
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

    print("Uploading file {} to container [{}]...".format(file_path, container_name))

    block_blob_client.create_blob_from_path(container_name, blob_name, file_path)

    sas_token = block_blob_client.generate_blob_shared_access_signature(
        container_name,
        blob_name,
        permission=azureblob.BlobPermissions.READ,
        expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=4),
    )

    sas_url = block_blob_client.make_blob_url(
        container_name, blob_name, sas_token=sas_token
    )

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
            encoding = "utf-8"
        return output.getvalue().decode(encoding)
    finally:
        output.close()
    raise RuntimeError("could not write data to stream or decode bytes")


def print_task_output(batch_service_client, job_id, encoding=None):
    """Prints the stdout.txt file for each task in the job.

    :param batch_client: The batch client to use.
    :type batch_client: `batchserviceclient.BatchServiceClient`
    :param str job_id: The id of the job with task output files to print.
    """

    print("Printing task output...")

    tasks = batch_service_client.task.list(job_id)

    for task in tasks:

        node_id = batch_service_client.task.get(job_id, task.id).node_info.node_id
        print("Task: {}".format(task.id))
        print("Node: {}".format(node_id))

        stream = batch_service_client.file.get_from_task(
            job_id, task.id, config._STANDARD_OUT_FILE_NAME
        )

        file_text = _read_stream_as_string(stream, encoding)
        print("Standard output:")
        print(file_text)

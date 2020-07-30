# Running `pyveg` on the cloud - Microsoft Azure

It is possible to make use of Azure cloud infrastructur in two ways when running pyveg:
* Using Azure blob storage to store downloaded images and results of the image processing.
* Using Azure batch to parallelize the running of the image processing.  This can vastly speed up the running time of your job.

In order to do these, you will need the following:
* An Azure account and an active subscription.
* An Azure "Blob Storage Account" - follow instructions on https://docs.microsoft.com/en-us/azure/storage/blobs/storage-blob-create-account-block-blob?tabs=azure-portal
* An Azure "Batch Account" - see https://docs.microsoft.com/en-us/azure/batch/batch-technical-overview for a description.  When setting up the batch account, it will ask you to link it to a Storage Account, so use the one above (and ensure that you select the same "Region" for both.
* You will probably need to increase the "quota" on your Batch Account - I believe that by default the quota for different types of VM are all set to zero.  There are instructions on how to do this at: https://docs.microsoft.com/en-us/azure/batch/batch-quota-limit - for our workflow we are using 100 dedicated cores of A1v2 VMs.

## Setting up `azure_config.py`

In the `pyveg/` directory there is a file `azure_config_template.py`.  Copy this to `azure_config.py` (in the same directory, and then start filling in the various fields.  The necessary info can be found in the Azure portal [https://portal.azure.com/#home] - perhaps the easiest way is to navigate via the "Subscriptions" icon at the top of the portal, then find the "Resources" (i.e. the Storage Account and the Batch Account).
* For the Storage Account, look on the left sidebar under "Settings" for the "Access keys" - then copy/paste one of the keys into the relevant field in `azure_config.py`.
* For the Batch Account, similarly there is a "Keys" icon under "Settings" which will lead to the relevant info.
For the "batch_pool_id" you can put any name you like - if there will be multiple people using the same batch and storage accounts, you might want to use your initials or something to identify you (and in this case, you should be careful that the sum of everyone's "node_count"s don't exceed the quota for the batch account.

Once you have populated the fields in `azure_config.py`, then do
```
pip install .
```
from the main `monitoring-ecosystem-resilience` directory.

## config settings to use Azure storage

We recommend that you create a new config file in the `pyveg/configs/` directory for every location/collection that you run.  You can use [pyveg/configs/test_Sentinel2_azure.py] as an example (you will want to increase the "date_range", and "n_sub_images" for the `NetworkCentralityCalculator` before running production though).

The key setting to tell the job to use Azure blob storage is:
```
output_location_type = "azure"
```

### config settings to use Azure batch

Note that using Azure storage as detailed above is a prerequisite for using Azure batch.
Again there is an example template [pyveg/configs/test_Sentinel2_batch.py] that you can copy and modify with your own coordinates, date range, collection etc.

Here, the important settings that enable the time-consuming parts of the pipeline to use Azure batch are in the `special_config` dictionary at the bottom of the file:
```
special_config = {
    "VegetationImageProcessor": {"run_mode": "batch"},
    "NetworkCentralityCalculator": {
        "n_sub_images": 10,
        "n_threads": 1,
        "run_mode": "batch",
    },
    "NDVICalculator": {"run_mode": "batch"},
}
```
This is setting "run_mode" to "batch" for these three Modules.  Note also that for `NetworkCentralityCalculator` we are setting "n_threads" to 1.  This is advisable if using the A1v2 nodes in the batch pool, as they only have a single vCPU per node.   If you instead choose more powerful VMs (which are also more expensive!) you can increase this accordingly.


### Checking the status of your job on Azure batch

If one or more Modules have "run_mode" set to "batch", the console output should give a running status of how many "tasks" are still running.  You can also check on the Azure portal [https://portal.azure.com/#home] - find the "Resource" that is your batch account, then on the left sidebar, navigate down to "Jobs", and click on the current job to see the status of all its "Tasks".  (A "Job" corresponds to an instance of a Module running over all the dates in the date range.  A "Task" is a single date within this date range.)

### Downloading data from Azure storage when it is ready

Azure blob storage is structured with "Containers" containing "Blobs", where the blobs themselves can have a directory-like structure.
A single pyveg pipeline job will produce a single Container, which will have the name
`<output_location>_<date_stamp>` where `output_location` was defined in your config file, and the `date_stamp` is when the job was launched.
You can find the container name in the logfile for your job, or via the Azure portal [https://portal.azure.com/#home] - if you find the "Resource" that is your Storage Account, then click on "Containers".

A script exists that can download the RGB images and the `results_summary.json` to a single local zipfile: `pyveg/scripts/download_from_azure.py`.   To run this, do
```
pyveg_azure_download --container <container_name> --output_zipfile <name_of_zipfile_to_write_to>
```


## What is going on "under the hood" when running on Azure batch?

(This section is only necessary if you are interested in knowing more about how this works - if you just want to run the jobs, the instructions above should suffice.)

When you run the command
```
pyveg_run_pipeline --config_file <some_config_file>
```
the script `pyveg/scripts/run_pyveg_pipeline` will read in the specified config file and use it to set parameters of a "Pipeline" that is composed of "Sequences", which are in turn composed of "Modules".

The batch functionality is implemented at the Module level.  If a Module has "run_mode" set to "batch", it will:
* Get a dictionary of batch Tasks on which its Tasks depends (e.g. the NetworkCentralityCalculator needs the ImageProcessor to have finished for a given date before it can run that date's Task).
* Creates a new batch "Job" with a name composed of the Module name and the current time.
* Create a dictionary of batch Tasks for the Job, dividing up the date range amongst the Tasks, and storing the configuration of the Module for each entry.
* Upload the `azure_config.py` and `pyveg/scripts/batch_commands.sh` to blob storage.

For each Task, the process is then:
* Write the configuration to a JSON file and upload to blob storage.
* Submit the batch Task, which will run `batch_commands.sh` on the batch node.

### What does `batch_commands.sh` do ?

The execution of a single Task on a batch node (which is an Ubuntu-16.04 VM) is governed by this shell script `batch_commands.sh`.  The basic flow is:
* Install some packages, including miniconda, and create and activate a Python 3.7 conda environment.
* Clone the `monitoring-ecosystem-resilience` repo, change to the `develop` branch, and do ```pip install .``` to install it.
* Run the command ```pyveg_run_module --config_file <json_config_for_module>``` where the json config file is the set of parameters needed to configure this Module, based on the dictionary created in `processor_modules.creta_task_dict`.

### What happens when all tasks are submitted?

The function `processor_modules.run_batch` will submit all the tasks and then return straightaway, rather than waiting for the tasks to finish.  This means that other Modules or Sequences (e.g. WeatherSequence) that do not depend on the results of this Module can still be executed.
However, usually the final Sequence in a Pipeline will be a "combiner" Sequence, that has a `depends_on` attribute.
If a Sequence listed in `depends_on` has one-or-more Modules with "run_mode" set to "batch", the logic in the `run` method of the `Sequence` class in `pyveg/src/pyveg_pipeline.py` will loop through all the Modules in that Sequence, and call `check_if_finished()` on all of them.  This in turn will query the Batch Job to see the status of all the Tasks.
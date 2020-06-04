"""
Functions for submitting batch jobs.  Currently only support Azure Batch.
"""

import json

from pyveg.src.azure_utils import *

def submit_tasks(list_of_configs):
    for i,config in enumerate(list_of_configs):
        with open(f"/tmp/config_{i}.json","w") as outfile:
            json.dump(config, outfile)

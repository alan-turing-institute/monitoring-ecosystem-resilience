
# copy this file to azure_config.py  and fill in the fields below with values
# that you should be able to obtain from the Azure portal https://portal.azure.com
# when you have an Azure batch account, and linked Azure storage account.

config = {
    "batch_account_name": < INSERT_BATCH_ACCOUNT_NAME_HERE >,
    "batch_account_key": < INSERT_BATCH_ACCOUNT_KEY_HERE >,
    "batch_account_url": < INSERT_BATCH_ACCOUNT_URL_HERE >,
    "storage_account_name":  < INSERT_STORAGE_ACCOUNT_NAME_HERE >,
    "storage_account_key": < INSERT_STORAGE_ACCOUNT_ACCESS_KEY_HERE >
    "batch_pool_id": < UNIQUE_NAME_FOR_POOL >,
    "pool_low_priority_node_count": < N_LOW_PRIORITY_NODES >,
    "pool_dedicated_node_count": < N_DEDICATED_NODES >,
    "pool_vm_size":  < VM_SIZE, e.g. 'STANDARD_A1_v2' >,
    "stdout_file_name": 'stdout.txt'
}

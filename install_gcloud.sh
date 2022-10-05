
# Download and install gcloud CLI
if [ $(uname -m) = "arm64" ] ; then
    # alt_url = https://storage.cloud.google.com/cloud-sdk-release/google-cloud-cli-376.0.0-linux-arm.tar.gz
    url=https://storage.googleapis.com/cloud-sdk-release/google-cloud-cli-405.0.0-linux-arm.tar.gz
    # chksum =
else
    # alt_url = https://storage.cloud.google.com/cloud-sdk-release/google-cloud-cli-374.0.0-linux-x86_64.tar.gz
    url=https://storage.googleapis.com/cloud-sdk-release/google-cloud-cli-405.0.0-linux-x86_64.tar.gz
    # chksum =
fi

curl $url > tmp_gcloud_install.tar.gz
tar -xzf tmp_gcloud_install.tar.gz -C .

./google-cloud-sdk/install.sh --quiet
# /bin/bash /gee_pipeline/google-cloud-sdk/path.bash.inc




# # For the latest version
# curl https://sdk.cloud.google.com > install.sh
# bash install.sh --disable-prompts

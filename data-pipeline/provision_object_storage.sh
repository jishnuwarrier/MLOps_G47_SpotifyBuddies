#!/bin/bash

# Exit immediately if any command fails
set -e

# Set OpenStack authentication environment variables
export OS_AUTH_URL="https://chi.tacc.chameleoncloud.org:5000/v3"
export OS_PROJECT_NAME="CHI-251409"
export OS_REGION_NAME="CHI@TACC"

# Define container name
container_name="object-persist-project47"

# Create the object storage container
echo "Creating container: $container_name"
openstack container create $container_names
echo "Container '$container_name' created successfully."
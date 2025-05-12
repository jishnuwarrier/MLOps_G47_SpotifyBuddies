#!/bin/bash

# Exit immediately if any command fails
set -e

# Set environment variables for OpenStack authentication
export OS_AUTH_URL="https://kvm.tacc.chameleoncloud.org:5000/v3"
export OS_PROJECT_NAME="CHI-251409"
export OS_REGION_NAME="KVM@TACC"

# Define resource names
block_vol_name="block-persist-project47"
instance_name="node1_cloud_project47"

# Create a new volume
echo "Creating volume: $block_vol_name"
openstack volume create --size 1 $block_vol_name
echo "Created block volume: $block_vol_name"

# Confirm volume creation
echo "Checking volume list for matching instance name..."
openstack volume list | grep $instance_name || echo "No volumes matching $instance_name found in list."

# Attach the volume to the instance
echo "Attaching volume $block_vol_name to instance $instance_name"
openstack server add volume $instance_name $block_vol_name
echo "Volume attached successfully."

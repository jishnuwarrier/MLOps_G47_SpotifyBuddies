#!/bin/bash

# Exit if any command fails
set -e

# Define mount point and remote path
mount_point="/mnt/object"
remote_name="chi_tacc"
container_name="object-persist-project47"

# Create the mount point
echo "Creating mount point at $mount_point..."
sudo mkdir -p "$mount_point"

# Change ownership to user 'cc'
echo "Changing ownership and group of $mount_point to 'cc'..."
sudo chown -R cc "$mount_point"
sudo chgrp -R cc "$mount_point"

# Mount the object storage container using rclone
echo "Mounting $remote_name:$container_name to $mount_point (read-only, allow-other)..."
rclone mount "$remote_name:$container_name" "$mount_point" --read-only --allow-other --daemon

echo "Mount complete."

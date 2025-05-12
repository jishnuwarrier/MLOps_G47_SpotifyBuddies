#!/bin/bash

# Exit if any command fails
set -e

# Install rclone
echo "Installing rclone..."
curl https://rclone.org/install.sh | sudo bash

# Enable 'user_allow_other' in /etc/fuse.conf
echo "Enabling 'user_allow_other' in /etc/fuse.conf..."
sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf

# Create rclone config directory
echo "Creating ~/.config/rclone directory..."
mkdir -p ~/.config/rclone

echo "rclone installation and setup complete."
echo "Please edit the rclone.conf file with Application ID and Secret"
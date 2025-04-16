# Data Pipeline Readme

## Instructions to use object store from an instance:
1. SSH into the instance
```
ssh -i ~/.ssh/id_rsa_chameleon cc@A.B.C.D
```
2. Install rclone
```
curl https://rclone.org/install.sh | sudo bash
```
3. Modify FUSE (Filesystem in USErspace) config file
```
sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf
```
4. Create a config file and copy the contents into the config file.
Substitute your UserId(Chameleon->Identity->Users), AppUserId, AppSecret(contact me)
```
mkdir -p ~/.config/rclone
nano  ~/.config/rclone/rclone.conf
```
```
[chi_tacc]
type = swift
user_id = YOUR_USER_ID
application_credential_id = APP_CRED_ID
application_credential_secret = APP_CRED_SECRET
auth = https://chi.tacc.chameleoncloud.org:5000/v3
region = CHI@TACC
```
5. Run this command and check if your object store is visible
```
rclone lsd chi_tacc:
```
6. Mount the object store into local file system
```
sudo mkdir -p /mnt/object
sudo chown -R cc /mnt/object
sudo chgrp -R cc /mnt/object
rclone mount chi_tacc:object-persist-project47 /mnt/object --read-only --allow-other --daemon
```
7. Run this command to check if the data directories are available
```
ls /mnt/object
```

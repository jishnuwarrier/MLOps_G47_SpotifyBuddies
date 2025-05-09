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

## Instructions to run docker container with MLFlow, Prometheus and Grafana

1. First attach the block storage to your instance(Do this on Chameleon. I'm not sure of any other way) - KVM@TACC->Volumes

2. Run this command to verify

```
df -h
```

3. You can run your docker compose command now

```
HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4 ) docker compose -f ~/data-persist-chi/docker/docker-compose-block.yaml up -d
```

**Things to note:**

1. Make sure the security groups of the instance allow TCP to all the ports(namely 22(SSH), 8888(Jupyter), 8000(MLFlow), 9000(MinIO API), 9001(MinIO Web UI), 9090(Prometheus), 3000(Grafana))

2. Make sure Prometheus and Grafana have permissions to edit their mounted files(TODO: Find a way to automate this)

```
sudo chmod -R 777 /mnt/block/prometheus_data

sudo mkdir -p /mnt/block/grafana_data
sudo chown -R 472:472 /mnt/block/grafana_data
```

**Important commands**
```
# copy from instance to object store
rclone copy /data/Food-11 chi_tacc:object-persist-project47 \
        --progress \
        --transfers=32 \
        --checkers=16 \
        --multi-thread-streams=4 \
        --fast-list

# top 20 dirs taking up space
du -h --max-depth=1 ~ | sort -hr | head -n 20
```

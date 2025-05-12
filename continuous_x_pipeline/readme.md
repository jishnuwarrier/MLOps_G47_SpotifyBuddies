## Part 4: Continuous X

Below is the folder structure for the continuous_x_pipeline code

```text
├── continuous_x_pipeline
│   ├── ansible
│   │   ├── argocd
│   │   ├── k8s
│   │   │   ├── inventory
│   │   │   │   └── mycluster
│   │   │   │       └── group_vars
│   │   │   └── kubespray
│   │   ├── post_k8s
│   │   └── pre_k8s
│   ├── k8s
│   │   ├── canary
│   │   │   └── templates
│   │   ├── monitoring
│   │   │   ├── airflow
│   │   │   ├── grafana
│   │   │   ├── prometheus
│   │   │   └── redis
│   │   ├── platform
│   │   │   ├── fastapi
│   │   │   └── templates
│   │   ├── production
│   │   │   └── templates
│   │   └── staging
│   │       └── templates
│   ├── terraform
│   └── workflows
```


--------

## Infrastructure-as-Code (IaC) - Terraform

We manage all of our Chameleon infrastructure with Terraform. The entire configuration lives in the ```/continuous_x_pipeline/terraform``` folder of the repository.

In [versions.tf](https://github.com/AguLeon/MLOps_G47_SpotifyBuddies/tree/main/continuous_x_pipeline/terraform/versions.tf/) -
We lock our Terraform toolchain to version 0.14 or newer, and pin the OpenStack provider plugin to the 1.51.x series to guarantee reproducability across machines.

In [provider.tf](https://github.com/AguLeon/MLOps_G47_SpotifyBuddies/tree/main/continuous_x_pipeline/terraform/provider.tf/) -
We declare a single OpenStack provider named "openstack", matching the cloud entry in your clouds.yaml. Terraform will read your Chameleon credentials from that file and use them for all API calls.

### Input Variables
In [variables.tf](https://github.com/AguLeon/MLOps_G47_SpotifyBuddies/tree/main/continuous_x_pipeline/terraform/variables.tf/) - 
We expose four key variables:

suffix (string, required) – our project’s unique suffix (project47 in our case), appended to every resource name to avoid collisions.

key (string, default id_rsa_chameleon) – the SSH keypair name to install on each VM.

nodes (map of strings) – a map of logical node names ("node1", "node2", "node3") to fixed private IP addresses on our 192.168.1.0/24 subnet and they are named node1-mlops-project47, node2-mlops-project47, node3-mlops-project47.

data_volume_size (default 50) – the size (in GB) of a shared block-storage volume we attach to node1 for logs, model artifacts, or other persistent files.

### Data Sources
In [data.tf](https://github.com/AguLeon/MLOps_G47_SpotifyBuddies/tree/main/continuous_x_pipeline/terraform/data.tf/) - 
We import existing OpenStack objects we do not manage in this repo:

The public network & subnet (sharednet2), so we can place VM ports there.

Seven security groups (allow-ssh, allow-9001, allow-8000, allow-8080, allow-8081, allow-http-80, and allow-9090) to lock down SSH and our various service ports.

### Core Resources
[main.tf](https://github.com/AguLeon/MLOps_G47_SpotifyBuddies/tree/main/continuous_x_pipeline/terraform/main.tf/):

Private Network & Subnet
We create a new isolated network and a 192.168.1.0/24 subnet with port security disabled. This hosts our control-plane communication between nodes.

Network Ports
For each node in var.nodes, we use up two ports:

-A private port on our new subnet, taking the IP from the map.

-A public port on sharednet2, bound to exactly the security groups we imported.

Compute Instances
We launch one Ubuntu VM per entry in var.nodes (flavor m1.medium). Each VM gets both its private and public port attached, plus a little user_data script to populate /etc/hosts and install your SSH keys via the Chameleon cloud-init hook.

Floating IP
We allocate a single floating IP and bind it to node1’s public port—this is the IP you’ll use to reach your cluster head node for Ansible, ArgoCD, or your FastAPI endpoints.

Persistent Block Storage
We create a volume of size var.data_volume_size (50 GB) and attach it to node1 at /dev/vdb, giving you a durable filesystem for model artifacts, data, or logs.

### Outputs
[outputs.tf](https://github.com/AguLeon/MLOps_G47_SpotifyBuddies/tree/main/continuous_x_pipeline/terraform/outputs.tf/) - 
We export only the floating IP’s address as floating_ip_out. This makes it easy for subsequent Ansible playbooks or for your README instructions to reference exactly terraform output floating_ip_out when building your inventory or service URLs.


The ```ansible/inventory.yml``` file has the IP’s of the 3 nodes provisioned using Terraform.


## Instructions to run terraform:

1. Open a Jupyter server on chameleon.
2. Upload your clouds.yaml file in the server.
3. Clone the github repo - 

`git clone --recurse-submodules https://github.com/AguLeon/MLOps_G47_SpotifyBuddies.git`

1. Go to KVM@TACC in chameleon > Identity > Application Credential > Create an application credential on Chameleon and download the clouds.yaml file.
2. Upload your clouds.yaml file to jupyter. (It should be stored in /work/clouds.yaml)
3. Open terminal and run the following commands:
    1. `cd /work/MLOps_G47_SpotifyBuddies/continuous_x_pipeline/terraform` #Navigating to the directory
    2. `chmod +x terraform_script.sh` #To make the script executable 
    3. `./terraform_script.sh` #Run the script

----

## Configuration-as-Code (CaC) & Kubernetes Bootstrapping

All cluster provisioning and application registration is fully automated with Ansible, under the continuous_x_pipeline/ansible directory. 


1. **OS Preparation**
    - **Playbook:** [pre_k8s_configure](https://github.com/AguLeon/MLOps_G47_SpotifyBuddies/tree/main/continuous_x_pipeline/ansible/pre_k8s/pre_k8s_configure.yml)
        
        Runs on all nodes to disable and mask firewalld, install containerd (or Docker), and configure /etc/docker/daemon.json with our insecure-registry settings. This ensures our cluster nodes will pull images from the local registry without manual steps.
        
2. **Kubernetes Installation (done using Kubespray)**
    - **Inventory:**
        - [hosts.yaml](https://github.com/AguLeon/MLOps_G47_SpotifyBuddies/tree/main/continuous_x_pipeline/ansible/k8s/inventory/mycluster/hosts.yaml)  - lists each node’s private IP and SSH user.
        - [all.yaml](https://github.com/AguLeon/MLOps_G47_SpotifyBuddies/tree/main/continuous_x_pipeline/ansible/k8s/inventory/mycluster/group_vars/all.yaml) -  overrides defaults to enable the dashboard, Helm, and disable features we don’t need (VIP, Node Feature Discovery).
    - **Run:**  ```ansible-playbook -i ansible/k8s/inventory/mycluster/ansible.cfg ansible/k8s/kubespray/cluster.yml```
        
        This playbook deploys a self-managed Kubernetes cluster across the three VMs, handling kube-adm, networking, and control-plane HA out-of-the-box.
    
    [nodes-on-k8s](./continuous_x_pipeline/images/nodes_on_kubernetes.png)
        
3. **Post-Install Configuration**
    - **Playbook:** [post_k8s_configure](https://github.com/AguLeon/MLOps_G47_SpotifyBuddies/tree/main/continuous_x_pipeline/ansible/post_k8s/post_k8s_configure.yml)
        
        On node1, copies the cluster’s `admin.conf` into each user’s `~/.kube/config`, sets up hostname mappings, and applies sysctl tweaks (e.g. disabling IPv6). This gives you immediate `kubectl` access via the head node.

----

## Applications:

The applications we are using are:<br>
Graphana - dashboard and visualization tool for monitoring metrics and logs.

Prometheus - Real-time monitoring by scraping FastAPI server in 15 second intervals

Minio - Storing MLflow artifacts and MLflow tracking using a bucket

Airflow - To schedule inference as well as simulating the users interacting with recommendation.

Redis - Stores the model 

Postgres - To store the feedback, and used by other applications.

MLFlow - Tracking experiments, storing model metrics and model registration.

FastAPI - Model serving


----

## List of namespaces:

I've created a total of 5 namespaces for our application.

### [`spotifybuddies-platform`](https://github.com/AguLeon/MLOps_G47_SpotifyBuddies/tree/main/continuous_x_pipeline/k8s/platform)  
**Purpose:** Core platform services that support experiment tracking, model registry, and artifact storage.  
**Contains:**  
- PostgreSQL
- MinIO
- MLflow Server
- FastAPI


---

### [`spotifybuddies-monitoring`](https://github.com/AguLeon/MLOps_G47_SpotifyBuddies/tree/main/continuous_x_pipeline/k8s/monitoring)
**Purpose:** Cluster-wide monitoring and visualization.  

**Contains:**  
- Prometheus   
- Grafana 
- Airflow  
- Redis 

---

### [`spotifybuddies-staging`](https://github.com/AguLeon/MLOps_G47_SpotifyBuddies/tree/main/continuous_x_pipeline/k8s/staging)
**Purpose:** Early testing environment for our FastAPI application before canary rollout.  

**Contains:**  spotifybuddies-app Deployment 

---

### [`spotifybuddies-canary`](https://github.com/AguLeon/MLOps_G47_SpotifyBuddies/tree/main/continuous_x_pipeline/k8s/canary)  
**Purpose:** Gradual roll-out environment to validate new image tags under real traffic.  

**Contains:** spotifybuddies-app Deployment  

---

### [`spotifybuddies-production`](https://github.com/AguLeon/MLOps_G47_SpotifyBuddies/tree/main/continuous_x_pipeline/k8s/production)

**Purpose:** Live environment serving real user traffic.  

**Contains:** spotifybuddies-app Deployment   

----

The code for running the ansible playbooks are in 
[Ansible-k8s](https://github.com/AguLeon/MLOps_G47_SpotifyBuddies/tree/main/continuous_x_pipeline/ansible.ipynb)

[Ansible-platform & build](https://github.com/AguLeon/MLOps_G47_SpotifyBuddies/tree/main/continuous_x_pipeline/ansible_build.ipynb)

The command ```ansible-playbook -i inventory.yml argocd/argocd_add_platform.yml``` was run to add platforms, after which all the applications could be accessed.
For example, MLFlow can be accessed using - [Link](http://129.114.25.50:8000/)

## Error Description

After completion of the kubernetes deployment using Ansible, I ran into running the ansible-playbook -i inventory.yml argocd/workflow_build_init.yml



This triggered an Argo Workflow with two steps: a git-clone to fetch the repository and a Kaniko container build.

While the Git clone step successfully cloned the repository into /mnt/workspace (as confirmed by the following output):

![Error](./continuous_x_pipeline/images/dockerfile_error.png)

the subsequent Kaniko step failed with the following error: error resolving dockerfile path: please provide a valid path to a Dockerfile within the build context with --dockerfile

As seen in the output of the ls command the Dockerfile is in the root directory itself. 
The error was during the execution of the file [build-initial.yaml](https://github.com/AguLeon/MLOps_G47_SpotifyBuddies/tree/main/continuous_x_pipeline/workflows/build-initial.yaml) in line 56 despite trying with both absolute, and relative path.

```- --dockerfile=/mnt/workspace/Dockerfile```

```- --dockerfile=Dockerfile```

---

While the Kubernetes infrastructure and ArgoCD integration were successfully set up, all three application environments—staging, canary, and production—failed during runtime due to missing container images. This is the reason that although the ansible notebook ran for each of the 3 environments, the deployment wasn't successful. 

![kubectl logs](./continuous_x_pipeline/images/kubectl_logs.png)

As seen in the ArgoCD UI screenshot, the spotifybuddies-staging, spotifybuddies-canary, and spotifybuddies-production applications are in a "Degraded" state. Correspondingly, the kubectl get pods output confirms that each environment's FastAPI deployment pod is stuck in the ImagePullBackOff state, which indicates Kubernetes is continuously attempting, and failing, to pull the required container image.



![argocd_degraded](./continuous_x_pipeline/images/argocd_degraded.png)



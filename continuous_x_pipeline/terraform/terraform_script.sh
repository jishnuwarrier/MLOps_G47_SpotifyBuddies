set -euo pipefail

cd /work
pwd

REPO_DIR="/work/MLOps_G47_SpotifyBuddies"
if [ ! -d "$REPO_DIR" ]; then
  git clone --recurse-submodules \
    https://github.com/AguLeon/MLOps_G47_SpotifyBuddies.git \
    "$REPO_DIR"
else
  echo "Repo already cloned at $REPO_DIR"
fi


TERRAFORM_VERSION="1.10.5"
LOCAL_BIN="/work/.local/bin"
mkdir -p "$LOCAL_BIN"

if ! command -v terraform >/dev/null 2>&1 \
   || ! terraform version | grep -q "$TERRAFORM_VERSION"; then
  echo "Installing Terraform $TERRAFORM_VERSION…"
  wget "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_amd64.zip"
  unzip -o -q "terraform_${TERRAFORM_VERSION}_linux_amd64.zip"
  mv terraform "$LOCAL_BIN/"
  rm "terraform_${TERRAFORM_VERSION}_linux_amd64.zip"
else
  echo "Terraform $TERRAFORM_VERSION already installed"
fi

export PATH="$LOCAL_BIN:$PATH"


# Copy your clouds.yaml into the Terraform folder
SRC_CLOUDS_YAML="/work/clouds.yaml"
DST_DIR="$REPO_DIR/continuous_x_pipeline/terraform"
mkdir -p "$DST_DIR"
cp "$SRC_CLOUDS_YAML" "$DST_DIR/clouds.yaml"

# Export Python user base so any Python installs also go to /work/.local
export PYTHONUSERBASE="/work/.local"

# Change into the Terraform directory
cd "$DST_DIR"

# Unset any OS_* env vars that might interfere
unset $(set | grep -o "^OS_[A-Za-z0-9_]*" || true)

# Inspect clouds.yaml
cat clouds.yaml


terraform init

# Set any TF_VARs your configs need
export TF_VAR_suffix="project47"
export TF_VAR_key="id_rsa_chameleon"

terraform validate
terraform plan
terraform apply -auto-approve

echo " Terraform apply complete!"

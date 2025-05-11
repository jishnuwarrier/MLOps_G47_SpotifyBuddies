# Verify that prometheus user has permission in prometheus block storage directory
sudo chown -R 655434:65534 /mnt/block/prometheus_data
# Verify that grafana user has permission in grafana block storage directory
sudo chown -R 472:472 /mnt/block/grafana_data

source ./run_docker_compose.sh

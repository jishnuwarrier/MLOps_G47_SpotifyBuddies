#! /bin/bash/

# Default value for the flag
run="prod"

# Usage Function
usage() {
    echo "Usage: $0 --run <prod|exp|all>"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
    --run)
        if [[ -z "$2" ]]; then
            echo "Error: --run requires a value (prod, exp, or all)"
            usage
        fi
        run="$2"
        shift 2
        ;;
    *)
        echo "Unknown option: $1"
        usage
        ;;
    esac
done

# Check if the value of run is valid
if [[ "$run" != "prod" && "$run" != "exp" && "$run" != "all" ]]; then
    echo "Error: Invalid value for --run. Valid options are: prod, exp, or all."
    usage
    run="prod"
fi

# Process based on the value of to_run
echo "Running with to_run: $run"

case "$run" in
prod)
    echo "Running Production Only Services"
    # Run only production services (without jupyter and mlflow)
    docker compose -f ./docker-compose-base.yaml -f ./docker-compose-airflow.yml -f ./docker-compose-model-server.yml up
    ;;
exp)
    echo "Running experiment Only Services"
    # Run only experiment services (only jupyter and mlflow with base)
    docker compose -f ./docker-compose-base.yaml -f ./docker-compose-experiment.yml up
    ;;
all)
    echo "Running All Services"
    # Run all
    docker compose -f ./docker-compose-base.yaml -f ./docker-compose-airflow.yml -f ./docker-compose-model-server.yml -f ./docker-compose-experiment.yml up
    ;;
esac

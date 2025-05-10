# ==== Variables ====
VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
REQUIREMENTS = requirements.txt
TRAIN_SCRIPT = spotifyBuddies_train_mlflow_v2.py
REPO_URL = https://github.com/AguLeon/MLOps_G47_SpotifyBuddies
REPO_DIR = MLOps_G47_SpotifyBuddies

# ==== Commands ====

clone:
	@if [ -d "$(REPO_DIR)" ]; then \
		echo "Repository already cloned."; \
	else \
		echo "Cloning repository..."; \
		git clone $(REPO_URL); \
	fi

setup:
	python3 -m venv $(VENV_DIR)
	. $(VENV_DIR)/bin/activate && $(PIP) install --upgrade pip && $(PIP) install -r $(REQUIREMENTS)

train:
	. $(VENV_DIR)/bin/activate && $(PYTHON) $(TRAIN_SCRIPT)


lint:
	. $(VENV_DIR)/bin/activate && flake8 .

clean:
	rm -rf __pycache__ *.pyc $(VENV_DIR) mlruns last_run_id.txt training_output


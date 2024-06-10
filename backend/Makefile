#################### LOCAL ###################
install_dependencies:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .


start_env:
	pyenv virtualenv 3.10.6 python-env
	pyenv local python-env


ENV_FILE := .env
.DEFAULT_GOAL := default

#################### PACKAGE ACTIONS ###################
ML_DIR=~/.lewagon/pokedex

reinstall_package:
	@pip uninstall -y pokedex || :
	@pip install -e .

additional_packages:
	sudo apt-get update && apt-get install libgl1
# rm -rf ${ML_DIR}
reset_local_files:
	rm -rf ${ML_DIR}
	mkdir -p ~/.lewagon/pokedex/data/
	mkdir ~/.lewagon/pokedex/data/raw
	mkdir ~/.lewagon/pokedex/data/processed
	mkdir ~/.lewagon/pokedex/training_outputs
	mkdir ~/.lewagon/pokedex/training_outputs/metrics
	mkdir ~/.lewagon/pokedex/training_outputs/models
	mkdir ~/.lewagon/pokedex/training_outputs/params
	mkdir ~/.lewagon/pokedex/gan
	mkdir ~/.lewagon/pokedex/gan/output_models
	mkdir ~/.lewagon/pokedex/gan/output_images

set_up_env:
	cp .env.sample .env
	echo "Please update env variables in .env"
	echo "Once you are done : make run_15 or make run_150 "

start: install_dependencies additional_packages reset_local_files set_up_env
	direnv allow



run:
	direnv reload
	direnv exec . python pokedex/interface/main.py

reset_classification_env:
	@echo "Modifying environment variable in $(ENV_FILE)..."
	@if grep -q "CLASSIFICATION_TYPE=" $(ENV_FILE); then \
		sed -i 's/CLASSIFICATION_TYPE=.*/CLASSIFICATION_TYPE=$(new_value)/' $(ENV_FILE); \
	else \
		echo "CLASSIFICATION_TYPE=$(new_value)" >> $(ENV_FILE); \
	fi
	@echo "Modification complete."

run_15:
	$(MAKE) reset_classification_env new_value=15
	$(MAKE) run

run_150:
	$(MAKE) reset_classification_env new_value=150
	$(MAKE) run




run_preprocess:
	python -c 'from pokedex.interface.main import preprocess; preprocess()'

run_train:
	python -c 'from pokedex.interface.main import train; train()'

run_pred_15:
	python -c 'from pokedex.interface.main import pred; pred('15')'

run_pred_150:
	python -c 'from pokedex.interface.main import pred; pred('150')'


run_evaluate:
	python -c 'from pokedex.interface.main import evaluate; evaluate()'

run_all: run_preprocess run_train run_pred run_evaluate

run_workflow:
	PREFECT__LOGGING__LEVEL=${PREFECT_LOG_LEVEL} python -m pokedex.interface.workflow

run_api:
	uvicorn pokedex.api.fast:app --reload

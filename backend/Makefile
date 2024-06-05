#################### LOCAL ###################
install_dependencies:
	pip install --upgrade pip
	pip install -r requirements.txt

start_env:
	pyenv virtualenv $(PYTHON_VERSION) $(ENV_NAME)
	pyenv local $(ENV_NAME)



.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
ML_DIR=~/.lewagon/pokedex

reinstall_package:
	@pip uninstall -y pokedex || :
	@pip install -e .

reset_local_files:
	rm -rf ${ML_DIR}
	mkdir -p ~/.lewagon/pokedex/data/
	mkdir ~/.lewagon/pokedex/data/raw
	mkdir ~/.lewagon/pokedex/data/processed
	mkdir ~/.lewagon/pokedex/training_outputs
	mkdir ~/.lewagon/pokedex/training_outputs/metrics
	mkdir ~/.lewagon/pokedex/training_outputs/models
	mkdir ~/.lewagon/pokedex/training_outputs/params

set_up_env:
	cp .env.sample .env
	echo "Please update env variables in .env"
	echo "Once you are done : make run_15 or make run_150 "

start: reinstall_package reset_local_files set_up_env
	direnv allow



run:
	direnv reload
	direnv exec . python pokedex/interface/main.py

run_15:
	echo "export CLASSIFICATION_TYPE=15" > .env;
	make run

run_150:
	echo "export CLASSIFICATION_TYPE=150" > .env;
	make run




run_preprocess:
	python -c 'from pokedex.interface.main import preprocess; preprocess()'

run_train:
	python -c 'from pokedex.interface.main import train; train()'

run_pred:
	python -c 'from pokedex.interface.main import pred; pred()'

run_evaluate:
	python -c 'from pokedex.interface.main import evaluate; evaluate()'

run_all: run_preprocess run_train run_pred run_evaluate

run_workflow:
	PREFECT__LOGGING__LEVEL=${PREFECT_LOG_LEVEL} python -m pokedex.interface.workflow

## LOCAL ##
install_dependencies:
	pip install --upgrade pip
	pip install -r requirements.txt

start_env:
	pyenv virtualenv $(PYTHON_VERSION) $(ENV_NAME)
	pyenv local $(ENV_NAME)

set_up_env:
	cp .env.sample .env
	direnv allow
# direnv reload



.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y pokedex || :
	@pip install -e .

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

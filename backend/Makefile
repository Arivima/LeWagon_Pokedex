PYTHON_VERSION := 3.10.6
ENV_NAME := pokemon-env
ENV_FILE := .env
.DEFAULT_GOAL := default
ML_DIR=~/.lewagon/pokedex

#################### LOCAL ###################

#################### SET-UP ###################
connect:
	@printf $(BOLDCYAN)"Makefile: Connecting team git repositories\n"$(RESET)
	./scripts/git_connect_repo.sh

install_dependencies:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

start_env:
	pyenv virtualenv $(PYTHON_VERSION) $(ENV_NAME)
	pyenv local $(ENV_NAME)

reinstall_package:
	@echo ""
	@echo "Reinstalling package ..."
	@pip install --upgrade pip
	@pip uninstall -y pokedex || :
	@pip install -e .

create_dataset_type :
	@echo ""
	@echo "Creating dataset_type..."
	bash ../scripts/data_pokemon_to_type.sh

reset_local_files:
	@echo ""
	@echo "Resetting local staging registries..."
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
	@echo ""
	@echo "Copying .env from .env.sample ..."
	@if [ -f .env ]; then \
		echo ".env already exists, creating backup as .env.bak ..."; \
		cp .env .env.bak; \
	fi
	cp .env.sample .env
	@echo ""
	@echo "UPDATE ENV VARIABLES IN .env"
	@echo ""
	@echo "Once you are done: make run_pred | run_api_local "


#################### PACKAGE ACTIONS ###################
# Always start the project with make start

start: start_env reinstall_package create_dataset_type reset_local_files set_up_env
	direnv allow


run:
	direnv reload
	direnv exec . python pokedex/interface/main.py

reset_classification_env:
	@echo ""
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

run_GAN:
	$(MAKE) reset_classification_env new_value=GAN
	$(MAKE) run



run_test:
	python -c 'from pokedex.interface.main import pred_both; pred_both(test=True)'

run_test_15:
	python -c 'from pokedex.interface.main import pred; pred('15', test=True)'

run_test_150:
	python -c 'from pokedex.interface.main import pred; pred('150', test=True)'



run_pred:
	python -c 'from pokedex.interface.main import pred_both; pred_both()'

run_pred_15:
	python -c 'from pokedex.interface.main import pred; pred('15')'

run_pred_150:
	python -c 'from pokedex.interface.main import pred; pred('150')'



run_api_local:
	uvicorn pokedex.api.fast:app --host 0.0.0.0 --port 8002
# uvicorn pokedex.api.fast:app --reload



run_save_model_to_gcs_15:
	python -c 'from pokedex.model_logic.registry import save_production_model_to_gcs; save_production_model_to_gcs('15')'

run_save_model_to_gcs_150:
	python -c 'from pokedex.model_logic.registry import save_production_model_to_gcs; save_production_model_to_gcs('150')'




##### DEPLOYMENT ######
GAR_IMAGE=pokedex
DOCKER_IMAGE=europe-west9-docker.pkg.dev/pokemon1672/pokedex/pokedex:prod
GCP_REGION=europe-west9

docker_build_local:
	docker build --tag=$(GAR_IMAGE):dev .

docker_run_local:
	docker run -it $(GAR_IMAGE):dev

docker_build_cloud:
	docker build -t  $(DOCKER_IMAGE) .

docker_run_cloud:
	docker run -e PORT=8000 -p 8000:8000 --env-file .env $(DOCKER_IMAGE)

docker_run_cloud_iterative:
	docker run -it -e PORT=8000 -p 8000:8000 --env-file .env $(DOCKER_IMAGE) bash

docker_push_cloud:
	docker push $(DOCKER_IMAGE)

docker_deploy_cloud:
	gcloud run deploy --image $(DOCKER_IMAGE) --memory 2Gi --region $(GCP_REGION) --env-vars-file .env.yaml --port 8000

# Variables
PYTHON_VERSION := 3.10.6
ENV_NAME := pokemon-env


create_type_dataset:
	bash scripts/data_pokemon_to_type.sh

install_dependencies:
	pip install --upgrade pip
	pip install -r requirements.txt

# To do after cloning the project
connect:
	@printf $(BOLDCYAN)"Makefile: Connecting team git repositories\n"$(RESET)
	./scripts/git_connect_repo.sh

# .python-version already created
# start_env:
# 	pyenv virtualenv $(PYTHON_VERSION) $(ENV_NAME)
# 	pyenv local $(ENV_NAME)


.PHONY: all start_env connect

# Colors
RESET					:= "\033[0m"
BLACK					:= "\033[30m"
RED						:= "\033[31m"
GREEN					:= "\033[32m"
YELLOW					:= "\033[33m"
BLUE					:= "\033[34m"
MAGENTA					:= "\033[35m"
CYAN					:= "\033[36m"
WHITE					:= "\033[37m"
BOLDBLACK				:= "\033[1m\033[30m"
BOLDRED					:= "\033[1m\033[31m"
BOLDGREEN				:= "\033[1m\033[32m"
BOLDYELLOW				:= "\033[1m\033[33m"
BOLDBLUE				:= "\033[1m\033[34m"
BOLDMAGENTA				:= "\033[1m\033[35m"
BOLDCYAN				:= "\033[1m\033[36m"
BOLDWHITE				:= "\033[1m\033[37m"

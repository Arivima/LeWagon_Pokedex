# Variables
PYTHON_VERSION := 3.10.6
ENV_NAME := pokemon-env

# .python-version already created
# start_env:
# 	pyenv virtualenv $(PYTHON_VERSION) $(ENV_NAME)
# 	pyenv local $(ENV_NAME)

install_dependencies:
	pip install --upgrade pip
	pip install -r https://gist.githubusercontent.com/krokrob/53ab953bbec16c96b9938fcaebf2b199/raw/9035bbf12922840905ef1fbbabc459dc565b79a3/minimal_requirements.txt
	pip list

# To do after cloning the project
connect:
	@printf $(BOLDCYAN)"Makefile: Connecting team git repositories\n"$(RESET)
	./scripts/git_connect_repo.sh


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

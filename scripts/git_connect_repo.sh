#!/bin/bash

#! TODO before this script
# add other users to your repo as collaborators if not done already

#! FOR INFORMATION, IF NEEDED USE MANUALLY, DO NOT UNCOMMENT
#! Following commands  = to delete repos from the push list
# git remote set-url --delete --push origin Just-PH/lewagon-pokedex-gan.git
# git remote set-url --delete --push origin 0xCez/Pokemon-GenAI.git
# git remote set-url --delete --push origin edouardfoussier/lewagon-pokedex.git
# git remote set-url --delete --push origin Arivima/LeWagon_Pokedex.git

#! persistency of setting
#! this script needs to be executed each time you git clone the project
# TODO : check if possible to be changed with git Actions

# colors
RESET='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'

# Define the remote repository URL
REMOTE_REPO_ARIELLE="git@github.com:Just-PH/lewagon-pokedex-gan.git"
REMOTE_REPO_CESAR="git@github.com:0xCez/Pokemon-GenAI.git"
REMOTE_REPO_EDOUARD="git@github.com:edouardfoussier/lewagon-pokedex.git"
REMOTE_REPO_PH="git@github.com:Arivima/LeWagon_Pokedex.git"

# check current settings
printf ${CYAN}"\nThese are the original settings for origin\n"${RESET}
git remote show origin
printf ${CYAN}"\n('Enter' to continue, 'q' to exit): \n"${RESET}
read USER_INPUT
if [ "$USER_INPUT" == 'q' ]; then
    exit 0;
fi

# Check if the remotes already exist, and if not, add it
printf ${CYAN}"\nNow adding the team repos ${RESET}\n"
REPO_ARRAY=($REMOTE_REPO_ARIELLE $REMOTE_REPO_CESAR $REMOTE_REPO_PH $REMOTE_REPO_EDOUARD)
for element in "${REPO_ARRAY[@]}"
do
    if ! git remote show origin | grep -q "Push  URL: $element"; then
        git remote set-url --add --push origin $element
        printf ${GREEN}"Remote repository added : $element\n"${RESET}
    fi
done

# Check if user repo is indeed in the list (can be replaced sometimes)
REPO_ARRAY=($REMOTE_REPO_ARIELLE $REMOTE_REPO_CESAR $REMOTE_REPO_PH $REMOTE_REPO_EDOUARD)
for element in "${REPO_ARRAY[@]}"
do
    if (git remote show origin | grep -q "Fetch URL: $element" &&
        ! git remote show origin | grep -q "Push  URL: $element"); then
        git remote set-url --add --push origin $element
        printf ${GREEN}"Remote repository added : $element\n"${RESET}
    fi
done

# Test the final settings
printf ${CYAN}"\nFinal configuration\n"${RESET}
git remote show origin

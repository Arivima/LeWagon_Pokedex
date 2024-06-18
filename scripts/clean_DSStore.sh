#!/bin/bash

# Définir le répertoire de base
base_dir="../all_data_name_cleaned"

# Utiliser find pour localiser et supprimer les fichiers :Zone.Identifier
find "$base_dir" -type f -name '*.DS_Store' -exec rm -f {} \;

echo

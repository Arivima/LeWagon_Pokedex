#!/bin/bash

# Définir le répertoire de base
base_dir="../all_data_type_cleaned"

# Utiliser find pour localiser et supprimer les fichiers :Zone.Identifier
find "$base_dir" -type f -name '*:Zone.Identifier' -exec rm -f {} \;

echo

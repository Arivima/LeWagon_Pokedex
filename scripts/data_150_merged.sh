#!/bin/bash

# creates a folder 'raw_data_all' where all pokemon images are sorted in 150 folders


# Définir les chemins
original_dir="raw_data"
new_dir="all_data_name"
csv_file="scripts/FirstGenPokemon.csv"



# Créer les nouveaux dossiers de type s'ils n'existent pas
mkdir -p "$new_dir"


# Lire le fichier CSV
while IFS=, read -r Number Name  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _; do
    # Ignorer la ligne d'en-tête
    if [[ "$Number" == "Number" ]]; then
        continue
    fi

    # Copier les fichiers dans les dossiers de type appropriés
    for dataset in train test valid; do
        if [ -d "$original_dir/$dataset/$Name" ]; then
            # Créer le dossier pour le Pokemon s'il n'existe pas
            mkdir -p "$new_dir/$Name"
            # Copier les fichiers dans le dossier de Type1
            cp "$original_dir/$dataset/$Name"/* "$new_dir/$Name/"
        fi
    done
done < "$csv_file"

# supprimer le dossier original
rm -rf "$original_dir"

echo "Les images ont été réorganisées avec succès."

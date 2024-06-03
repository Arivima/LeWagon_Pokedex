#!/bin/bash

# creates a folder 'raw_data_types' where all pokemon images are sorted by type within test/train/valid datasets


# Définir les chemins
original_dir="raw_data"
new_dir="raw_data_type"
csv_file="scripts/FirstGenPokemon.csv"

# Créer les nouveaux dossiers de type s'ils n'existent pas
mkdir -p "$new_dir/train"
mkdir -p "$new_dir/test"
mkdir -p "$new_dir/valid"

# Lire le fichier CSV
while IFS=, read -r Number Name Types Type1 Type2 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _; do
    # Ignorer la ligne d'en-tête
    if [[ "$Number" == "Number" ]]; then
        continue
    fi

    # Copier les fichiers dans les dossiers de type appropriés
    for dataset in train test valid; do
        if [ -d "$original_dir/$dataset/$Name" ]; then
            # Créer le dossier pour le Type1 s'il n'existe pas
            mkdir -p "$new_dir/$dataset/$Type1"
            # Copier les fichiers dans le dossier de Type1
            cp "$original_dir/$dataset/$Name"/* "$new_dir/$dataset/$Type1/"

            # Si le Pokémon a plus d'un type
            if [ "$Types" == 2 ]; then
                # Créer le dossier pour le Type2 s'il n'existe pas
                mkdir -p "$new_dir/$dataset/$Type2"
                # Copier les fichiers dans le dossier de Type2
                cp "$original_dir/$dataset/$Name"/* "$new_dir/$dataset/$Type2/"
            fi
        fi
    done
done < "$csv_file"

echo "Les images ont été copiées avec succès dans les dossiers de types appropriés."

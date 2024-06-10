#!/bin/bash

# creates a folder 'raw_data_types' where all pokemon images are sorted by type within test/train/valid datasets


# Définir les chemins
original_dir="../all_data_name_cleaned"
new_dir="../all_data_type_cleaned"
csv_file="FirstGenPokemon.csv"

# Lire le fichier CSV
while IFS=, read -r Number Name Types Type1 Type2 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _; do
    # Ignorer la ligne d'en-tête
    if [ "$Number" == "Number" ]; then
        continue
    fi

    # Copier les fichiers dans les dossiers de type appropriés
    if [ -d "$original_dir/$Name" ]; then
        if [ "$Type2" == "flying" ] && [ "$Type1" == "normal" ]; then
            # Créer le dossier pour le Type2 s'il n'existe pas
            mkdir -p "$new_dir/$Type2"
            # Copier les fichiers dans le dossier de Type2
            cp "$original_dir/$Name"/* "$new_dir/$Type2/"
        else
            # Créer le dossier pour le Type1 s'il n'existe pas
            mkdir -p "$new_dir/$Type1"
            # Copier les fichiers dans le dossier de Type1
            cp "$original_dir/$Name"/* "$new_dir/$Type1/"
        fi

    fi
done < "$csv_file"

echo "Les images ont été copiées avec succès dans les dossiers de types appropriés."

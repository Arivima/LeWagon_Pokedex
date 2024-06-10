# Use the official Python image from the Docker Hub
FROM python:3.10.6-slim-buster

# COPY -> select the folder you need
COPY pokedex pokedex
COPY requirements.txt requirements.txt

# RUN run terminal command
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# install your package
COPY setup.py setup.py
RUN pip install .
RUN apt update && apt install -y libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx
EXPOSE 8000

# controls functinnality of the container
# uvicorn to control the server port
# local

# CMD uvicorn pokedex.api.fast:app --host 0.0.0.0

# Deploy to GCP
CMD uvicorn app_folder.app_file:app --host 0.0.0.0 --port 8080

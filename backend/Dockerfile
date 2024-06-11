# Use the official Python image from the Docker Hub
FROM python:3.10.6-slim-buster

# COPY -> select the requirements
COPY requirements.txt requirements.txt

# RUN run terminal command
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# COPY -> select the folder you need
COPY pokedex pokedex
COPY Makefile Makefile

# install your package
COPY setup.py setup.py
RUN pip install .
RUN apt update && apt install -y libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx

# controls functinnality of the container
# uvicorn to control the server port
# Deploy to GCP
CMD uvicorn pokedex.api.fast:app --host 0.0.0.0 --port 8000

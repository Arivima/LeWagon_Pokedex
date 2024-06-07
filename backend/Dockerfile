# Use the official Python image from the Docker Hub
# FROM python:3.10.6-alpine
FROM python:3.10.6-slim-buster

COPY requirements.txt /requirements.txt
COPY pokedex /pokedex
COPY setup.py /setup.py
COPY Makefile Makefile
COPY .env.sample .env

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN pip install -e .

# RUN apt-get update && apt-get install mesa-libGL

RUN mkdir -p ~/.lewagon/pokedex/data/
RUN mkdir ~/.lewagon/pokedex/data/raw
RUN mkdir ~/.lewagon/pokedex/data/processed
RUN mkdir ~/.lewagon/pokedex/training_outputs
RUN mkdir ~/.lewagon/pokedex/training_outputs/metrics
RUN mkdir ~/.lewagon/pokedex/training_outputs/models
RUN mkdir ~/.lewagon/pokedex/training_outputs/params


# RUN pip install -r requirements.txt

# RUN reset_local_files
# RUN mkdir -p ~/.lewagon/pokedex/data/
# RUN mkdir ~/.lewagon/pokedex/data/raw
# RUN mkdir ~/.lewagon/pokedex/data/processed
# RUN mkdir ~/.lewagon/pokedex/training_outputs
# RUN mkdir ~/.lewagon/pokedex/training_outputs/metrics
# RUN mkdir ~/.lewagon/pokedex/training_outputs/models
# RUN mkdir ~/.lewagon/pokedex/training_outputs/params

# Start FastAPI
CMD uvicorn pokedex.api.fast:app --host 0.0.0.0

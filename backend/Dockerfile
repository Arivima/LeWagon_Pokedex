# Use the official Python image from the Docker Hub
FROM python:3.10-slim-bullseye

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Start FastAPI
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

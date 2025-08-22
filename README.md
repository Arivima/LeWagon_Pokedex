# Pokedex_project_V1.672  @ Le Wagon
  <p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white" alt="python" height="26" /></a>
  <a href="#"><img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="numpy" height="26" /></a>
  <a href="#"><img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas" height="26" /></a>
  <a href="#"><img src="https://img.shields.io/badge/matplotlib-175880.svg?style=for-the-badge&logo=matplotlib&logoColor=white" alt="matplotlib" height="26" /></a>
  <br>
  <a href="#"><img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="tf" height="26" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white" alt="keras" height="26" /></a>
  <a href="#"><img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="sklearn" height="26" /></a>
  <br>
  <a href="#"><img src="https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white" alt="gcp" height="26" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Docker-0092e7.svg?style=for-the-badge&logo=docker&logoColor=white" alt="docker" height="26" /></a>
  <a href="#"><img src="https://img.shields.io/badge/mlflow-blue.svg?style=for-the-badge&logo=mlflow&logoColor=white" alt="mlflow" height="26" /></a>
  <br>
  <a href="#"><img src="https://img.shields.io/badge/fastapi-05978a.svg?style=for-the-badge&logo=fastapi&logoColor=white" alt="fastapi" height="26" /></a>
  <a href="#"><img src="https://img.shields.io/badge/uvicorn-pink.svg?style=for-the-badge&logo=gunicorn&logoColor=purple" alt="uvicorn" height="26" /></a>
  </p>

This project was done at Le Wagon in two weeks, during the Data Science Bootcamp. With this project you can classify by types and names the 151 first pocket monsters (A.K.A Pokémon), thanks to a CNN model. But that's not it ! You can also generate new ones thanks to a GAN model 🔥

To test the app, go on this website : https://pokemon-generator-1672.streamlit.app/

## Installation
First let's clone the repository :
```
git clone https://github.com/Just-PH/lewagon-pokedex-gan.git
```

Then run the installation :
```
cd backend
make start
```
⚠️ Then you need to modify in the file /backend/.env the variable WHO with your name ⚠️

## Test

Still in /backend
To test both predictions functions on all images :
```
make run_test
```
If you only want to test for the types :
```
make run_test_15
```
If you only want to test for the names :
```
make run_test_150
```
## Predictions

⚠️ PUT any image you want to test in the repository all_prediction_images/images ⚠️

To predict them :
```
make run_pred
```
If you only want to predict for the types :
```
make run_pred_15
```
If you only want to predict for the names :
```
make run_pred_150
```

## Generation

To generate this kind of images :

![Example of generated fakemon :](output_gan/Example/Example_gan.jpeg)

You can use this command :

```
make run_generate
```
The image will be in the repository named output_gan

## Api
To run locally the api :
```
make run_api_local
```

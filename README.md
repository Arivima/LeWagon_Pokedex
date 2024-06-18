# Pokedex_project_V1.672 at Le Wagon

This project was done at le Wagon in two weeks, during the Data Science Bootcamp. With this project you can classify by types and names the 151 first pocket monster (A.K.A Pokémon), thanks to a CNN model. But that's not it ! You can also genrate new one thanks to a GAN model.

## Installation
First let's clone the repository :
```
git clone https://github.com/Just-PH/lewagon-pokedex-gan.git
```

The run the installation :
```
cd backend
make start
```
!!! Then you need to modify in file /backend/.env the variable WHO with your name !!!

## Test

Still in /backend
To test the both predictions function on all images :
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

!!! PUT any images you want to test in the repository all_prediction_images/images !!!

To predict them :
```
make pred_test
```
If you only want to predict for the types :
```
make pred_test_15
```
If you only want to predict for the names :
```
make pred_test_150
```

## Api
To run locally the api :
```
make run_api_local
```

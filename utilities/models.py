# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 10:45:30 2022

@author: Léa
"""

# Importer les librairies python 
#----------------------------------------------
#----- Arguments 
import argparse
#----- Calculs 
import pandas as pd
import numpy as np
from math import sqrt, log
#----- Machine Learning models
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
import catboost as cb 
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
# tuning des paramètres
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import matplotlib.pyplot as plt
#performance indicators 
from sklearn.metrics import mean_squared_error
#----- Sauvegarder et loader les modèles 
import pickle 

import warnings

warnings.filterwarnings("ignore")
import sys

# Importer nos propres librairies 
#----------------------------------------------

import os
from set_path import *
from adversarial_validation import main_adversarial_validation
import data_preprocessing as DP

# Implémentation des fonctions 
#----------------------------------------------
def xgboost(train, X_train, Y_train, X_val, Y_val, n_estimators=2000, max_depth=10, min_child_weight=4, learning_rate=0.05,
                           subsample=0.9):
  """
  Cette fonction permet de créer le modèle XGboost avec scaling des données (pipeline)
  Input:
  -----
  -train(booléen): indique si on veut réoptimiser les paramètres du modèle par cross validation
  - X_train(dataframe): ensemble des X d'entrainement 
  - y_train(dataframe): ensemble des valeurs de prix que l'on souhaite prédire (entrainement)
  - X_val(dataframe): ensemble des X de validation
  - y_val(dataframe): ensemble des valeurs de prix que l'on souhaite prédire (validation)
  -n_estimators(int): number of trees of the method
  -max_depth(int) : profondeur maximale d'un arbre (à partir du sommet)
  -min_child_weight(int): nombre minimale de données dans un noeud terminal (sinon on fusionne)
  -learning_rate(float): learning rate de la méthode 
  -subsample(float): percentage of data taken to build each tree
  
  Output:
  ------
  model
  """
  if train==1:
      modelML = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, min_child_weight=min_child_weight, learning_rate=learning_rate,
                           subsample=subsample, silent=0, colsample_bytree=0.7, nthread=4, objective = 'reg:squarederror')
  else :
      #la première étape est de choisir le nombre d'arbres que l'on veut utiliser dans notre modèle de boosting. 
      #Pour ce faire, on va faire de l'early stopping pour trouver le nombre d'arbres à conserver 
      #de manière à ne pas overfitter 
      model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=3000, max_depth=6, learning_rate=0.1)
      model.fit(X_train, Y_train, 
                eval_set=[(X_train, Y_train), (X_val, Y_val)], 
                early_stopping_rounds=20) 
      results = model.evals_result()
      plt.figure(figsize=(10,7))
      begin=100
      x=np.arange(begin, 3000, 1)
      plt.plot(x, results["validation_0"]["rmse"][begin:], label="Training loss")
      plt.plot(x, results["validation_1"]["rmse"][begin:], label="Validation loss")
      plt.axvline(n_estimators, color="gray", label="current number of trees")
      plt.xlabel("Number of trees")
      plt.ylabel("Loss")
      plt.legend()
      plt.savefig("early_stopping_xgboost.png")
      xgb1=xgb.XGBRegressor()
      parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
                    'objective':["reg:squarederror"],#: regression with squared loss.
                    'learning_rate': [0.01, 0.03, 0.05], #so called `eta` value
                    'max_depth': [5, 10],
                    'min_child_weight': [4],
                    'silent': [0], #only print the warning 
                    'subsample': [0.9],
                    'colsample_bytree': [0.7],
                    'n_estimators': [500, 800, 1000]} 
      
      xgb_grid = GridSearchCV(xgb1,
                              parameters,
                              cv = 5,
                              n_jobs = 5,
                              verbose=True)
      xgb_grid.fit(X_train,
                 Y_train)
      print(xgb_grid.best_params_)
      modelML = xgb.XGBRegressor()
      modelML.set_params(**xgb_grid.best_params_) 
  return modelML
  
def lightgbm(train, X_train, Y_train, colsample_bytree=0.7,learning_rate=0.5, max_depth=10, n_estimators=2000,
                           subsample=0.9):
  """
  Cette fonction permet de créer le modèle XGboost avec scaling des données (pipeline)
  Input:
  -----
  -train(booléen): indique si on veut réoptimiser les paramètres du modèle par cross validation
  - X_train(dataframe): ensemble des X d'entrainement 
  - y_train(dataframe): ensemble des valeurs de prix que l'on souhaite prédire (entrainement)
  - X_val(dataframe): ensemble des X de validation
  - y_val(dataframe): ensemble des valeurs de prix que l'on souhaite prédire (validation)
  -n_estimators(int): number of trees of the method
  -max_depth(int) : profondeur maximale d'un arbre (à partir du sommet)
  -min_child_weight(int): nombre minimale de données dans un noeud terminal (sinon on fusionne)
  -learning_rate(float): learning rate de la méthode 
  -subsample(float): percentage of data taken to build each tree
  
  Output:
  ------
  model
  """
  if train==1:
      modelML = lgb.LGBMRegressor(colsample_bytree=colsample_bytree,learning_rate= learning_rate,max_depth= max_depth,n_estimators=n_estimators,
      nthread= 4,subsample=subsample,n_jobs = 2)
  else :
      print("erreur")
  return modelML

def randomforest(train, X_train, Y_train, n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=1):
  """
  Cette fonction permet de créer le modèle RandomForest avec scaling des données (pipeline)
  Input:
  -----
  - train(int): indique si on veut réoptimiser les paramètres du modèle par cross validation(1: pas d'optimisation, 2: optimisation des paramètres)
  - X_train(dataframe): ensemble des X d'entrainement 
  - y_train(dataframe): ensemble des valeurs de prix que l'on souhaite prédire (entrainement)
  - X_val(dataframe): ensemble des X de validation
  - y_val(dataframe): ensemble des valeurs de prix que l'on souhaite prédire (validation)
  -n_estimators(int): number of trees of the method
  - max_depth(int) : profondeur maximale d'un arbre (à partir du sommet)
  - min_samples_split(int): nombre minimal de données pour faire un split
  - min_samples_leaf(int): nombre minimal de samples dans une feuille
  
  Output:
  ------
  model
  """
  if train==1:
    modelML = RandomForestRegressor(n_estimators=n_estimators, max_depth= max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=True)
  if train ==2:
    parameters = {'bootstrap': [True],
    'max_depth': [5, 10, 20, 50],
    'n_estimators': [100, 200, 500, 800, 1000]}

    rbf_grid = GridSearchCV(RandomForestRegressor(),
                         parameters,
                            cv = 5,
                            n_jobs = 5,
                            verbose=True)
    rbf_grid.fit(X_train, Y_train)
    print(rbf_grid.best_params_)
    modelML = RandomForestRegressor()
    modelML.set_params(**rbf_grid.best_params_) 
  return modelML
  
def catboost(train, X_train, Y_train, depth=10, learning_rate=0.2, iterations=300, l2_leaf_reg=0.1):
  """
  Cette fonction permet de créer le modèle Catboost avec scaling des données (pipeline)
  Input:
  -----
  - train(int): indique si on veut réoptimiser les paramètres du modèle par cross validation(1: pas d'optimisation, 2: optimisation des paramètres)
  - X_train(dataframe): ensemble des X d'entrainement 
  - y_train(dataframe): ensemble des valeurs de prix que l'on souhaite prédire (entrainement)
  - X_val(dataframe): ensemble des X de validation
  - y_val(dataframe): ensemble des valeurs de prix que l'on souhaite prédire (validation)
  -n_estimators(int): number of trees of the method
  - depth(int) : profondeur maximale d'un arbre (à partir du sommet)
  - learning_rate(float): taux d'apprentissage
  - iterations(int): nombre d'itérations
  - l2_leaf_reg(float): coefficient de régularisation pour éviter le surapprentissage 
  
  Output:
  ------
  model
  """
  if train==1:
    modelML = cb.CatBoostRegressor(loss_function="RMSE", depth=depth, learning_rate=learning_rate, iterations=iterations, l2_leaf_reg=l2_leaf_reg)
    

  if train ==2:
    grid={
    "iterations": [100, 150, 200, 300, 500, 600, 700, 800],
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "depth": [2, 4, 6, 8],
    "l2_leaf_reg": [0.2, 0.5, 1, 3]
    }
    ctb_grid = GridSearchCV(estimator=cb.CatBoostRegressor(loss_function="RMSE"), param_grid = grid, cv = 2, n_jobs=-1)
    ctb_grid.fit(X_train, Y_train)
    print(ctb_grid.best_params_)
    modelML = cb.CatBoostRegressor(loss_function="RMSE")
    modelML.set_params(**ctb_grid.best_params_) 
  return modelML

def train_and_save_model(modelML, X_train, y_train, namesave): 
  """
  Cette fonction permet d'entrainer un modèle et d'enregistrer les poids du modèle 
  Input:
  -----
  - X_train: ensemble des X d'entrainement 
  - y_train: ensemble des valeurs de prix que l'on souhaite prédire (entrainement)
  -namesave(str): nom d'enregistrement des poids du modèle
  -hotelid(bool):doit on supprimer la colonne hotelid
  Output:
  -----
  model trained (modèle entrainé à partir duquel on peut faire de l'inférence) 
 """
  to_remove=[]
  #if hotelid:
    #to_remove.append("hotel_id")
  #model=make_pipeline(columnDropperTransformer(to_remove), StandardScaler(), modelML)
  model=make_pipeline(StandardScaler(), modelML)
  model.fit(X_train, Y_train)
  ypred=model.predict(X_train)
  print("erreur du modèle", mean_squared_error(Y_train**2, ypred**2))
  filename = namesave # enregistrement du modèle dans X_gboost
  pickle.dump(model, open(PATH_MODELS+"/"+filename, 'wb')) # enregistrement du fichier 
  return model

def inference(model, X_test, y_test): 
  """
  Cette fonction permet de faire de l'inference sur le modèle qu'on a préalablement entrainé
  Input:
  -----
  - model : sklearn modèle qui a été entrainé
  - X_test: ensemble des X de test
  - y_test: ensemble des valeurs de prix que l'on souhaite prédire (test)
  ----
  Output:
  -mse(float):erreur entre ce qui est prédit et ce qu'on voulait prédire
  -score(float): score du modèle
  -ypred(pandaframe): prédiction de l'algorithme 
  """
  ypred = model.predict(X_test) 
  mse = mean_squared_error(Y_test**2, ypred**2) 
  #Calcul du mse entre les prédictions. ATTENTION : les mettre au carré 
  #car le prix doit être au carré car on a fait une transformation racine carrée
  return mse, model.score(X_test, Y_test), ypred
  
class columnDropperTransformer(): #class used to remove columns 
    def __init__(self,columns):
        self.columns=columns

    def transform(self,X,y=None):
        return X.drop(self.columns,axis=1)

    def fit(self, X, y=None):
        return self 
  
if __name__ == "__main__":
  #choix du type d'encoding, du modèle et si on veut réoptimiser ou non les paramètres par rapport à ceux déjà appris
  parser = argparse.ArgumentParser()
  parser.add_argument('--name', type=int, default=0, help='0: XGboost, 1: Random Forest, 2:Catboost, 3:lightgbm')
  parser.add_argument('--train', type=int, default=1, help='0: just test the model(inference), 1:train without parameters optimization, 2: train with gridsearch parameters optimisation')
  parser.add_argument('--adversarial', type=int, default=0, help=' 0: not adversarial validation, 1: adversarial validation')
  parser.add_argument('--encoding', type=int, default=0, help='0: target+frequency encoding, 1: one hot encoding')
  parser.add_argument('--hotels', type=str, default='features_hotels.csv', help='name of the features hotels')
  parser.add_argument('--dataset', type=str, default='pricing_requests_done.csv', help='name of the dataset')
  parser.add_argument('--name_save', type=str, default="./xgboost_model_sav.sav", help="name of the file .sav")
  parser.add_argument('--hotelid', type=int, default=1, help="1 : on enlève hotelid des variables explicatives (0 sinon)")
  parser.add_argument('--drop_duplicates', type=int, default=1, help="1 : on supprime les doublons (0 sinon)")
  parser.add_argument('--save_train_test', type=int, default=0, help="1: on enregistre X_train, y_train, y_test, X_test, 0 sinon")
  args = parser.parse_args()
  train =  args.train
  adversarial = args.adversarial
  encoding = args.encoding 
  name = args.name
  hotels = args.hotels
  dataset = args.dataset
  name_save = args.name_save
  hotelid = args.hotelid
  save_train_test=args.save_train_test
  drop_duplicates=args.drop_duplicates
  
  #------------------------------------------------------------------------------------------------------
  
  encoding_vec=["targetFreq", "oneHot"]
  print("début de la création des datasets (train, test, validation")
  print(adversarial, hotels, train, drop_duplicates)
  #----------------création du jeu de donnée train, test et validation à partir de packages préalablement implémentés 
  X_train,Y_train,X_val,Y_val,X_test,Y_test = DP.main_data_preprocessing(
  name_requests=dataset,
  name_featHotels=hotels,
  encoding=encoding_vec[encoding],
  name_testKaggle="test_set_cleaned_addedFeatures.csv", 
  price=True,stock=True,
  drop_requests_duplicates=drop_duplicates,
  adversarial_validation= adversarial,
  save_encoding=True,
  encode_all_same=False,
  save_datasets=save_train_test)
  
  print("création des datasets ok")
  #suppression de la colonne request_number, inutile ici
  X_train = X_train.drop('request_number', axis=1)
  X_test = X_test.drop('request_number', axis=1)
  X_val = X_val.drop('request_number', axis=1)
  #entraînement du modèle
  parameters = ["request_number"]
  if hotelid==1:
    parameters.append("hotel_id")
    X_train = X_train.drop('hotel_id', axis=1)
    X_test = X_test.drop('hotel_id', axis=1)
    X_val = X_val.drop('hotel_id', axis=1)
  pickle.dump(parameters, open(PATH_MODELS+"/"+"parameters_"+name_save, 'wb'))
  print(X_train.columns.tolist())
 
  if (train==1 or train==2): #on veut entrainer le modèle avec des paramètres qu'on a déterminé avant 
    if name==0:
      modelML = xgboost(train, X_train, Y_train, X_val, Y_val, n_estimators=2000, max_depth=10, min_child_weight=4, learning_rate=0.05,
                           subsample=0.9)
    if name==1:
      modelML = randomforest(train, X_train, Y_train, n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=1) 
    if name==2:
      modelML = catboost(train, X_train, Y_train) 
    if name==3:
      modelML = lightgbm(train, X_train, Y_train)
    model_trained = train_and_save_model(modelML, X_train, Y_train, name_save)
  if train==0: #on n'entraine pas de modèle, on récupère juste les poids du modèle
      model_trained = pickle.load(open(PATH_MODELS+"/"+name_save, 'rb')) #recharger le modèle
  
  #inférence du modèle 
  mse, score, _ = inference(model_trained, X_test, Y_test)
  print("MSE: %.2f" % mse)
  print("le score en inférence vaut %.2f" % score)
    

        

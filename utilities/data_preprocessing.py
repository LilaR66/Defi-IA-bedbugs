# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:33:09 2022

@author: Lila
"""

# Importer les librairies python 
#----------------------------------------------
#----- Calculs 
import pandas as pd
import numpy as np
from math import sqrt, log
#----- Machine Learning Preprocessing 
from sklearn.model_selection import train_test_split
import category_encoders as ce #pip install category_encoders #target encoding 
#----- Save model 
import pickle 

import warnings
warnings.filterwarnings("ignore")

# Importer nos propres librairies 
#----------------------------------------------
from set_path import *
from adversarial_validation import main_adversarial_validation


# Implémentation des fonctions 
#----------------------------------------------
def retrieve_data(name_requests,name_featHotels):
  """
  Cette fonction permet de charger les fichiers .csv donnés en argument en
  tant que dataframe et d'associer le bon type aux données.

  Input:
  -----
  - name_requests (str): nom du fichier .csv contenant toutes les requêtes,
    situé à l'emplacement indiqué par PATH_DATA.
  - name_featHotels (str): nom du fichier .csv contenant les caractéristiques
    des hôtels, situé à l'emplacement indiqué par PATH_DATA.

  Output:
  ------
  - pricing_requests (dataframe): dataframe contenant la concantéantion de 
    name_requests et name_featHotels. 
    Le bon type a été associé à chaque variable.
  """
  # Récupération du fichier contentant l'ensemble des données issues des requêtes
  pricing_requests = pd.read_csv(PATH_DATA + '/' + name_requests)
  # Récupération du fichier contenant les features des hotels
  hotels = pd.read_csv(PATH_DATA + '/' + name_featHotels, index_col=['hotel_id', 'city']) 
  # Jointure des deux fichiers 
  pricing_requests = pricing_requests.join(hotels, on=['hotel_id', 'city'])
  # Convert to integer: 
  int_list = INT_VAR.copy()
  int_list.remove("request_nb")
  pricing_requests[int_list] = pricing_requests[int_list].astype(int) 
  # Convert to categorical: 
  cat_list = CAT_VAR
  for var in cat_list:
    pricing_requests[var] = pd.Categorical(pricing_requests[var],ordered=False)
  return pricing_requests



def add_external_features(pricing_requests) :
  """
  Cette fonction Ajoute des features extérieures comme:
  - le PIB du pays,
  - le prix moyen par m2 dans chaque ville, 
  - le nombre de touristes par an dans chaque ville,
  - le nombre d’habitants par km2 dans chaque ville.

  Input:
  -------
  - pricing_requests (dataframe): ensemble des requêtes avec les features des
    hotels.

  Output:
  ------
  - pricing_requests (dataframe): ensemble des requêtes avec les 
    features des hotels et les features extérieures.
  """
  # Ajout du PIB par pays
  # source: https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal) 
  # estimate by IMF consulté le 17/11/2022.
  #-----------------------------------------
  # dictionnaire contenant la ville et le pib du pays où se situe la ville
  PIB = {"amsterdam": 990583,
      "copenhagen": 386724,
      "paris": 2778090,
      "sofia": 85008,
      "vienna": 468046,
      "rome": 1996934,
      "madrid":1389927,
      "vilnius":68031,
      "valletta":17156}

  # ajout du pib au dataframe général 
  pricing_requests["pib"] = np.zeros(len(pricing_requests),dtype = int)

  for city in PIB.keys():
      city_idx = np.where(pricing_requests["city"]== city)
      pricing_requests["pib"].iloc[city_idx] = PIB[city]

  # Ajout du prix moyen par m2 par ville
  # source: https://checkinprice.com/europe-square-meter-prices/ 
  # consulté le 17/11/22. Données datant de 2018.
  #-----------------------------------------
  # dictionnaire contenant la ville et le prix moyen par m2 en euros
  price_m2 = {"amsterdam": 4610,
              "copenhagen": 5236,
              "paris": 9160,
              "sofia": 1095,
              "vienna": 6550,
              "rome": 3044,
              "madrid": 3540,
              "vilnius": 1469,
              "valletta": 3600}

  # ajout du prix par m2 au dataframe général 
  pricing_requests["price_m2"] = np.zeros(len(pricing_requests),dtype = int)

  for city in price_m2.keys():
      city_idx = np.where(pricing_requests["city"]== city)
      pricing_requests["price_m2"].iloc[city_idx] = price_m2[city]


  # Ajout du nombre de touristes par pays en 2020
  # source: https://www.indexmundi.com/facts/indicators/ST.INT.ARVL/rankings 
  # consulté 17/11/22. Chiffres de 2020.
  #-----------------------------------------
  # dictionnaire contenant le nombre de touristes par pays en 2020
  nb_tourists = {"amsterdam": 7265, 
                  "copenhagen": 15595 ,
                  "paris": 117109,
                  "sofia": 4973,
                  "vienna": 15091,
                  "rome": 38419,
                  "madrid": 36410,
                  "vilnius": 2284,
                  "valletta": 718}
  # ajouter 10^3 à toutes les valeurs pour obtenir les vraies valeurs

  # ajout du nombre de touristes par pays
  pricing_requests["nb_tourists"] = np.zeros(len(pricing_requests),dtype = int)

  for city in nb_tourists.keys():
      city_idx = np.where(pricing_requests["city"]== city)
      pricing_requests["nb_tourists"].iloc[city_idx] = nb_tourists[city]

    
  # Ajout du nombre de d'habitants par km2 dans chaque ville
  # source: wikikepia: https://fr.wikipedia.org/wiki/La_Valette, 
  # même source pour les autres villes. Consulté en 2022. 
  #-----------------------------------------
  # dictionnaire contenant le nombre de touristes par pays en 2020
  nb_hab_km2 = {"amsterdam": 3530, 
              "copenhagen": 7064,
              "paris": 20545,
              "sofia": 7354,
              "vienna": 4607,
              "rome": 2213,
              "madrid": 5437,
              "vilnius": 1432,
              "valletta": 8344}

  # ajout du nombre d'habitants par km2 par ville
  pricing_requests["nb_hab_km2"] = np.zeros(len(pricing_requests),dtype = int)

  for city in nb_hab_km2.keys():
      city_idx = np.where(pricing_requests["city"]== city)
      pricing_requests["nb_hab_km2"].iloc[city_idx] = nb_hab_km2[city]

  return pricing_requests



def normalise_quanti(pricing_requests, price:bool, stock:bool):
    """
    Transforme les variables quantitatives indiquées pour les
    rendre plus gaussiennes.
    
     Input:
    -------
    - pricing_requests (dataframe): ensemble des requêtes avec les features 
      ajoutées.
    - price (bool): si True, applique une transformation sur la variable price
    - stock (bool): si True, applique une transformation sur la variable stock.
    
    Output:
    ------
    - pricing_requests (dataframe): ensemble des requêtes avec les features des
    hotels, avec certaines variables transformées.
    """
    # Variables à modifier pour rendre + gaussiennes 
    if price:
        pricing_requests["price"]=pricing_requests["price"].map(lambda x: sqrt(x))
    if stock:
        pricing_requests["stock"]=pricing_requests["stock"].map(lambda x: log(x+1)) #log(x+1) car log(0) n'existe pas.
    return pricing_requests



def add_request_nb(all_requests, pricing_requests, save_csv = False):
    """
      Ajoute une nouvelle feature : request_nb
      Cette feature correspond au nombre de fois qu'un utilisateur requête la base.
      Nous avons fait des requêtes de façon à ce que les requêtes d'un utilisateur se suivent.
     
      Input:
      -----
      - all_requests (dataframe): ensemble des requêtes (leurs paramètres)
      - pricing_requests (dataframe) : ensemble des requêtes et leurs résultats
      - save_csv = True pour enregistrer les nouveaux dataframe
     
      Output:
      ------
      - all_requests (dataframe): ensemble des requêtes avec request_nb
      - pricing_requests (dataframe) : ensemble des requêtess et leurs résultats avec la colonne request_nb
    """

    all_requests.loc.__setitem__((0, ('request_nb')), 1)
    c = 1
    for i in range(1, len(all_requests)):
        if all_requests.avatar_id.loc[i] == all_requests.avatar_id.loc[i-1] :
            c= c+1
            all_requests.loc.__setitem__((i, ('request_nb')), int(c))
        else :
            c = 1
            all_requests.loc.__setitem__((i, ('request_nb')), int(c))

    pricing_requests = pricing_requests.merge(all_requests , on=['avatar_id', 'date', 'city', 'language', 'mobile'])
    pricing_requests.drop('id_request', axis=1, inplace = True)
    if save_csv :
        pricing_requests.to_csv(PATH_DATA  + "/" +'pricing_requests.csv', index=False)
        all_requests.to_csv(PATH_DATA  + "/" +'all_our_requests_done.csv', index = False)
    
    return pricing_requests


def drop_duplicate (pricing_requests, save_csv = False):
    """
      Elimine les lignes redondantes
     
      Input:
      -----
      - pricing_requests (dataframe) : ensemble des requêtes et leurs résultats
        avec la ligne request_nb ! très important !
      - save_csv = True si enregistrement du dataframe nettoyé
    
      Output:
      ------
      - pricing_requests (dataframe) : ensemble nettoyé des requêtes et leurs résultats
    """
    pricing_requests = pricing_requests.drop_duplicates(subset=['hotel_id', 'price', 'stock', 'city', 'date','language', 'mobile', 'request_nb'], keep='first', inplace=False)
    if save_csv :
        pricing_requests.to_csv(PATH_DATA  + "/" + 'pricing_requests_clean.csv', index=False)
    return pricing_requests


def build_train_test(pricing_requests, print_size:bool= False): 
  """
  Cette fonction sépare le jeu de données en jeu de train, validation et test.

  Input:
  -------
  - pricing_requests (dataframe): ensemble des requêtes avec les features 
    ajoutées.
  - print_size (bool): si True, affiche la taille du jeu de train, 
    validation et test. 

  Output:
  ------
  - X_train,Y_train,X_val,Y_val,X_test,Y_test (dataframe): jeu d'entraînement,
    validation et test avec les labels correspondants.
  """

  # Variables explicatives
  requests_pricingQual=pricing_requests[CAT_VAR]
  request_pricingQuant = pricing_requests[NUM_VAR]

  # Créer une dataframe pour les variables explicatives
  dfC = pd.concat([requests_pricingQual,request_pricingQuant],axis=1)

  # Variable à expliquer 
  Y = pricing_requests["price"]

  # Construction de l'échantillon de test, échantillon de train et échantillon de validation
  X_train, X_test, Y_train, Y_test = train_test_split(dfC, Y, test_size=0.2, random_state=11)
  X_train, X_val, Y_train, Y_val =  train_test_split(X_train, Y_train, test_size=0.25, random_state=11) # 0.25 x 0.8 = 0.2

  if print_size:
    print("X_train ---> taille: {} x {}".format(X_train.shape[0], X_train.shape[1]))
    print("X_val   ---> taille: {}  x {}".format(X_val.shape[0], X_val.shape[1]))
    print("X_test  ---> taille: {}  x {}".format(X_test.shape[0], X_test.shape[1]))

  return X_train,Y_train,X_val,Y_val,X_test,Y_test


def target_freq_encoding(X_train,X_val,X_test,Y_train,encode_all_same=False, 
                       save_encoding=True):
  """
  Cette fonction effectue du target encoding pour et du frequency encoding
  sur les variables qualitatives. On fait du frequency encoding pour les 
  variables qualitatives avec moins de 3 modalités données par la liste 
  CAT_VAR_FREQ, et target encoding pour les autres variables qualitatives.
  Si encoder_toutes_les_variables_pareil = True, toutes les variables sont 
  encodées avec le target encoding.

  #https://towardsdatascience.com/dealing-with-categorical-variables-by-using-target-encoder-a0f1733a4c69
  #https://towardsdatascience.com/all-about-target-encoding-d356c4e9e82

  Input:
  -------
  - X_train,X_val,X_test,Y_train (dataframe): jeu d'entraînement, de 
    de validation et de test ainsi que les labels associés au jeu 
    d'entraînement.
  - save_encoding (bool): si True, le target et frequency encoding est 
    sauvegardé dans le dossier indiqué par PATH_ENCODING. 
  - encode_all_same (bool): Si true, on fait du target encoding sur toutes les
    variables qualitatives. Sinon , on fait du target encoding seulement sur 
    les variables qualitatives contenues dans CAT_VAR et non dans CAT_VAR_FREQ. 

  Output:
  ------
  - X_train_encoding, X_val_encoding, X_test_encoding (dataframe): variables 
    encodées. 
  """

  # Frequency encoding pour les variables quantitatives avec moins de 3 modalités
  # et target encoding pour les autres variables quantitatives. 
  # -----------------------------------------------
  if not encode_all_same:
    #----- Recréer le tableau total avec les prix 
    temp = pd.concat([X_train, Y_train], axis=1) # X_train concaténé avec Y_train

    #----- Frequency encoding 
    for var in CAT_VAR_FREQ:
      frequency_by_hotel_var = (temp.groupby(var).size()) / len(temp) #dictionnaire

      X_train[var] = X_train[var].apply(lambda x : frequency_by_hotel_var[x])
      X_test[var] = X_test[var].apply(lambda x : frequency_by_hotel_var[x])
      X_val[var] = X_val[var].apply(lambda x : frequency_by_hotel_var[x])
      
      X_test[var] = X_test[var].astype(float) 
      X_train[var] = X_train[var].astype(float) 
      X_val[var] = X_val[var].astype(float)

      # Enregistrement des poids du frequency encoding.
      if save_encoding:
        pickle.dump(frequency_by_hotel_var, open(PATH_ENCODING + "/" + str(var), 'wb')) 

    #----- Target encoding  
      # Création du modèle de target encoding
      target_encoding=ce.target_encoder.TargetEncoder(smoothing = 1000, min_samples_leaf = 100)
      
      # Entraînement du target ecodeur
      target_encoding.fit(X_train,Y_train)
      X_train_encoding = target_encoding.transform(X_train)
      X_val_encoding = target_encoding.transform(X_val)
      X_test_encoding = target_encoding.transform(X_test)

    # Enregistrement des poids du target encoding.
      if save_encoding:
        pickle.dump(target_encoding, open(PATH_ENCODING + "/target_encoding", 'wb')) 

  else:
    # Création et entraînement du target encodeur
    target_encoding=ce.target_encoder.TargetEncoder()
    target_encoding.fit(X_train,Y_train)
    X_train_encoding = target_encoding.transform(X_train)
    X_val_encoding = target_encoding.transform(X_val)
    X_test_encoding = target_encoding.transform(X_test)
    
    # Enregistrement des poids du target encoding.
    if save_encoding:
      pickle.dump(target_encoding, open(PATH_ENCODING + "/target_encoding_allVar", 'wb')) 

  return X_train_encoding,X_val_encoding,X_test_encoding


def oneHot_encoding(X_train,X_val,X_test):
  """
  Cette fonction permet de one-hot encoder le jeu 
  d'entraînement, le jeu de validation et le jeu de test. 
  Comme hotel_id a trop de modalité, on le considère comme 
  une variable quantitative. Il n'est donc pas  one-hot encodé. 

  Input:
  -------
  - X_train,X_val,X_test: jeu d'entraînement, de 
    de validation et de test.
    
  Output:
  ------
  - X_train_oh, X_val_oh, X_test_oh (dataframe): variables 
    encodées. 
  """
  dfs = [X_train,X_val,X_test]
  # Comme hotel_id a trop de modalité, on le considère comme 
  # une variable quantitative 
  num_var = NUM_VAR + ['hotel_id']
  cat_var = CAT_VAR.copy()
  cat_var.remove('hotel_id')

  for i in range(len(dfs)):
    X_num = dfs[i][num_var]
    X_cat = pd.get_dummies(dfs[i][cat_var],drop_first = True)
    dfs[i] = pd.concat([X_num,X_cat],axis=1)

  X_train_oh, X_val_oh, X_test_oh = dfs
  return X_train_oh,X_val_oh,X_test_oh


def main_data_preprocessing(name_requests,name_featHotels,encoding,
                          name_testKaggle=None, price:bool=True,stock:bool=True,
                          drop_requests_duplicates:bool = True,
                          adversarial_validation:bool=False,
                          save_encoding:bool=True,encode_all_same:bool=False,
                          save_datasets:bool = False):

    """
    Cette fonction réalise plusieurs étapes de pre-processing des données: 
    - récupérer les requêtes et les features hotels: retrieve_data() 
    - ajoute des features extérieures: add_external_features()
    - Transforme cerataines variables quantitatives pour les rendre 
      plus gaussiennes: normalise_quanti()
    - Réalise l'adversarial validation: main_adversarial_validation() du 
      package adversarial_validation.py
    - Construit le jeu d'entraînement, de validation et de test: 
      build_train_test()
    - Performe du target et frequency encoding ou du one-hot encoding: 
      target_freq_encoding() ou oneHot_encoding().
      >>> Pour plus d'info, se référer à la doc de chaque fonction <<<
    
      Input:
      -----
      - name_requests (str): nom du fichier .csv contenant toutes les requêtes,
        situé à l'emplacement indiqué par PATH_DATA.
      - name_featHotels (str): nom du fichier .csv contenant les caractéristiques
        des hôtels, situé à l'emplacement indiqué par PATH_DATA.
      - encoding (str): peut prendre les valeurs "targetFreq" pour effectuer
        du target et frequency encoding ou "oneHot" pour faire du one-hot
        encoding des variables qualitatives.
      - price,stock (bool): si True, transforme ces variables pour les rendre 
        plus gaussiennes. 
      - name_testKaggle (str): nom du fichier .csv contenant le jeu de test Kaggle,
        situé à l'emplacement indiqué par PATH_DATA.
      - drop_requests_duplicates(bool): si True, supprime les requêtes qui ont 
        été faites en double.
      - adversarial_validation (bool): si True, réalise l'adversarial validation
        pour sélectionner les requêtes qui ressemblent le plus au jeu de test.
      - save_encoding (bool): si True, le target et frequency encoding est 
        sauvegardé dans le dossier indiqué par PATH_ENCODING. 
      - encode_all_same (bool): Si true, on fait du target encoding sur toutes les
        variables qualitatives. Sinon , on fait du target encoding seulement sur 
        les variables qualitatives contenues dans CAT_VAR et non dans CAT_VAR_FREQ.
        Les variables contenues dans CAT_VAR_FREQ seront frequency encodées.
     -  save_datasets(bool): si True, sauve dans PATH_DATA X_train,Y_train,
        X_test, Y_test. 
    
      Output:
      ------
      - X_train,Y_train,X_val,Y_val,X_test,Y_test (dataframe): jeu d'entraînement,
        validation et test avec les labels correspondants.
    """
    pricing_requests = retrieve_data(name_requests,name_featHotels)
    all_requests = pd.read_csv(PATH_DATA + "/all_our_requests_done.csv")
    pricing_requests = add_request_nb(all_requests, pricing_requests, save_csv = False)
  
    if drop_requests_duplicates:
        pricing_requests = drop_duplicate (pricing_requests, save_csv = False)
    
    pricing_requests = add_external_features(pricing_requests)
    pricing_requests = normalise_quanti(pricing_requests, price, stock)

    if adversarial_validation:
        X_test_name = pd.read_csv(PATH_DATA  + "/" + name_testKaggle)
        X_train_keep = main_adversarial_validation (pricing_requests,X_test_name,load_csv=False)
        pricing_requests = pricing_requests.loc[X_train_keep.index] #réduire pricing_requests aux données sélectionnées

    X_train,Y_train,X_val,Y_val,X_test,Y_test =  build_train_test(pricing_requests)

    if encoding == "targetFreq":
        X_train,X_val,X_test= target_freq_encoding(X_train,X_val,X_test,Y_train,encode_all_same, 
                      save_encoding)
    elif encoding == "oneHot":
        X_train,X_val,X_test = oneHot_encoding(X_train,X_val,X_test)
    else:
        print("Pas d'encoding des variables qualitatives")

    if save_datasets:
        X_train.to_csv(PATH_DATA + "/X_train.csv") 
        Y_train.to_csv(PATH_DATA + "/Y_train.csv")
        X_test.to_csv(PATH_DATA + "/X_test.csv")
        Y_test.to_csv(PATH_DATA + "/Y_test.csv")
        
    return X_train,Y_train,X_val,Y_val,X_test,Y_test



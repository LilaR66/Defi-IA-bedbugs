# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 21:21:21 2022

@author: Flavie
"""

"""
Ce fichier permet de génerer une soumission pour Kaggle. 
i.e. de predire les prix des hotels du tests set
"""

# Importation des libraries python
# ------------------------------
#----- Arguments 
import argparse
#----- Dataframe
import pandas as pd
import pickle

# Importation de nos libraries
# ------------------------------
from set_path import *
import data_preprocessing as DP

NUM_VAR = NUM_VAR + ['avatar_id'] 
INT_VAR_TEST = INT_VAR.copy()
INT_VAR_TEST.remove('price')

# Fonctions
# -----------------------------
def retrieve_data_test (name_test_set, name_featHotels):
  """
    Cette fonction permet de charger les fichiers .csv donnés en argument en
    tant que dataframe et d'associer le bon type aux données.

    Input:
    -----
    - name_test_set (str): nom du fichier .csv contenant le test set de Kaggle,
        situé à l'emplacement indiqué par PATH_DATA.
    - name_featHotels (str): nom du fichier .csv contenant les caractéristiques
        des hôtels, situé à l'emplacement indiqué par PATH_DATA.

    Output:
    ------
    - test_set (dataframe): dataframe contenant la concantéantion de 
        name_test_set et name_featHotels. 
        Le bon type a été associé à chaque variable.
  """
  # Récupération du fichier contentant l'ensemble des données issues des requêtes
  test_set = pd.read_csv(PATH_DATA + '/' + name_test_set)
  # Récupération du fichier contenant les features des hotels
  hotels = pd.read_csv(PATH_DATA + '/' + name_featHotels, index_col=['hotel_id', 'city']) 
  # Jointure des deux fichiers 
  test_set = test_set.join(hotels, on=['hotel_id', 'city'])
  # Convert to integer: 
  int_list = INT_VAR_TEST[0:3] #récupérer que les 3 premières (car request_number et request_nb n'existe pas encore)
  test_set[int_list] = test_set[int_list].astype(int) 
  # Convert to categorical: 
  cat_list = CAT_VAR
  for var in cat_list:
    test_set[var] = pd.Categorical(test_set[var],ordered=False)
  return test_set
    
def formating_test_set (test_set, load_csv = False): 
  '''Modifie le test set pour lui ajouter les features nécéssaires aux prédictions par le modèle.
	Liste des features à ajouter : 
	request_nb le nombre de fois q un avatar requête la base de données
	Depuis hotel_features : group, brand, parking, pool, children_policy
	Depuis nos recherches : pib, price_m2, nb_tourists, nb_hab_km2

	Input:
	-----
    - test_set (dataframe) contenant les features hotel et les bons types
    -(supprime insérer dans le pipeline) col_to_delete list de str des noms des colonnes à ne pas garder dans le modèle
	- load_csv = True si enregistrement du dataframe nettoyé

	Output:
	------
	- test_set avec les ajouts'''
	
  # ajout de la colonne request_nb 
  # --------------------------------------------
  test_test_set = test_set.sort_values(by = ['avatar_id','order_requests'])
  test_test_set.loc.__setitem__((0, ('request_nb')), 1)
  c = 1
  for i in range(1, len(test_test_set)):
  # cas avatar précédent = avatar présent et requete !=
    if (test_test_set.avatar_id.loc[i] == test_test_set.avatar_id.loc[i-1] and test_test_set.order_requests.loc[i] != test_test_set.order_requests.loc[i-1]) :
        c = c + 1
        test_test_set.loc.__setitem__((i, ('request_nb')), int(c))
	# cas avatar présent = avatar précédent et requete ==
    elif (test_test_set.avatar_id.loc[i] == test_test_set.avatar_id.loc[i-1]) :
        test_test_set.loc.__setitem__((i, ('request_nb')), int(c))
	# cas nouvel avatar
    else : 
        c = 1
        test_test_set.loc.__setitem__((i, ('request_nb')), int(c))
	# réordonnement
  test_test_set = test_test_set.sort_values(by='index')  

  # Ajout des features additionnelles
  #########################################################################
  test_test_set = DP.add_external_features(test_test_set)

  # Mise en forme 
  # --------------------------------------------------
  # on met les colonnes dans le même ordre que jeu obtenu par requetes et utilisé pour l'apprentissage
  # nb : order_requests = request_number : juste pour avoir les mêmes noms.
  test_test_set = test_test_set.reindex(columns=['city','language','brand','group','mobile','parking', 'pool', 'children_policy', 'hotel_id', 'stock', 'date', 'request_nb', 'pib', 'nb_tourists', 'nb_hab_km2', 'price_m2', 'order_requests', 'avatar_id'])
  test_test_set.rename(columns = {'order_requests':"request_number"}, inplace = True)
  
  '''for col in col_to_delete :
    # Supression des colonnes non utilisées par le modèle
    test_test_set.drop(col,axis=1,inplace=True)'''
  # enregistrement si load_csv : 
  if load_csv : 
    test_test_set.to_csv(PATH_DATA  + "/" + 'test_set_addedFeatures.csv', index=False)
	
  test_set = test_test_set

  return test_set

def load_model(model_name):
  """
  Charge le modèle et son piepeline enregisté dans le dossier modèle. 

  Input:
  -------
  - model_name (str) nom du modèle
    
  Output:
  ------
  - model : le modèle chargé
  """
  model = pickle.load(open(PATH_MODELS +"/"+ model_name, 'rb'))
  print('modèle' + model_name + 'chargé avec succès')
  return model

def get_transform_target(test_set, encode_all_same=False): 
  """
  Cette fonction permet d'effectuer le target encoding sur le jeu 
  de test gâces aux dictionnaires du dossiers dictionnaires.


  Input:
  -------
  - test_set (dataframe): jeu de test.
  - encode_all_same (bool) Si true, on fait du target encoding sur toutes les
    variables qualitatives. Sinon , on fait du target encoding seulement sur 
    les variables qualitatives contenues dans CAT_VAR et non dans CAT_VAR_FREQ. 
  Output:
  ------
  - test_set_encoding (dataframe) avec target encoding. 
  """

  frequency_by_hotel_pool = pickle.load(open( PATH_ENCODING + "/" + "pool", 'rb')) # dictionnaire
  frequency_by_hotel_child = pickle.load(open(  PATH_ENCODING + "/" + "children_policy", 'rb')) #dictionnaire
  frequency_by_hotel_parking = pickle.load(open(  PATH_ENCODING + "/" + "parking", 'rb')) #dictionnaire
  frequency_by_hotel_mobile = pickle.load(open(  PATH_ENCODING + "/" + "mobile", 'rb'))  #dictionnaire
  target_encoding = pickle.load(open( PATH_ENCODING + "/" + 'target_encoding', 'rb'))
  

  if not encode_all_same:
    test_set['pool'] = test_set['pool'].apply(lambda x : frequency_by_hotel_pool[x])
    test_set['children_policy'] = test_set['children_policy'].apply(lambda x : frequency_by_hotel_child[x])
    test_set['mobile'] = test_set['mobile'].apply(lambda x : frequency_by_hotel_mobile[x])
    test_set['parking'] = test_set['parking'].apply(lambda x : frequency_by_hotel_parking[x])
  
    float_list = ["parking", "pool", "mobile", "children_policy"] 
    test_set[float_list] = test_set[float_list].astype(float) 

  #si on passe pas dans le if on fait direct ça  
  test_set_encoding = target_encoding.transform(test_set)
  print('target encoding effecué avec succès')
  return test_set_encoding

def get_transform_one_hot(test_set):
  """
  Cette fonction permet de one-hot encoder le jeu 
  de test. 
  Comme hotel_id a trop de modalité, on le considère comme 
  une variable quantitative. Il n'est donc pas  one-hot encodé. 

  Input:
  -------
  - test_set (dataframe): jeu de test.
    
  Output:
  ------
  - test_set_oh (dataframe) avec one hote encoding. 
  """
  # Comme hotel_id a trop de modalité, on le considère comme 
  # une variable quantitative 
  num_var = INT_VAR_TEST + ['hotel_id']
  cat_var = CAT_VAR.copy()
  cat_var.remove('hotel_id')
  dfs = test_set
  X_num = dfs[num_var]
  X_cat = pd.get_dummies(dfs[cat_var],drop_first = True)
  dfs = pd.concat([X_num,X_cat],axis=1)
  test_set_oh = dfs
  print('One hot encoding effecué avec succès')
  return test_set_oh


def predict(model, X_eval, name_sub_csv, load_csv): 
  """
  Cette fonction génère le dataframe de prédictions.

  Input:
  -------
  - model : modèle chargé avec un sav
  - X_eval (dataframe ) : test_set mis en forme et one hot encodé ou target encodé
  - name_sub_csv (str) : nom d'enregistrement de la soumission en csv
  - load_csv (bool) enregistrement ou non en csv
    
  Output:
  ------
  - predict (dataframe) dataframe de prédiction : 2 colonnes : index et price 
  """
  idx = X_eval.index
  pred = model.predict(X_eval)
  predict = pd.DataFrame(list(zip(idx, pred**2)), columns=['index', 'price']) #### Mettre la prediction au carré car tranfor sqrt dans la transformation des données
  if load_csv : 
    predict.to_csv( PATH_DATA + "/" + name_sub_csv, index=False)
    print('prédictions téléchargées')
  return predict

if __name__ == "__main__":

  '''
  - name_test_set (str): nom du fichier .csv contenant le test set de Kaggle, situé à l'emplacement indiqué par PATH_DATA
  - name_featHotels (str): nom du fichier .csv contenant les caractéristiques
      des hôtels, situé à l'emplacement indiqué par PATH_DATA.
  - encoding (str): peut prendre les valeurs 0 pour effectuer
    du target et frequency encoding ou 1 pour faire du one-hot
    encoding des variables qualitatives.
  - name model (str): nom du model dans le dossier PATH_MODEL
  - name soumission (str) : nom d'enregistrement de la soumission en csv
  - load_csv : bool (enregistrement ou non de la soumission)
  '''
  #choix du type d'encoding, du modèle
  parser = argparse.ArgumentParser()
  parser.add_argument('--name_test_set', type=str, default='test_set.csv', help='name of the test set kaggle')
  parser.add_argument('--encoding', type=int, default=0, help='0: target+frequency encoding, 1: one hot encoding')
  parser.add_argument('--name_featHotels', type=str, default='features_hotels.csv', help='name of the features hotels')
  parser.add_argument('--name_model', type=str, default="xgboost_model_sav.sav", help="name of the file .sav")
  parser.add_argument('--name_soumission', type=str, default="soumission.csv", help="name of the file .csv")
  parser.add_argument('--load_csv', type = str, default= True, help = " Tru if you want to load result .csv")
  args = parser.parse_args()

  encoding = args.encoding 
  name_test_set = args.name_test_set
  name_featHotels = args.name_featHotels
  name_model = args.name_model
  name_soumission = args.name_soumission
  load_csv = args.load_csv
  #------------------------------------------------------------------------------------------------------
  # recupération du test_set
  test_set = retrieve_data_test(name_test_set, name_featHotels)
  #formatage et ajout features extérieurs
  test_set = formating_test_set(test_set, False)
  # normalisation de la variable stock
  test_set = DP.normalise_quanti(test_set, False, True)
  test_set = test_set.drop("avatar_id", axis=1)
  # Chargement du pipeline du modèle
  #test_set = test_set.drop("avatar_id", axis=1) #suppression de la colonne avatar_id (dans fichier model.py désormais )
  if encoding ==  0 : #'targetFreq'
    test_set = get_transform_target(test_set)
  elif encoding == 1: #'oneHot' :
    test_set = get_transform_one_hot(test_set)
  else : 
    print ('paramètre encoding non valable')
  parameters =  pickle.load(open(PATH_MODELS +"/"+ "parameters_" + name_model , 'rb'))
  for i in parameters :
    test_set = test_set.drop(i, axis=1)
  model = load_model(name_model)
  pred = predict(model, test_set, name_soumission, load_csv)
  print('Prédictions : ', pred)
		

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 22:06:46 2022

@author: Lila
"""


"""
Ce fichier permet de sélectionner les données du jeu d'entraînement (requêtes)
les plus pertinentes i.e. celles qui s'approchent le plus de la distribution 
du jeu de test Kaggle.
"""

# Importation des libraries python
# ------------------------------
import pandas as pd
from catboost import Pool, CatBoostClassifier
from catboost.utils import get_roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
plt.rcParams.update({'font.size': 12})


# Importation de nos libraries
# ------------------------------
from set_path import PATH_DATA, NUM_VAR, CAT_VAR, DATES

NUM_VAR = NUM_VAR + ['avatar_id'] ##########

# Fonctions
# -----------------------------
def data_preprocessing_adv(X_train_name,X_test_name,load_csv:bool):
    """
    Cette fontion met en forme les données pour l'adversarial validation.
    
    Input:
    -----
    - X_train_name(str or dataframe): si load_csv = True, nom du fichier 
      contenant les requêtes effectuées situé dans le dossier indiqué par 
      PATH_DATA. Si load_csv = False, X_train_name est un dataframe qui a déjà
      été chargé.  
    - X_test_name(str or dataframe): si load_csv = True, nom du fichier 
      contenant le jeu de test Kaggle situé dans le dossier indiqué par 
      PATH_DATA. Si load_csv = False, X_train_name est un dataframe qui a déjà
      été chargé. 
    - load_csv (bool): si True, alors X_train_name et X_test_name sont 
      le nom de fichiers .csv à charger, sinon, ce sont des dataframes déjà
      chargées.
      
    Output:
    ------
    - X_train (dataframe): jeu d'entraînement pour adversarial validation 
     =  dataframe des requêtes effectuées.
    - X_test (dataframe): jeu de test pour adversarial validation
     = dataframe du jeu de test Kaggle.
    - all_cols(list): nom des colonnes de dataframes (variables explicatives 
      + variable réponse)
    - features(list): nom des variables explicatives 
    - target(str): target variable
    """


    #--- read in the data
    if load_csv:
      X_train = pd.read_csv(PATH_DATA  + "/" + X_train_name)
      X_test  = pd.read_csv(PATH_DATA  + "/" + X_test_name)
    else:
      X_train = X_train_name.copy()
      X_test = X_train_name.copy()
    #--- drop target value (price) from the train set
    X_train = X_train.drop(['price'], axis=1)
    #--- shuffle train set
    X_train = X_train.sample(frac = 1)
    #--- from X_train, select only dates that are in the Kaggle test set
    X_train = X_train[X_train['date'].isin(DATES)]
    #--- Add columns 
    target = 'dataset_label'
    X_train[target] = 0 #add column to specify that data from X_train don't belong to Kaggle dataset
    X_test[target] = 1 #add column to specify that data from X_est belong to Kaggle dataset
    
    
    features = CAT_VAR + NUM_VAR  
    all_cols = features + [target]   
    
    return X_train,X_test,all_cols,features,target

def create_adversarial_data(df_train, df_test, cols, N_val=5000):
    """
    Create train and test sets for adversarial validation.
    
    Input:
    -----
    - df_train(dataframe): dataframe des requêtes effectuées.
    - df_test(dataframe): dataframe du jeu de test Kaggle.
    - cols (list): nom des colonnes de dataframes
    
    Output:
    ------
    - adversarial_train, adversarial_val: df_train et df_test mis 
    sous le bon format pour effectuer l'adversarial validation avec 
    CatBoostClassifier.    
    """
    df_master = pd.concat([df_train[cols], df_test[cols]], axis=0)
    adversarial_val = df_master.sample(N_val, replace=False)
    adversarial_train = df_master[~df_master.index.isin(adversarial_val.index)]
    return adversarial_train, adversarial_val



def train_CatBoost(features,cat_cols,adversarial_train,adversarial_test,target,plot_ROC = False):
    """
    Format data for CatBoostClassifier and train model CatBoostClassifier. 
    It plots the ROC curve if specified.
    
    Input:
    -----
    - features (list): all explanatory variables (categorical and numerical)
    - cat_cols (list): categorical explanatory variables
    - adversarial_train (dataframe): train dataset for adversarial validation
    - adversarial_test (dataframe): test dataset for adversarial validation
    - target(str): target variable
    
    Output:
    ------
    - model: trained CatBoostClassifier model
    - train_data: adversarial_train formatted for CatBoostClassifier
    - holdout_data: holdout_data formatted for CatBoostClassifier
    """
    # Formatting adversarial training and testing data for catBoost
    train_data = Pool(
        data=adversarial_train[features],
        label=adversarial_train[target],
        cat_features=cat_cols
    )

    holdout_data = Pool(
        data=adversarial_test[features],
        label=adversarial_test[target],
        cat_features=cat_cols
    )

    # fit model CatBoostClassifier
    params = {
        'iterations': 100,
        'eval_metric': 'AUC',
        'od_type': 'Iter',
        'od_wait': 50,
    }
    params.update({"ignored_features": ['avatar_id', 'request_number','hotel_id']})
    model = CatBoostClassifier(**params)
    _ = model.fit(train_data, eval_set=holdout_data,verbose=False)

    # plot ROC curve
    if plot_ROC:
        plt.figure(figsize= (8,8))
        get_roc_curve(model, 
                      train_data,
                      thread_count=-1,
                      plot=True)
        plt.show()
    return model, train_data, holdout_data



def plot_conf_matrix(label,prediction):
    """
    Plots the confusion matrix.
    
    Input:
    -----
    - label: true label
    - prediction: label predicted by a model
    
    Output:
    ------
    - plot of the confusion matrix
    """
    
    conf = confusion_matrix(label, prediction)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = conf, display_labels = [0, 1])

    cm_display.plot()
    plt.show()


def main_adversarial_validation (X_train_name,X_test_name,load_csv:bool,n_split=1500):   
    """
    Ce code performe l'adversarial validation sur n_split morceaux
    du dataset d'entraînement (requêtes) afin de sélectionner uniquement
    les données s'approchant le plus du set de test Kaggle.
    
    Input:
    -----
    - X_train_name(str or dataframe): si load_csv = True, nom du fichier 
      contenant les requêtes effectuées situé dans le dossier indiqué par 
      PATH_DATA. Si load_csv = False, X_train_name est un dataframe qui a déjà
      été chargé.  
    - X_test_name(str or dataframe): si load_csv = True, nom du fichier 
      contenant le jeu de test Kaggle situé dans le dossier indiqué par 
      PATH_DATA. Si load_csv = False, X_train_name est un dataframe qui a déjà
      été chargé. 
    - load_csv (bool): si True, alors X_train_name et X_test_name sont 
      le nom de fichiers .csv à charger, sinon, ce sont des dataframes déjà
      chargées.
    - n_split(int):  nombre de splits pour le jeu de train.
      
    Output:
    ------
    - X_train_keep(dataframe): requêtes sélectionnés de sorte qu'elles 
      s'approchent le plus du jeu de test Kaggle. 
    """
    
    X_train,X_test,all_cols,features,target = data_preprocessing_adv(X_train_name,X_test_name,load_csv)
    
    # liste de dataframes où chaque dataframe est 
    # un morceau de X_train qui s'approche au mieux du set de test kaggle
    X_train_keep_l = [] 
    
    for i in range(0,len(X_train), n_split):
        #--- X_train_i est un morceau de X_train
        X_train_i = X_train[i:n_split+i]
        #--- Entraîner un modèle de adversarial validation sur X_train_i
        adversarial_train, adversarial_test = create_adversarial_data(X_train_i, X_test, all_cols)
        model,train_data, holdout_data = train_CatBoost(features,CAT_VAR,adversarial_train,adversarial_test,target)
        #--- Définir le subset à garder 
        # ajout d'une colonne prédiction
        adversarial_train['is_in_test'] = model.predict(train_data)  
        # on ne garde que les données de train (requêtes)
        adversarial_train_keep = adversarial_train[adversarial_train['dataset_label'] == 0] 
        # on ne garder parmi les requêtes, que celles qui ont été prédites comme appartenant au set de test
        adversarial_train_keep = adversarial_train_keep[adversarial_train_keep['is_in_test'] == 1] 
        #--- Ajouter le subset à garder à la liste 
        X_train_keep_l.append(adversarial_train_keep)
    
    # concaténation de tous le dataframes qu'on garde
    X_train_keep = pd.concat(X_train_keep_l,axis=0)
    # on enlève les colonnes ajoutées inutiles à présent
    X_train_keep = X_train_keep.drop(["dataset_label","is_in_test"],axis = 1)
    
    return X_train_keep






    
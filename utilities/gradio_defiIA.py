import gradio as gr
import pandas as pd 
import numpy as np
#----- Machine Learning Preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
#save model 
import os
from set_path import *
import pickle 
import argparse #arguments à passer avec la ligne de commande :)
from math import *
import catboost as cat
import xgboost as xgb
import category_encoders 
import warnings
warnings.filterwarnings('ignore')
#pour l'appeler : python gradio_defiIA.py --model_name "2000_XGboost_target_encoding_12-11.sav"
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default = "X_gboost_tuned_model.sav", help='name of the file corresponding to the modelfitted')
parser.add_argument('--var_kept', type=int, default = 2, help='variables we have decided to kept')

args = parser.parse_args()
model = args.model_name
variables_kept=args.var_kept
model_folder=PATH_MODELS + "/"
dic_folder=PATH_ENCODING + "/"

loaded_model = pickle.load(open(model_folder+model, 'rb'))

###########################PIB###############################
PIB = {"amsterdam": 990583,
     "copenhagen": 386724,
     "paris": 2778090,
     "sofia": 85008,
     "vienna": 468046,
     "rome": 1996934,
     "madrid":1389927,
     "vilnius":68031,
     "valletta":17156}
     
#########################PRICE_M^2###############################
price_m2 = {"amsterdam": 4610,
            "copenhagen": 5236,
            "paris": 9160,
            "sofia": 1095,
            "vienna": 6550,
            "rome": 3044,
            "madrid": 3540,
            "vilnius": 1469,
            "valletta": 3600}
            
########################NB_TOURISTS###############################
nb_tourists = {"amsterdam": 7265, 
                "copenhagen": 15595 ,
                "paris": 117109,
                "sofia": 4973,
                "vienna": 15091,
                "rome": 38419,
                "madrid": 36410,
                "vilnius": 2284,
                "valletta": 718}
                
######################NB_HAB_KM2#################################
nb_hab_km2 = {"amsterdam": 3530, 
            "copenhagen": 7064,
            "paris": 20545,
            "sofia": 7354,
            "vienna": 4607,
            "rome": 2213,
            "madrid": 5437,
            "vilnius": 1432,
            "valletta": 8344}
            
            
def predict_price(stock_g,city_g,date_g,language_g,mobile_g, request_nb_g,group_g,brand_g,parking_g,pool_g,children_policy_g):
    
    if children_policy_g=="interdit aux mineurs":# récupération variable children_policy
        children_policy=2
    elif children_policy_g=="interdits aux moins de 12 ans":
        children_policy=1
    else :
        children_policy=0
        
    parking = 1 if parking_g else 0 #récupération var parking et transformation en int 
    pool = 1 if pool_g else 0  #récupération var piscine et transformation en int 
    mobile=1 if mobile_g else 0 #récupération var mobile et transformation en int 
        
    ar = np.array([[city_g, language_g, brand_g, group_g, mobile, parking, pool, children_policy, 50, stock_g,date_g,request_nb_g,0,0, 0,0, 1]])#création d'un np array avec les différentes valeurs des différentes variables
    print("voici ar", ar)
    df = pd.DataFrame(ar, index = ['a1'], columns = ["city", "language","brand", "group", "mobile", "parking","pool","children_policy","hotel_id", "stock", "date", "request_nb","pib", "nb_tourists", "nb_hab_km2", "price_m2", "request_number"] ) #création du dataframe avec les bons noms de colonnes 
    int_list = ["date","stock", "request_nb", "pib", "price_m2", "nb_tourists", "nb_hab_km2", "children_policy", "pool", "mobile", "parking"] #variables quantitatives 
    df[int_list] = df[int_list].astype(int) #transformations en int 
    
    #--- Convert to categorical: 
    df["city"] = pd.Categorical(df["city"],ordered=False)
    df["language"] = pd.Categorical(df["language"],ordered=False)
    #df["mobile"] = pd.Categorical(df["mobile"],ordered=False)
    #df["parking"] = pd.Categorical(df["parking"],ordered=False)
    #df["pool"] = pd.Categorical(df["pool"],ordered=False)
    df["group"] = pd.Categorical(df["group"],ordered=False)
    df["brand"] = pd.Categorical(df["brand"],ordered=False)
    df["stock"]=df["stock"].map(lambda x: log(x+1))
    print("city_g", str(city_g))
    df["price_m2"]=price_m2[str(city_g)]
    df["nb_hab_km2"]=nb_hab_km2[str(city_g)]
    df["pib"]=PIB[str(city_g)]
    df["nb_tourists"]=nb_tourists[str(city_g)]
    #-------------------------------------------------
    
    frequency_by_hotel_pool = pickle.load(open( dic_folder +"pool", 'rb')) #on peut ensuite le recharger 
    frequency_by_hotel_child = pickle.load(open( dic_folder + "children_policy", 'rb')) #dictionnaire
    frequency_by_hotel_parking = pickle.load(open( dic_folder + "parking", 'rb')) #dictionnaire
    frequency_by_hotel_mobile = pickle.load(open( dic_folder + "mobile", 'rb'))  #dictionnaire

    df['children_policy'] = df['children_policy'].apply(lambda x : frequency_by_hotel_child[x]) #frequence encoding de children policy
    
    df['pool'] = df['pool'].apply(lambda x : frequency_by_hotel_pool[x]) #frequence encoding de pool
    df['mobile'] = df['mobile'].apply(lambda x : frequency_by_hotel_mobile[x]) #frequence encoding de mobile
   
    df['parking'] = df['parking'].apply(lambda x : frequency_by_hotel_parking[x]) #frequence encoding de parking
    
    print("encodage fréquentiel de mobile, children_policy, parking et pool faits")
    
    #--- Convert to float: 
    float_list = ["parking", "pool", "mobile", "children_policy"] 
    df[float_list] = df[float_list].astype(float) 
    target_encoding=pickle.load(open( dic_folder + "target_encoding", 'rb')) 
    df = target_encoding.transform(df) #target encoding appliqué à df 
    # Liste des variables categorical:
    df.drop('hotel_id', axis=1, inplace = True)
    df.drop("request_number", axis=1, inplace = True)
    print(loaded_model.get_params([loaded_model]))
    return "la chambre coutera " + str( round((loaded_model.predict(df)[0])**2,2) ) + " €" #prédiction avec le modèle (on met au carré car on avait transformé avec une racine carrée 


if __name__=='__main__':
   
    
    
    gr.Interface(fn=predict_price, 
                inputs=[
                # gr.Slider(minimum=0, maximum=998, value=1, step=1, label="hotel_id"), #curseur entre 0 et 998 de 1 en 1 
                gr.Slider(minimum=0, maximum=199, value=1, step=1, label="stock"), #curseur de 1 et 1 
                gr.Dropdown(['amsterdam', 'copenhagen', 'madrid','paris', 'rome', 'sofia', 'valletta', 'vienna', 'vilnius'], label="ville"),#liste déroulante 
                gr.Slider(minimum=0,maximum=44, value=1, step=1, label="date"),
                gr.Dropdown(['austrian', 'belgium', 'bulgarian', 'croatian', 'cypriot', 'czech', 'danish', 'dutch', 'estonian', 'finnish', 'french', 'german', 'greek', 'hungarian', 'irish', 'italian', 'latvian', 'lithuanian', 'luxembourgish', 'maltese', 'polish', 'portuguese', 'romanian', 'slovakian', 'slovene', 'spanish','swedish'], label="langage"),
                gr.Checkbox(label="mobile?"), #case à cocher 
                gr.Slider(minimum=1,maximum=4, value=1, step=1, label="nombre de fois que vous avez fait la requête"), #curseur de 1 et 1 
                #gr.Textbox(value=1, label="request_number"), #case à remplir
                gr.Dropdown(['Accar Hotels', 'Boss Western', 'Chillton Worldwide', 'Independant',
 'Morriott International', 'Yin Yang'], label="group"),
                gr.Dropdown(['8 Premium', 'Ardisson', 'Boss Western' ,'Chill Garden Inn', 'Corlton',
 'CourtYord', 'Ibas', 'Independant', 'J.Halliday Inn', 'Marcure' 'Morriot',
 'Navatel', 'Quadrupletree', 'Royal Lotus', 'Safitel', 'Tripletree'], label="brand"),
                gr.Checkbox(label="Parking?"),
                gr.Checkbox(label="Piscine?"),
                gr.Radio(["interdit aux mineurs", "interdits aux moins de 12 ans", "pas de restriction"], label="Politique enfant" )], 
                outputs="text", #sortie de l'application (ici un int)
                live=True,
                description="Prédit le prix d'une nuit à l'hotel donnée par l'application de voyage",
                css="div {background-image: url('file=https://www.punaises-expert.com/wp-content/uploads/2020/04/une-seule-punaise-de-lit.jpg')}"
                ).launch(debug=True, share=True);

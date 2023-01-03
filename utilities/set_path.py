# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 21:44:00 2022

@author: Lila
"""

'''
This code is for indicating the path where to load the data, 
the models weigths and other global variables. 
In this way you only need to change the variables here, and if you add 
"from set_path import PATH,DATA_PATH" at the beginning of all your other 
files, they will be referenced with the same path 
'''

# Paths 
PATH = '/content/drive/My Drive/Defi-IA-bedbugs' #ou si docker PATH=/mnt 
PATH_CODE = PATH + '/utilities' #path to code
PATH_DATA = PATH + '/data' #path to data
PATH_MODELS = PATH + '/models' #path to trained models
PATH_ENCODING = PATH + '/dictionary'

# Global variables
NUM_VAR = ["stock", "date", "request_nb","pib","nb_tourists","nb_hab_km2","price_m2","request_number"] #list of quantitatives variables 
CAT_VAR = ["city", "language", "brand", "group","mobile","parking","pool","children_policy","hotel_id"] #list of qualitatives variables
INT_VAR = ["date","avatar_id","stock","price", "request_nb","request_number"] #list of integer variables 
DATES = [0,1,2,3,4,5,6,15,16,17,18,19,20,21,34,35,36,37] #list of dates of interest 
CAT_VAR_FREQ = ['pool','children_policy','mobile','parking'] #list of variables to frequency encode.
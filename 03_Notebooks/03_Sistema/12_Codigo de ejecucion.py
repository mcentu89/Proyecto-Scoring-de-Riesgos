#!/usr/bin/env python
# coding: utf-8

# # CODIGO DE EJECUCION

# *NOTA: Para poder usar este código de ejecución hay que lanzarlo desde exactamente el mismo entorno en el que fue creado.*
# 
# *Se puede instalar ese entorno en la nueva máquina usando el environment.yml que creamos en el set up del proyecto*
# 
# *Copiar el proyecto1.yml al directorio y en el terminal o anaconda prompt ejecutar:*
# 
# conda env create --file proyecto1.yml --name proyecto1

# In[17]:


#1 IMPORTACION

import numpy as np
import pandas as pd
import cloudpickle

#Automcompletar rápido
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

from janitor import clean_names

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

#2 CARGA DATOS

ruta_proyecto = r'C:\Users\mcent\OneDrive\Escritorio\PROYECTOS ML\SCORING_DE_RIESGOS'
nombre_fichero_datos = 'prestamos.csv'
ruta_completa = ruta_proyecto + '/02_Datos/01_Originales/' + nombre_fichero_datos
df = pd.read_csv(ruta_completa,index_col=0)

#3 VARIABLES Y REGISTROS FINALES
                     
variables_finales = ['antiguedad_empleo',
'estado',                     
'dti',
'finalidad',
'imp_cuota',
'imp_amortizado',
'imp_recuperado',
'ingresos',
'ingresos_verificados',
'num_cancelaciones_12meses',
'num_cuotas',
'num_derogatorios',
'num_hipotecas',
'num_lineas_credito',
'porc_tarjetas_75p',
'porc_uso_revolving',
'principal',
'rating',
'tipo_interes',
'vivienda']

df = clean_names(df)
df = df[variables_finales]

a_eliminar = ['porc_tarjetas_75p','porc_uso_revolving','dti','num_lineas_credito','num_derogatorios']

def eliminar_registros(temp, a_eliminar):
    for variable in a_eliminar:
        indices = temp.loc[df[variable].isna()].index
        temp.drop(indices, inplace= True)

eliminar_registros(df, a_eliminar)

a_dejar = df.loc[~((df.ingresos < 12000) | (df.ingresos > 300000))].index
df = df.loc[a_dejar]

#4 FUNCIONES SOPORTE

def calidad_datos(temp):
    
    temp = temp.astype({'num_hipotecas':'Int64','num_lineas_credito':'Int64','num_cancelaciones_12meses':'Int64',
                'num_derogatorios':'Int64'})
    
    
    def imputar_moda(variable):
        return(variable.fillna(variable.mode()[0]))
    temp['antiguedad_empleo'] = imputar_moda(temp.antiguedad_empleo)
    
    
    temp['finalidad'] = temp.finalidad.replace({'house':'other','renewable_energy':'other'})
    temp['vivienda'] = temp.vivienda.replace({'NONE':'MORTGAGE','ANY':'MORTGAGE','OTHER':'MORTGAGE'})
    temp['estado'] = temp.estado.replace({'Default':'Charged Off'})
    
    
    minimo = 0
    maximo = 100           
    temp['dti'] = temp['dti'].clip(minimo,maximo)
    temp['porc_uso_revolving'] = temp['porc_uso_revolving'].clip(minimo,maximo)
    temp['porc_tarjetas_75p'] = temp['porc_tarjetas_75p'].clip(minimo,maximo)                                                           
    
    return(temp)

def creacion_pd(df):
    temp = df.copy()
    
    temp['pendiente'] = temp['principal'] - temp['imp_amortizado']
    temp['pd'] = np.where(temp.estado == 'Charged Off', 1, 0)
    
    #Eliminamos variables que ya no usaremos
    temp.drop(columns=['imp_recuperado','imp_amortizado','pendiente','estado'], inplace = True)
    
    temp_x = temp.iloc[:,:-1]
    temp_y = temp.iloc[:,-1]
    
    return (temp_x, temp_y)

def creacion_ead(df):
    temp = df.copy()
    
    temp['ead'] = (1- temp.imp_amortizado/temp.principal)
    
    #Eliminamos variables que ya no usaremos
    temp.drop(columns=['imp_recuperado','imp_amortizado','estado'], inplace = True)
    
    temp_x = temp.iloc[:,:-1]
    temp_y = temp.iloc[:,-1]
    
    return (temp_x, temp_y)

def creacion_lg(df):
    temp = df.copy()
    
    temp['pendiente'] = temp.principal - temp.imp_amortizado
    temp['lg'] = (1 - temp.imp_recuperado/temp.pendiente)
    temp['lg'].fillna(0,inplace=True)
    
    #Eliminamos variables que ya no usaremos
    temp.drop(columns=['imp_recuperado','imp_amortizado','pendiente','estado'], inplace = True)
    
    temp_x = temp.iloc[:,:-1]
    temp_y = temp.iloc[:,-1]
    
    return (temp_x, temp_y)

#5 CALIDAD Y CREACION DE VARIABLES

x_pd, y_pd = creacion_pd(calidad_datos(df))
x_ead, y_ead = creacion_ead(calidad_datos(df))
x_lg, y_lg = creacion_lg(calidad_datos(df))

#6 CARGA DE PIPES Y EJECUCION
nombre_pipe_ejecucion_pd = 'pipe_ejecucion_pd.pickle'
nombre_pipe_ejecucion_ead = 'pipe_ejecucion_ead.pickle'
nombre_pipe_ejecucion_lg = 'pipe_ejecucion_lg.pickle'

ruta_pipe_ejecucion_pd = ruta_proyecto + '/04_Modelos/' + nombre_pipe_ejecucion_pd
ruta_pipe_ejecucion_ead = ruta_proyecto + '/04_Modelos/' + nombre_pipe_ejecucion_ead
ruta_pipe_ejecucion_lg = ruta_proyecto + '/04_Modelos/' + nombre_pipe_ejecucion_lg

with open(ruta_pipe_ejecucion_pd, mode='rb') as file:
   pipe_ejecucion_pd = cloudpickle.load(file)
with open(ruta_pipe_ejecucion_ead, mode='rb') as file:
   pipe_ejecucion_ead = cloudpickle.load(file)
with open(ruta_pipe_ejecucion_lg, mode='rb') as file:
   pipe_ejecucion_lg = cloudpickle.load(file)

scoring_pd = pipe_ejecucion_pd.predict_proba(x_pd)[:, 1]
ead = pipe_ejecucion_ead.predict(x_ead)
lg = pipe_ejecucion_lg.predict(x_lg)      

#RESULTADO
principal = x_pd.principal
EL = pd.DataFrame({'principal':principal,
                   'pd':scoring_pd,
                    'ead':ead,
                    'lg':lg})
EL['perdida_esperada'] = round(EL.principal * EL.pd * EL.ead * EL.lg, 2)


# In[ ]:





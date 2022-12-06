#1 IMPORTACION

import numpy as np
import pandas as pd
import cloudpickle


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

#4 FUNCIONES SOPORTE

def calidad_datos(temp):

    temp = clean_names(temp)
    temp = temp.astype({'num_hipotecas':'Int64','num_lineas_credito':'Int64',
                'num_derogatorios':'Int64'})


    def imputar_moda(variable):
        return(variable.fillna(variable.mode()[0]))
    temp['antiguedad_empleo'] = imputar_moda(temp.antiguedad_empleo)


    temp['finalidad'] = temp.finalidad.replace({'house':'other','renewable_energy':'other'})
    temp['vivienda'] = temp.vivienda.replace({'NONE':'MORTGAGE','ANY':'MORTGAGE','OTHER':'MORTGAGE'})

    return (temp)

#5 CALIDAD Y CREACION DE VARIABLES

def ejecutar_modelos(df):
    x_pd = calidad_datos(df)
    x_ead = calidad_datos(df)
    x_lg = calidad_datos(df)

    #6 CARGA DE PIPES Y EJECUCION

    with open('pipe_ejecucion_pd.pickle', mode='rb') as file:
       pipe_ejecucion_pd = cloudpickle.load(file)

    with open('pipe_ejecucion_ead.pickle', mode='rb') as file:
       pipe_ejecucion_ead = cloudpickle.load(file)

    with open('pipe_ejecucion_lg.pickle', mode='rb') as file:
       pipe_ejecucion_lg = cloudpickle.load(file)

# EJECUCION

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

    return (EL)

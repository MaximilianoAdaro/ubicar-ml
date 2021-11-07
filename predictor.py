import joblib
import pandas as pd
import warnings
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def predictValue(propertyId):

    pipeline_LR_V1 = joblib.load('model/model.joblib')
    csv_path = f"/home/maxi/projects/ubicar/ubicar-api/src/main/resources/properties_csv/{propertyId}"

    num_cols = ['surface_total_in_m2', 'surface_covered_in_m2', 'rooms', 'dRailway', 'dIndustrialArea', 'dAirport',
                'dEducation', 'dFireStation', 'dHealthBuilding', 'dPenitentiary', 'dPort', 'dSecureBuilding',
                'dTrainStation', 'dUniversity', ]
    columnNames = ["surface_total_in_m2", "surface_covered_in_m2", "rooms", "state_name", "dRailway",
                   "dIndustrialArea", "dAirport", "dEducation", "dFireStation", "dHealthBuilding",
                   "dPenitentiary", "dPort", "dSecureBuilding", "dTrainStation", "dUniversity",
                   "property_type"]
    testingDF = pd.read_csv(csv_path, names=columnNames, error_bad_lines=False, header=0)
    for name in num_cols:
        testingDF[name] = pd.to_numeric(testingDF[name], errors='coerce')
    # predArray_RF_v1 = pipeline_RF_V1.predict(testingDF)[0]
    # predArray_RF_v3 = pipeline_RF_V3.predict(testingDF)[0]
    predArray_LR_v1 = pipeline_LR_V1.predict(testingDF)[0]

    return [predArray_LR_v1]

def printResults(regressionName, pricePredicted, errorEstimated):
    print(regressionName)
    print(f"Price: $ {int(pricePredicted)}")
    print(f"Price with error: $ {int(pricePredicted * errorEstimated)}")

def showRresults(predictionPrice):
    # print("-----------------------------------------------------------")
    # printResults("--- Random Forest v1 ---", predictionPrice[0], 1.176)

    print("-----------------------------------------------------------")
    printResults("--- Linear Regression v1 ---", predictionPrice[0], 0.8)

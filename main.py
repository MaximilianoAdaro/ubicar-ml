from predictor import showRresults
from extra import predict

if __name__ == '__main__':
    propertyId = "2d5a13e0-3243-446f-aa66-564b41e047bb.csv"
    predictionPrice = predict(propertyId)
    showRresults(predictionPrice)

from data import Data
from svm_train import SVM
from LSTM import LSTMClassifier

data = Data('InputDataCSV.csv', 'StockData.csv', 'ProcessedData.csv')
data.createData()

# perform SVM to calculate the prediction for next five days and print the accuracy
for day in range(1, 6):
    googleSvm = SVM('ProcessedData.csv', day, 'GOOGLE')
    googleSvm.trainAndTest()
    appleSvm = SVM('ProcessedData.csv', day, 'APPLE')
    appleSvm.trainAndTest()

# perform LSTM to calculate the prediction for next five days and print the accuracy
for day in range(1, 6):
    googleLstm = LSTMClassifier('GOOGLE', 'ProcessedData.csv', day)
    googleLstm.performLstm()
    appleLstm = LSTMClassifier('APPLE', 'ProcessedData.csv', day)
    appleLstm.performLstm()




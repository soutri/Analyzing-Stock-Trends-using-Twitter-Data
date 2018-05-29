import random
import numpy as np
import math
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import datetime
import csv
import pandas as pd


class SVM:
    def __init__(self, processesDataFilename, daysOffset, companyName):
        self.processesDataFilename = processesDataFilename
        self.daysOffset = daysOffset
        self.companyName = companyName
        self.dataForCompany = []
        self.tomorrow = 8
        
    def getDataForCompany(self):
        with open(self.processesDataFilename,'r') as csvFile:
            datareader = csv.reader(csvFile)
            for count, row in enumerate(datareader):
                if count == 0:
                    continue
                else:
                    if row[4] == self.companyName:
                        data = [0]*13
                        data[0] = float(row[2]) # sentiment
                        data[1] = float(row[3]) # impact factor
                        data[2] = float(row[7]) # t-1
                        data[3] = float(row[8]) # t-2
                        data[4] = float(row[9]) # t-3
                        data[5] = float(row[10]) # t-4
                        data[6] = float(row[11]) # t-5
                        data[7] = float(row[6]) # label
                        data[8] = float(row[12]) # t+1
                        data[9] = float(row[13]) # t+2
                        data[10] = float(row[14]) # t+3
                        data[11] = float(row[15]) # t+4
                        data[12] = float(row[16]) # t+5
                        self.dataForCompany.append(data)
    
    def getTrainAndTestData(self):
        self.getDataForCompany()
        random.seed(5)
        random.shuffle(self.dataForCompany)
        split = math.floor(0.8 * len(self.dataForCompany))
        return self.dataForCompany[:split], self.dataForCompany[split:]

    def trainAndTest(self):
        train, test = self.getTrainAndTestData()

        x_header = ['sentiment','impactFactor','t-1','t-2','t-3','t-4','t-5']
        # x_header = ['sentiment','impactFactor','t-1','t-2','t-3']
        y_header = ['Outcome']
        
        train_x = np.array([x[0:7] for x in train]) # take last 5 days trend into consideration
        # train_x = np.array([x[0:5] for x in train])
        train_x = train_x.reshape(-1,7)
        print(train_x.shape)

        test_x = np.array([x[0:7] for x in test])
        # test_x = np.array([x[0:5] for x in test])
        test_x = test_x.reshape(-1,7)
        print(test_x.shape)

        train_y = np.array([x[self.tomorrow] for x in train])
        train_y = train_y.flatten()
        print(train_y.shape)

        test_y = np.array([x[self.tomorrow] for x in test])
        test_y = test_y.flatten()
        print(test_y.shape)

        model = svm.SVC(kernel='rbf', C=1, cache_size=7000)
        scaling = MinMaxScaler(feature_range=(-1,1)).fit(train_x)
        train_x = scaling.transform(train_x)
        testscaling = MinMaxScaler(feature_range=(-1,1)).fit(test_x)
        test_x = testscaling.transform(test_x)
        model.fit(train_x, train_y)


        train_x = pd.DataFrame(train_x, None, x_header)
        test_x = pd.DataFrame(test_x, None, x_header)
        train_y = pd.DataFrame(train_y, None, y_header)
        test_y = pd.DataFrame(test_y, None, y_header)

        for day in range(1, self.daysOffset):
            self.tomorrow += 1
            label = 'day_{0}'.format(day)
            predicted_label = [label]

            predicted_train_x = model.predict(train_x)
            predicted_train_x_df = pd.DataFrame(predicted_train_x, None, predicted_label)
            train_x = train_x.join(predicted_train_x_df)

            predicted_test_x = model.predict(test_x)
            predicted_test_x_df = pd.DataFrame(predicted_test_x, None, predicted_label)
            test_x = test_x.join(predicted_test_x_df)

            train_y = np.array([x[self.tomorrow] for x in train])
            train_y = train_y.flatten()

            test_y = np.array([x[self.tomorrow] for x in test])
            test_y = test_y.flatten()

            model.fit(train_x.as_matrix(), train_y)

        score = model.score(test_x.as_matrix(),test_y)
        score = score *100.0
        print ('For Company {0} and Today+{1} days, SVM test accuracy is {2}'.format(self.companyName, self.daysOffset, score))


    
        
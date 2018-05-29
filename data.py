from fetchData import ReadData 
import csv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import pandas as pd
import time
import datetime
import re
import csv

class Data:
    def __init__(self, tweetFilename, stocksFilename, outputFilename):
        self.tweetFilename = tweetFilename
        self.stocksFilename = stocksFilename
        self.outputFilename = outputFilename
        self.stockData = []
        self.tweetData = []
        self.data = []

    def calculatePolarityAndImpactFactor(self):
        sentimentInternsityAnalyzer = SentimentIntensityAnalyzer()
        for count, row in enumerate(self.tweetData):
            if count == 0:
                row.append('polarity')
                row.append('impactFactor')
                continue
            else: 
                polarity = sentimentInternsityAnalyzer.polarity_scores(row[1])['compound']
                row.append(polarity)
                if row[2] == 0:
                    impactFactor = polarity * 1
                else:    
                    impactFactor = polarity * float(row[2]) # multiplied polarity of a tweet with number of followers for that user.
                row.append(impactFactor)
    
    def combineData(self):
        for count, row in enumerate(self.tweetData):
            if count == 0:
                continue
            else:
                data = [0]*17
                data[0] = row[3] # Date
                data[1] = row[1] # Tweet
                data[2] = row[9] # Sentiment analysis
                data[3] = row[10] # Impact factor
                data[4] = row[7] # Company
                data[5] = row[8] # Source
                for stock in self.stockData:
                    currentDate = row[3]
                    Bday1 = currentDate - datetime.timedelta(days=1)
                    Bday2 = currentDate - datetime.timedelta(days=2)
                    Bday3 = currentDate - datetime.timedelta(days=3)
                    Bday4 = currentDate - datetime.timedelta(days=4)
                    Bday5 = currentDate - datetime.timedelta(days=5)

                    Fday1 = currentDate + datetime.timedelta(days=1)
                    Fday2 = currentDate + datetime.timedelta(days=2)
                    Fday3 = currentDate + datetime.timedelta(days=3)
                    Fday4 = currentDate + datetime.timedelta(days=4)
                    Fday5 = currentDate + datetime.timedelta(days=5)
                    if currentDate.date() == stock[1].date() and row[7] == stock[7]:
                        openVal = stock[2] # Open
                        closeVal = stock[5] # Close
                        if openVal > closeVal:
                            data[6] = 0 # Current day trend
                        else:
                            data[6] = 1 # Current day trend
                    if Bday1.date() == stock[1].date() and row[7] == stock[7]:
                        openVal = stock[2] # Open
                        closeVal = stock[5] # Close
                        if openVal > closeVal:
                            data[7] = 0 # day1 trend
                        else:
                            data[7] = 1 # day1 trend
                    if Bday2.date() == stock[1].date() and row[7] == stock[7]:
                        openVal = stock[2] # Open
                        closeVal = stock[5] # Close
                        if openVal > closeVal:
                            data[8] = 0  # day2 trend
                        else:
                            data[8] = 1  # day2 trend
                    if Bday3.date() == stock[1].date() and row[7] == stock[7]:
                        openVal = stock[2] # Open
                        closeVal = stock[5] # Close
                        if openVal > closeVal:
                            data[9] = 0  # day3 trend
                        else:
                            data[9] = 1  # day3 trend
                    if Bday4.date() == stock[1].date() and row[7] == stock[7]:
                        openVal = stock[2] # Open
                        closeVal = stock[5] # Close
                        if openVal > closeVal:
                            data[10] = 0  # day4 trend
                        else:
                            data[10] = 1  # day4 trend
                    if Bday5.date() == stock[1].date() and row[7] == stock[7]:
                        openVal = stock[2] # Open
                        closeVal = stock[5] # Close
                        if openVal > closeVal:
                            data[11] = 0  # day5 trend
                        else:
                            data[11] = 1  # day5 trend
                    if Fday1.date() == stock[1].date() and row[7] == stock[7]:
                        openVal = stock[2] # Open
                        closeVal = stock[5] # Close
                        if openVal > closeVal:
                            data[12] = 0  # day5 trend
                        else:
                            data[12] = 1  # day5 trend
                    if Fday2.date() == stock[1].date() and row[7] == stock[7]:
                        openVal = stock[2] # Open
                        closeVal = stock[5] # Close
                        if openVal > closeVal:
                            data[13] = 0  # day5 trend
                        else:
                            data[13] = 1  # day5 trend
                    if Fday3.date() == stock[1].date() and row[7] == stock[7]:
                        openVal = stock[2] # Open
                        closeVal = stock[5] # Close
                        if openVal > closeVal:
                            data[14] = 0  # day5 trend
                        else:
                            data[14] = 1  # day5 trend
                    if Fday4.date() == stock[1].date() and row[7] == stock[7]:
                        openVal = stock[2] # Open
                        closeVal = stock[5] # Close
                        if openVal > closeVal:
                            data[15] = 0  # day5 trend
                        else:
                            data[15] = 1  # day5 trend
                    if Fday5.date() == stock[1].date() and row[7] == stock[7]:
                        openVal = stock[2] # Open
                        closeVal = stock[5] # Close
                        if openVal > closeVal:
                            data[16] = 0  # day5 trend
                        else:
                            data[16] = 1  # day5 trend
                self.data.append(data)
    
    def saveData(self):
        with open(self.outputFilename,'w', encoding='utf-8') as csvFile:
            inputdatawriter = csv.writer(csvFile)
            inputdatawriter.writerow(['Date','Tweet','Sentiment','ImpactFactor','Company','Source','Label','T-1','T-2','T-3','T-4','T-5','T+1','T+2','T+3','T+4','T+5'])
            for row in self.data:
                inputdatawriter.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13],row[14],row[15],row[16]])

    def createData(self):
        data = ReadData('InputDataCSV.csv','StockData.csv')
        self.stockData = data.Stock()
        self.tweetData = data.Tweet()
        self.calculatePolarityAndImpactFactor()
        self.combineData()
        self.saveData()






import csv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import pandas as pd
import time
import datetime
import re
import csv


class ReadData:
    def __init__(self, tweetDataFile, stockDataFile):
        self.tweetDataFile = tweetDataFile
        self.stockDataFile = stockDataFile
        self.tweetData = []
        self.stockData = []
    
    def Tweet(self):
        utcTime = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[ Z]$") # regex to check if the date is in UTC format
        with open(self.tweetDataFile,'r', encoding='utf-8') as csvFile:
            datareader = csv.reader(csvFile)
            for count, row in enumerate(datareader):
                if count == 0: # First row contains the headers
                    self.tweetData.append(row)
                    continue
                else:
                    val = str(row[3])
                    # convert the string date to datetime object
                    if utcTime.search(row[3]):
                        date = datetime.datetime.strptime(row[3],'%Y-%m-%dT%H:%M:%SZ')
                    else:
                        date = datetime.datetime.strptime(row[3], "%m/%d/%y %H:%M")
                    row[3] = date
                    self.tweetData.append(row)
        return self.tweetData

    def Stock(self):
        with open(self.stockDataFile,'r', encoding='utf-8') as csvFile:
            datareader = csv.reader(csvFile)
            for count, data in enumerate(datareader):
                if count == 0:
                    continue
                else:
                    # as the stock information is present in first cell seperated with tabs.
                    row = data[0].split("\t")
                    date = datetime.datetime.strptime(row[1], "%m/%d/%y")
                    row[1] = date
                    self.stockData.append(row)
        return self.stockData
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tweepy
import csv
import datetime
import re
import preprocessor as p
# import urllib.request
import json
import requests
from operator import itemgetter


API_KEY = ''
API_SECRET = ''
auth = tweepy.AppAuthHandler(API_KEY, API_SECRET)
api = tweepy.API(auth,wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

firstTimeTweetCheck = True
enddate = datetime.datetime(2018, 4, 15).date()
startdate = enddate - datetime.timedelta(days = 5)

keywords = {'$APPL':'Apple', '$GOOGL':'Google'}
keywords = {'$appl':'AppleSymbolSmall'}
keywords = {'$googl':'GoogleSymbolSmall'}
keywords = {'$GOOGL':'Google'}

for keyword in keywords.keys():
    filename = "InputData.csv".format(keywords[keyword], startdate, enddate)
    with open(filename, 'a') as csvfile:
        for periodMultiplier in range(6):
                inputdatawriter = csv.writer(csvfile)
                if firstTimeTweetCheck:
                    newestTweet = tweepy.Cursor(api.search, q=keyword, rpp=100).items(300)
                    tweets = []
                    tweets.extend(newestTweet)
                    print(keyword)
                    if len(tweets) == 0:
                        break
                    oldest = tweets[-1]
                    while oldest.created_at.date() > enddate:
                        tweets = []
                        newestTweet = tweepy.Cursor(api.search, q=keyword, rpp=100, max_id=oldest.id).items(300)
                        tweets.extend(newestTweet)
                        if len(tweets) == 0:
                            break
                        oldest = tweets[-1]
                # oldest = tweets[-1]
                while oldest.created_at.date() > startdate:
                    allusefultweets = []
                    usefultweets = tweepy.Cursor(api.search, q=keyword, rpp=100, max_id=oldest.id).items(300)
                    allusefultweets.extend(usefultweets)
                    print(len(allusefultweets))
                    for usefultweet in allusefultweets:
                        # tweet = re.sub('[^A-Za-z0-9!@#$%^&*()+=-]', '', usefultweet.text)
                        p.set_options(p.OPT.URL, p.OPT.EMOJI)
                        tweet = p.clean(usefultweet.text.encode("utf-8"))
                        inputdatawriter.writerow([usefultweet.author.name.encode("utf-8"), tweet.encode("utf-8"), usefultweet.author.friends_count, usefultweet.created_at, usefultweet.id, usefultweet.favorite_count, usefultweet.retweet_count])
                    if len(allusefultweets) == 0:
                        break
                    oldest = allusefultweets[-1]  
            enddate = enddate - datetime.timedelta(days = 5)
            startdate = enddate - datetime.timedelta(days = 5)
            firstTimeTweetCheck = False
        firstTimeTweetCheck = True

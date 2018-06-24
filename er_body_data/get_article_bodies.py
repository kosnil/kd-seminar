import eventregistry as ER
import datetime
import pandas as pd
import time
from eventregistry import *

er = ER.EventRegistry(apiKey="5ba73408-ea81-459b-abf4-6fedd8cb8ec6")  # dany
#er = ER.EventRegistry(apiKey = "5fed3642-762a-4abc-aabf-ac6213c1bcea")  #philipp
#er = ER.EventRegistry(apiKey = "7571801b-6710-4166-90cc-9c5352ddeedd")  #andi
#er = ER.EventRegistry(apiKey="1b673182-c9e4-4554-90cf-d082a0bd6b53") #  Hendrik?
analytics = ER.Analytics(er)

# DEFINE companies
companies = ['Samsung', 'BASF', 'Apple', 'Tesla', 'Airbus', 'Bayer', 'BMW', 'Telefonica', 'Google', 'Allianz', 'Total']

# DEFINE start and end date
startDate = datetime.date(2017, 12, 7)
endDate = datetime.date(2017, 12, 9)
# Get all Business Days in Period
time_frame = pd.bdate_range(startDate, endDate)


# Set maximum number of articles per day
number_of_articles = 50

# DEFINE df results columns
result = dict()

for company in companies:
    print("- Starting article processing for company :", company)
    # Dictionary
    result.update({company:{}})
    for day in time_frame:
        # QUERY articles related to current company
        print("-- Start article processing for Date: ", day)

        result[company].update({day.strftime('%Y-%m-%d'): []})
        q = ER.QueryArticlesIter(conceptUri=er.getConceptUri(company), lang="eng", dateStart=day.date(),
                                 dateEnd=day.date())
        articles = q.execQuery(er, sortBy=["date", "sourceImportance"], sortByAsc=False, lang=["eng"],
                               returnInfo=ReturnInfo(
                                   articleInfo=ArticleInfoFlags(socialScore=True, originalArticle=True, categories=True,
                                                                concepts=True, sentiment=True, duplicateList=True)),
                               maxItems=number_of_articles, articleBatchSize=50)



        # Iterate over all articles about the current company
        # Calculate Sentiment and save in day`s column and index
        while True:
            try:
                article = next(articles)
            except AssertionError:
                print("Article throws assertion error!")
                continue
            except StopIteration:
                break

            result[company][day.strftime('%Y-%m-%d')].append(article['body'])


    print("- Company fully processed : ", company)



print("All Articles fully processed")
print("Save Data to csv")
PATH = "data/article_bodies_" + str(startDate) + "_" + str(endDate) + ".json"

import json
with open(PATH, 'w') as fp:
    json.dump(result, fp)

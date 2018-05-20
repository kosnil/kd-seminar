import eventregistry as ER
import datetime
from collections import defaultdict
import json
import pandas as pd
import csv
import os.path

#INITIALIZE global variables
companies_csv = "../companies.csv"
er = ER.EventRegistry(apiKey = "5ba73408-ea81-459b-abf4-6fedd8cb8ec6")  #dany
er = ER.EventRegistry(apiKey = "5fed3642-762a-4abc-aabf-ac6213c1bcea")  #philipp
analytics = ER.Analytics(er)
results = defaultdict(defaultdict)

#DEFINE companies
companies = ['BASF','Samsung']
#If file is found in 'companies_csv', file is loaded instead
if os.path.exists(companies_csv):
    companies = []
    with open(companies_csv) as f:
        reader = csv.reader(f)
        for line in reader:
            companies.append(line[0])
print('Companies viewed:',companies)

#DEFINE date
date = datetime.date(2018, 5, 19)


#ITERATE over companies
for company in companies:
    # QUERY articles related to current company
    q = ER.QueryArticlesIter(conceptUri=er.getConceptUri(company), dateStart=date, dateEnd=date)
    articles = q.execQuery(er)

    #INITIALIZE local variables
    results[company] = defaultdict()
    sentiment_df = pd.DataFrame()
    ibm_sentiment = 0
    article_count = 0
    stock_occurences = 0

    #ITERATE over all articles about the current company
    for article in articles:

        #TEXT FEATURES
        if 'stock' in article['body']:
             stock_occurences += 1

        #SENTIMENT
        #calculating sentiment value from 'article body'
        sentiment_value = analytics.sentiment(article['body'])['avgSent']
        #append article sentiment value to the companies sentiment data frame
        sentiment_df = sentiment_df.append({0: sentiment_value}, ignore_index=True)

        #IBM Sentiment
        #ibm_sentiment = ibm.getSentiment(article['body'])


    #COMBINING RESULTS
    #summarize the sentiment distribution
    sentiment_df_summary = sentiment_df.describe()
    #gefine the total number of articles
    article_count = int(sentiment_df_summary[0]["count"])

    #fill the results dictionary
    results[company]['articleCount'] = article_count
    results[company]['avgSentiment'] = sentiment_df_summary[0]["mean"]
    results[company]['stdSentiment'] = sentiment_df_summary[0]["std"]
    results[company]['25quantileSentiment'] = sentiment_df_summary[0]["25%"]
    results[company]['50quantileSentiment'] = sentiment_df_summary[0]["50%"]
    results[company]['75quantileSentiment'] = sentiment_df_summary[0]["75%"]
    results[company]['maxSentiment'] = sentiment_df_summary[0]["max"]
    results[company]['minSentiment'] = sentiment_df_summary[0]["min"]
    results[company]['avgStockOccurence'] = stock_occurences/article_count
    results[company]['ibmSentiment'] = ibm_sentiment

#Print resulting JSON
print(json.dumps(results, indent=2))






##Faster approach. However sentiment is always 'None'. returnInfo needs to be understood better
# q = ER.QueryArticles(
#         # set the date limit of interest
#         dateStart = date, dateEnd = date,
#         # find articles mentioning the company Apple
#         conceptUri = er.getConceptUri(companies[0]))
# # return the list of top 30 articles, including the concepts, categories and article image
# q.setRequestedResult(ER.RequestArticlesInfo(returnInfo = ER.ReturnInfo(articleInfo = ER.ArticleInfoFlags(url = False, title = False, body = False, sentiment = True, concepts = True, categories = True))))
# res = er.execQuery(q)
# print(type(res['articles']))
# print(res['articles']['results'][0])




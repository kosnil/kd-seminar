import eventregistry as ER
import datetime
import pandas as pd
from ibm import tone_ibm
import time
from eventregistry import *

er = ER.EventRegistry(apiKey="5ba73408-ea81-459b-abf4-6fedd8cb8ec6")  # dany
# er = ER.EventRegistry(apiKey = "5fed3642-762a-4abc-aabf-ac6213c1bcea")  #philipp
# er = ER.EventRegistry(apiKey = "7571801b-6710-4166-90cc-9c5352ddeedd")  #andi
# er = ER.EventRegistry(apiKey="dfa0a9e9-a9d7-497f-acab-54d08234bf88") #  Hendrik?
analytics = ER.Analytics(er)

# DEFINE companies
companies = ['Samsung', 'BASF', 'Apple', 'Tesla', 'Airbus', 'Bayer', 'BMW', 'Telefonica', 'Google', 'Allianz', 'Total']
# companies =['Samsung']
# DEFINE start and end date
startDate = datetime.date(2017, 9, 2)
endDate = datetime.date(2017, 11, 1)
# Get all Business Days in Period
time_frame = pd.bdate_range(startDate, endDate)

# DEFINE df results columns

columns = ['Timestamp', "ID", "articleCount", "avgSentiment", "stdSentiment", "25quantileSentiment",
           "50quantileSentiment", "75quantileSentiment", "maxSentiment", "minSentiment", "socialScore",
           "nbOfDuplicates"]
results = pd.DataFrame(index=range(0, time_frame.shape[0] * len(companies)), columns=columns)
result_index = 0

# Set maximum number of articles per day
number_of_articles = 50

for company in companies:
    print("- Starting article processing for company :", company)

    # Init Company Sentiment DF
    # Each day equals one column --> All sentiments of one day in one column
    sentiment_df = pd.DataFrame(index=range(0, number_of_articles), columns=time_frame)

    sentiment_ibm_df = pd.DataFrame(index=range(0, 7), columns=time_frame)
    sentiment_ibm_df.fillna(value=0, inplace=True)

    social_df = pd.DataFrame(index=range(0, 1), columns=time_frame)
    social_df.fillna(value=0, inplace=True)

    duplicate_df = pd.DataFrame(index=range(0, 1), columns=time_frame)
    duplicate_df.fillna(value=0, inplace=True)

    for day in time_frame:
        # QUERY articles related to current company
        print("-- Start article processing for Date: ", day)
        q = ER.QueryArticlesIter(conceptUri=er.getConceptUri(company), lang="eng", dateStart=day.date(),
                                 dateEnd=day.date())
        articles = q.execQuery(er, sortBy=["date", "sourceImportance"], sortByAsc=False, lang=["eng"],
                               returnInfo=ReturnInfo(
                                   articleInfo=ArticleInfoFlags(socialScore=True, originalArticle=True, categories=True,
                                                                concepts=True, sentiment=True, duplicateList=True)),
                               maxItems=number_of_articles, articleBatchSize=50)

        # INITIALIZE local variables
        ibm_sentiment = 0
        article_count = 0
        social_value = 0
        stock_occurences = 0
        index = 0

        # Iterate over all articles about the current company
        # Calculate Sentiment and save in day`s column and index
        index = 0
        while True:
            try:
                article = next(articles)
            except AssertionError:
                print("Article throws assertion error!")
                continue
            except StopIteration:
                break

            # Calculate text feature
            # Count Occurences of word "Stock" in article
            if 'stock' in article['body']:
                stock_occurences += 1

            # duplicateList
            duplicate_df[day] += len(article['duplicateList'])

            # SOCIAL SHARE - right now just the sum of all article-shares through all social nets
            if bool(article['shares'].values()):
                social_df[day] += sum(article['shares'].values())
            # print(social_df)

            # SENTIMENT
            # calculating sentiment value from 'article body'
            sentiment_value = analytics.sentiment(article['body'])['avgSent']
            sentiment_df[day][index] = sentiment_value
            index += 1

        # Sentiment ibm
    # ibm_time = time.time()
    # sentiment_ibm_df[article['date']] += tone_ibm.getSentiment(article['body'])
    # print("IBM TIME: " , time.time() - ibm_time)
    # print("Article TIME: ", time.time() - article_time)

    # Fill in the resulting df from sentiment_df
    for day in sentiment_df.columns:
        results.iloc[result_index]['Timestamp'] = day.strftime("%Y-%m-%d")
        results.iloc[result_index]['ID'] = company
        results.iloc[result_index]['articleCount'] = sentiment_df[day].count()
        results.iloc[result_index]['avgSentiment'] = sentiment_df[day].mean()
        results.iloc[result_index]['stdSentiment'] = sentiment_df[day].std()
        results.iloc[result_index]['25quantileSentiment'] = sentiment_df[day].quantile(0.25)
        results.iloc[result_index]['50quantileSentiment'] = sentiment_df[day].quantile(0.50)
        results.iloc[result_index]['75quantileSentiment'] = sentiment_df[day].quantile(0.75)
        results.iloc[result_index]['maxSentiment'] = sentiment_df[day].min()
        results.iloc[result_index]['minSentiment'] = sentiment_df[day].max()
        results.iloc[result_index]["socialScore"] = social_df[day].values[0]
        results.iloc[result_index]['nbOfDuplicates'] = duplicate_df[day].values[0]
        # ibm
        # results.iloc[result_index]['ibm_articleCount'] = sentiment_ibm_df[day].sum()

        # if results.iloc[result_index]['ibm_articleCount'] > 0:
        #     results.iloc[result_index]['sadness_count'] = sentiment_ibm_df[day].iloc[0]/results.iloc[result_index]['ibm_articleCount']
        #     results.iloc[result_index]['anger_count'] = sentiment_ibm_df[day].iloc[1]/results.iloc[result_index]['ibm_articleCount']
        #     results.iloc[result_index]['fear_count'] = sentiment_ibm_df[day].iloc[2]/results.iloc[result_index]['ibm_articleCount']
        #     results.iloc[result_index]['joy_count'] = sentiment_ibm_df[day].iloc[3]/results.iloc[result_index]['ibm_articleCount']
        #     results.iloc[result_index]['analytical_count'] = sentiment_ibm_df[day].iloc[4]/results.iloc[result_index]['ibm_articleCount']
        #     results.iloc[result_index]['confident_count'] = sentiment_ibm_df[day].iloc[5]/results.iloc[result_index]['ibm_articleCount']
        #     results.iloc[result_index]['tentative_count'] = sentiment_ibm_df[day].iloc[6]/results.iloc[result_index]['ibm_articleCount']

        result_index += 1

    print("-- Company fully processed : ", company)

results.fillna(value=0, inplace=True)
print(" - All Articles fully processed")
results['Timestamp'] = pd.to_datetime(results['Timestamp'], format="%Y-%m-%d")
print(" - Save Data to csv")
PATH = "data/sentiment_features_" + str(startDate) + "_" + str(endDate) + ".csv"
results.to_csv(PATH, sep=",", header=True)

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


# Return vom Vortag als Feature rein für jede Zeile
# Returns so drinne lassen

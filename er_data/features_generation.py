import eventregistry as ER
import datetime
import pandas as pd
from ibm import tone_ibm
import time
from eventregistry import *

er = ER.EventRegistry(apiKey="5ba73408-ea81-459b-abf4-6fedd8cb8ec6")  # dany
#er = ER.EventRegistry(apiKey = "5fed3642-762a-4abc-aabf-ac6213c1bcea")  #philipp
#er = ER.EventRegistry(apiKey = "7571801b-6710-4166-90cc-9c5352ddeedd")  #andi
#er = ER.EventRegistry(apiKey="dfa0a9e9-a9d7-497f-acab-54d08234bf88") # von wem? Hendrik?
analytics = ER.Analytics(er)

# DEFINE companies
companies = ['Samsung', 'BASF', 'Apple', 'Tesla', 'Airbus', 'Bayer', 'BMW', 'Telefonica', 'Google', 'Allianz', 'Total']
# DEFINE start and end date
startDate = datetime.date(2018, 5, 31)
endDate = datetime.date(2018, 5, 31)

# DEFINE df results columns
columns = ['Timestamp', "ID", "articleCount", "avgSentiment", "stdSentiment", "25quantileSentiment",
           "50quantileSentiment", "75quantileSentiment", "maxSentiment", "minSentiment","ibm_articleCount", "sadness_count", "anger_count",
           "fear_count", "joy_count", "analytical_count", "confident_count", "tentative_count"]
results = pd.DataFrame(index=range(0, pd.date_range(startDate, endDate).shape[0] * len(companies)), columns=columns)
#results.fillna(value=0,inplace=True)
result_index = 0

for company in companies:

    print("- Starting article processing for company :", company)
    # QUERY articles related to current company
    q = ER.QueryArticlesIter(conceptUri=er.getConceptUri(company), lang="eng", dateStart=startDate, dateEnd=endDate)

    articles = q.execQuery(er, sortBy=["date","sourceImportance"], sortByAsc=False, lang= ["eng"],
              returnInfo=ReturnInfo(articleInfo=ArticleInfoFlags(socialScore = True, originalArticle=True, categories= True, concepts= True, sentiment=True, duplicateList=True)),
              articleBatchSize=50)
    #print(articles)

    # Init Company Sentiment DF
    # Each day equals one column --> All sentiments of one day in one column
    sentiment_df = pd.DataFrame(index=range(0, 2000), columns=pd.date_range(startDate, endDate).format("%Y-%m-%d")[1:])
    sentiment_ibm_df = pd.DataFrame(index=range(0, 7), columns=pd.date_range(startDate, endDate).format("%Y-%m-%d")[1:])
    sentiment_ibm_df.fillna(value=0, inplace=True)
    social_df = pd.DataFrame(index=range(0, 1), columns=pd.date_range(startDate, endDate).format("%Y-%m-%d")[1:])
    social_df.fillna(value=0, inplace=True)
    duplicate_df = pd.DataFrame(index=range(0, 1), columns=pd.date_range(startDate, endDate).format("%Y-%m-%d")[1:])
    duplicate_df.fillna(value=0, inplace=True)

    # INITIALIZE local variables
    ibm_sentiment = 0
    article_count = 0
    social_value = 0
    stock_occurences = 0
    index = 0
    date = pd.date_range(startDate, endDate).format("%Y-%m-%d")[len(pd.date_range(startDate, endDate))]
    print("-- Start  prcessing day : ", date)
    # Iterate over all articles about the current company
    # Calculate Sentiment and save in day`s column and index
    while True
        try:
            article_time = time.time()
            article = next(articles)
        except AssertionError:
            print("Article throws assertion error!")
            continue
        except StopIteration:
            break

        if date != article['date']:
            index = 0
            print("-- Day fully processed : ", date)

        print(article)

        # Calculate text feature
        # Count Occurences of word "Stock" in article
        if 'stock' in article['body']:
            stock_occurences += 1

        #duplicateList
        duplicate_df[article['date']] += len(article['duplicateList'])

        #SOCIAL SHARE - right now just the sum of all article-shares through all social nets
        if bool(article['shares'].values()):
            social_df[article['date']] += sum(article['shares'].values())
        #print(social_df)

        # SENTIMENT
        # calculating sentiment value from 'article body'
        er_time = time.time()
        sentiment_value = analytics.sentiment(article['body'])['avgSent']
        sentiment_df[article['date']][index] = sentiment_value
        #print("ER TIME: " , time.time() - er_time)
        index += 1
        date = article['date']

        # Sentiment ibm
        #ibm_time = time.time()
        #sentiment_ibm_df[article['date']] += tone_ibm.getSentiment(article['body'])
        #print("IBM TIME: " , time.time() - ibm_time)
        #print("Article TIME: ", time.time() - article_time)

    # Fill in the resulting df from sentiment_df
    for day in sentiment_df.columns:
        results.iloc[result_index]['Timestamp'] = str(day)
        results.iloc[result_index]['ID'] = company
        results.iloc[result_index]['articleCount'] = sentiment_df[day].count()
        results.iloc[result_index]['avgSentiment'] = sentiment_df[day].mean()
        results.iloc[result_index]['stdSentiment'] = sentiment_df[day].std()
        results.iloc[result_index]['25quantileSentiment'] = sentiment_df[day].quantile(0.25)
        results.iloc[result_index]['50quantileSentiment'] = sentiment_df[day].quantile(0.50)
        results.iloc[result_index]['75quantileSentiment'] = sentiment_df[day].quantile(0.75)
        results.iloc[result_index]['maxSentiment'] = sentiment_df[day].min()
        results.iloc[result_index]['minSentiment'] = sentiment_df[day].max()
        results.iloc[result_index]["socialScore"] = social_df[day]
        results.iloc[result_index]['nbOfDuplicates'] = duplicate_df[day]
        #ibm
        #results.iloc[result_index]['ibm_articleCount'] = sentiment_ibm_df[day].sum()

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

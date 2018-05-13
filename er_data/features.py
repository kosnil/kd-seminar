import eventregistry as ER
import datetime
from collections import defaultdict

er = ER.EventRegistry(apiKey = "5ba73408-ea81-459b-abf4-6fedd8cb8ec6")  #dany
#er = ER.EventRegistry(apiKey = "5fed3642-762a-4abc-aabf-ac6213c1bcea")  #philipp

analytics = ER.Analytics(er)

companies = ['Apple','Samsung']
date = datetime.date(2018, 5, 1)

results = defaultdict(defaultdict)

#iterate over companies
for company in companies:
    # query articles related to current company
    q = ER.QueryArticlesIter(conceptUri=er.getConceptUri(company), dateStart=date, dateEnd=date)

    #initialize variables
    results[company] = defaultdict()
    articles = q.execQuery(er)
    article_count = 0
    stock_occurences = 0
    art_sentiment_total = 0
    art_sentiment_max = 0
    art_sentiment_min = 0

    #iterate over articles
    for article in articles:

        #article count
        article_count += 1

        #Text features
        if 'stock' in article['body']:
             stock_occurences += 1

        #sentiment
        sentiment_value = analytics.sentiment(article['body'])['avgSent']
        art_sentiment_total += sentiment_value
        if sentiment_value > art_sentiment_max:
            art_sentiment_max = sentiment_value
        if sentiment_value < art_sentiment_min:
            art_sentiment_min = sentiment_value

    #creating results
    results[company]['articleCount']= article_count
    results[company]['avgStockOccurence'] = stock_occurences/article_count
    results[company]['avgArtSentiment'] = art_sentiment_total/article_count
    results[company]['maxArtSentiment'] = art_sentiment_max
    results[company]['minArtSentiment'] = art_sentiment_min

    print(results[company])
    for c in results:
        print(c)
    print(results)







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




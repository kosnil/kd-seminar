from watson_developer_cloud import ToneAnalyzerV3
from watson_developer_cloud import WatsonApiException
import json
import time
import pandas as pd

# -------------TONE ANALYZER SPECIFICATION
# https://www.ibm.com/watson/developercloud/tone-analyzer/api/v3/python.html?python#tone
# submit  no more than 128KB of total input content
#         no more than 1000 individual sentences in json/plain text/html
# service analyzes    the first 1000 sentences for document level analysis
#                     the first 100 sentences for sentence level analysis
#
# test = {'tst': 0}
ibm_sentiment_count_df = pd.DataFrame()



def getSentiment(article):
    start = time.time()
    tone_analyzer = ToneAnalyzerV3(
        version='2017-09-21',
        username="c80a8e92-463d-44f6-8f59-0361510df470",  # danny
        password="154pwVYbbWJD",  # danny
        url="https://gateway.watsonplatform.net/tone-analyzer/api"
    )

    try:
        # tone_analyzer.set_detailed_response(True) #to receive header and body in once
        tone_analyzer.set_default_headers({'x-watson-learning-opt-out': 'true'})  # opt out from data collection by ibm
        response = tone_analyzer.tone({
            'text': article},
            content_type='application/json',
            sentences='false')
    except WatsonApiException as ex:
        print("Tone Analyzer failed with status code " + str(ex.code) + ": " + ex.message)
    end = time.time()
    print(json.dumps(response, indent=2))
    # print('time: {} '.format(end - start))
    #ibm_sentiment = treatScore(response)
    ibm_sentiment = countScore(response)
    return ibm_sentiment


def treatScore(score):  #
    if score:
        for tone in score['document_tone']['tones']:
            if tone['tone_id'] == 'analytical':
                analytical_value = tone['score']
            else:
                analytical_value = -1
            if tone['tone_id'] == 'confident':
                confident_value = tone['score']
            else:
                confident_value = -1
            if tone['tone_id'] == 'tentative':
                tentative_value = tone['score']
            else:
                tentative_value = -1
            if tone['tone_id'] == 'sadness':
                sadness_value = tone['score']
            else:
                sadness_value = -1  # 0, -1
            if tone['tone_id'] == 'anger':
                anger_value = tone['score']
            else:
                anger_value = -1
            if tone['tone_id'] == 'fear':
                fear_value = tone['score']
            else:
                fear_value = -1
            if tone['tone_id'] == 'joy':
                joy_value = tone['score']
            else:
                joy_value = -1

    ibm_sentiment_df = ibm_sentiment_df.append(
        {'sadness_value': sadness_value, 'anger_value': anger_value, 'fear_value': fear_value, 'joy_value': joy_value,
         'analytical_value': analytical_value, 'confident_value': confident_value, 'tentative_value': tentative_value},
        ignore_index=True)
    print(ibm_sentiment_df)
    ibm_sentiment_df_summary = ibm_sentiment_df.describe()
    print(ibm_sentiment_df_summary)
    return ibm_sentiment_df_summary


def countScore(score):
    analytical_count = 0
    confident_count = 0
    tentative_count = 0
    sadness_count = 0
    anger_count = 0
    fear_count = 0
    joy_count = 0
    if score:
        for tone in score['document_tone']['tones']:
            if tone['tone_id'] == 'analytical':
                analytical_count = 1
            elif tone['tone_id'] == 'confident':
                confident_count = 1
            elif tone['tone_id'] == 'tentative':
                tentative_count = 1
            elif tone['tone_id'] == 'sadness':
                sadness_count = 1
            elif tone['tone_id'] == 'anger':
                anger_count = 1
            elif tone['tone_id'] == 'fear':
                fear_count = 1
            elif tone['tone_id'] == 'joy':
                joy_count = 1
    print(analytical_count)
    sentiment_ibm_df = [sadness_count, anger_count, fear_count, joy_count,
                        analytical_count, confident_count, tentative_count]
    print(sentiment_ibm_df)
    return sentiment_ibm_df

    # #ibm
    # results[company]['avg_ibmSentiment'] = ibm_sentiment[0]["mean"]
    # results[company]['std_ibmSentiment'] = ibm_sentiment[0]["std"]
    # results[company]['25quantile_ibmSentiment'] = ibm_sentiment[0]["25%"]
    # results[company]['50quantile_ibmSentiment'] = ibm_sentiment[0]["50%"]
    # results[company]['75quantile_ibmSentiment'] = ibm_sentiment[0]["75%"]
    # results[company]['max_ibmSentiment'] = ibm_sentiment[0]["max"]
    # results[company]['min_ibmSentiment'] = ibm_sentiment[0]["min"]

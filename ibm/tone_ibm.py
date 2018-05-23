from watson_developer_cloud import ToneAnalyzerV3
from watson_developer_cloud import WatsonApiException
import json
import time
# -------------TONE ANALYZER SPECIFICATION
# https://www.ibm.com/watson/developercloud/tone-analyzer/api/v3/python.html?python#tone
# submit  no more than 128KB of total input content
#         no more than 1000 individual sentences in json/plain text/html
# service analyzes    the first 1000 sentences for document level analysis
#                     the first 100 sentences for sentence level analysis
#
def getSentiment(article):
    start = time.time()
    tone_analyzer = ToneAnalyzerV3(
        version='2017-09-21',
        username="c80a8e92-463d-44f6-8f59-0361510df470",
        password="154pwVYbbWJD",
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
    #print('time: {} '.format(end - start))
    return response['document_tone']


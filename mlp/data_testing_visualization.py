import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

'''
Determine Input-Vector
- to define input-vector demonstrate significant linear relationship between Input and Output
- necessary to show, because input needs information about output
- want to maximize the input information, 
    - no random information 
    - input needs to carry highly significant information
'''

# read in data
dataset     = pd.read_csv("final_data/complete_data.csv")
dataset.head()

# companies
companies   = dataset.ID.unique()

for i in range(0, len(companies)):
    company     = dataset.loc[dataset['ID'] == companies[i]]

    # define input vector
    X_train = company[[ 'articleCount', 'avgSentiment','stdSentiment',
                        '25quantileSentiment', '50quantileSentiment', '75quantileSentiment',
                        'maxSentiment', 'minSentiment', 'Previous_Day_Return']]

    # define output vector
    Y_train = company[['Next_Day_Return']]

    plt.figure('Company: %s' % companies[i])
    plt.subplot(211).set_title('Input-Data')
    # subplot 1
    plt.plot(X_train['Previous_Day_Return'], label='Previous-Day Return')
    plt.plot(X_train['avgSentiment'], label='AVG Sentiment')
    plt.plot(X_train['stdSentiment'], label='Std Sentiment')
    plt.plot(X_train['25quantileSentiment'], label='25 quantile Sentiment')
    plt.plot(X_train['50quantileSentiment'], label='50 quantile Sentiment')
    plt.plot(X_train['75quantileSentiment'], label='75 quantile Sentiment')
    plt.plot(X_train['maxSentiment'], label='Max Sentiment')
    plt.plot(X_train['minSentiment'], label='Min Sentiment')
    plt.plot(Y_train['Next_Day_Return'], label='Next_Day_Return')
    plt.legend()

    ### Test for linear relationship
    Y = Y_train['Next_Day_Return']

    print("# -- Linear Regression --- ")

    X_Previous_Day_Return = X_train['Previous_Day_Return']
    results_Previous_Day_Return = sm.OLS(Y, sm.add_constant(X_Previous_Day_Return)).fit()
    print("# -- Next-Day Return and avgSentiment --- ")
    print("ß0 ", results_Previous_Day_Return.params[0])
    print("ß1 ", results_Previous_Day_Return.params[1])
    print("t-value ", results_Previous_Day_Return.results.pvalues)
    print("r^2 ", results_Previous_Day_Return.rsquared)
    print("# ------------------------------------------")

    X_avgSentiment = X_train['avgSentiment']
    results_avgSentiment = sm.OLS(Y, sm.add_constant(X_avgSentiment)).fit()

    X_stdSentiment = X_train['stdSentiment']
    results_stdSentiment = sm.OLS(Y, sm.add_constant(X_stdSentiment)).fit()

    X_25quantileSentiment = X_train['25quantileSentiment']
    results_25quantileSentiment = sm.OLS(Y, sm.add_constant(X_25quantileSentiment)).fit()

    X_50quantileSentiment = X_train['50quantileSentiment']
    results_50quantileSentiment = sm.OLS(Y, sm.add_constant(X_50quantileSentiment)).fit()

    X_75quantileSentiment = X_train['75quantileSentiment']
    results_75quantileSentiment = sm.OLS(Y, sm.add_constant(X_75quantileSentiment)).fit()

    X_maxSentiment = X_train['maxSentiment']
    results_maxSentiment = sm.OLS(Y, sm.add_constant(X_maxSentiment)).fit()

    X_minSentiment = X_train['minSentiment']
    results_minSentiment = sm.OLS(Y, sm.add_constant(X_minSentiment)).fit()

    X_articleCount = X_train['articleCount']
    results_articleCount = sm.OLS(Y, sm.add_constant(X_articleCount)).fit()

    # subplot 2
    # calculate correlation: how strong is this linear relationship?
    # calculate covariance: is there a linear relationship?

    cov_ndR_pdR     = np.cov(np.matrix(Y_train['Next_Day_Return']), np.matrix(X_train['Previous_Day_Return']))
    corr_ndR_pdR    = np.corrcoef(np.matrix(Y_train['Next_Day_Return']), np.matrix(X_train['Previous_Day_Return']))
    print("# -- Next-Day Return and Previous_Day_Return --- ")
    print("Covariance: \n", cov_ndR_pdR)
    print("Correlation: \n", corr_ndR_pdR)
    print("# ------------------------------------------")

    cov_ndR_avgS     = np.cov(np.matrix(Y_train['Next_Day_Return']), np.matrix(X_train['avgSentiment']))
    corr_ndR_avgS    = np.corrcoef(np.matrix(Y_train['Next_Day_Return']), np.matrix(X_train['avgSentiment']))
    print("# -- Next-Day Return and avgSentiment --- ")
    print("Covariance: \n", cov_ndR_avgS)
    print("Correlation: \n", corr_ndR_avgS)
    print("# ------------------------------------------")

    cov_ndR_stdS     = np.cov(np.matrix(Y_train['Next_Day_Return']), np.matrix(X_train['stdSentiment']))
    corr_ndR_stdS    = np.corrcoef(np.matrix(Y_train['Next_Day_Return']), np.matrix(X_train['stdSentiment']))
    print("# -- Next-Day Return and stdSentiment --- ")
    print("Covariance:  \n", cov_ndR_stdS)
    print("Correlation: \n", corr_ndR_stdS)
    print("# ------------------------------------------")

    cov_ndR_maxS     = np.cov(np.matrix(Y_train['Next_Day_Return']), np.matrix(X_train['maxSentiment']))
    corr_ndR_maxS    = np.corrcoef(np.matrix(Y_train['Next_Day_Return']), np.matrix(X_train['maxSentiment']))
    print("# -- Next-Day Return and maxSentiment --- ")
    print("Covariance: \n", cov_ndR_maxS)
    print("Correlation: \n", corr_ndR_maxS)
    print("# ------------------------------------------")

    cov_ndR_minS     = np.cov(np.matrix(Y_train['Next_Day_Return']), np.matrix(X_train['minSentiment']))
    corr_ndR_minS    = np.corrcoef(np.matrix(Y_train['Next_Day_Return']), np.matrix(X_train['minSentiment']))
    print("# -- Next-Day Return and minSentiment --- ")
    print("Covariance: \n", cov_ndR_minS)
    print("Correlation: \n", corr_ndR_minS)
    print("# ------------------------------------------")

    cov_ndR_artC     = np.cov(np.matrix(Y_train['Next_Day_Return']), np.matrix(X_train['articleCount']))
    corr_ndR_artC    = np.corrcoef(np.matrix(Y_train['Next_Day_Return']), np.matrix(X_train['articleCount']))
    print("# --- Next-Day Return and articleCount --- ")
    print("Covariance: \n", cov_ndR_artC)
    print("Correlation: \n", corr_ndR_artC)
    print("# ------------------------------------------")

    X_plot = np.linspace(-0.1, 0.1, 100)
    colors = ['b', 'c', 'y', 'm', 'r']

    plt.figure()
    plt_pdR = plt.scatter(Y_train['Next_Day_Return'], X_train['Previous_Day_Return'], color=colors[0])
    plt.plot(X_plot, X_plot * results_Previous_Day_Return.params[1] + results_Previous_Day_Return.params[0])
    plt.plot(X_plot, X_plot * results_avgSentiment.params[1] + results_avgSentiment.params[0])
    plt.plot(X_plot, X_plot * results_stdSentiment.params[1] + results_stdSentiment.params[0])
    plt.plot(X_plot, X_plot * results_25quantileSentiment.params[1] + results_25quantileSentiment.params[0])
    plt.plot(X_plot, X_plot * results_50quantileSentiment.params[1] + results_50quantileSentiment.params[0])
    plt.plot(X_plot, X_plot * results_75quantileSentiment.params[1] + results_75quantileSentiment.params[0])
    plt.plot(X_plot, X_plot * results_maxSentiment.params[1] + results_maxSentiment.params[0])
    plt.plot(X_plot, X_plot * results_minSentiment.params[1] + results_minSentiment.params[0])
    plt_avgS = plt.scatter(Y_train['Next_Day_Return'], X_train['avgSentiment'], color=colors[1])
    plt_stdS = plt.scatter(Y_train['Next_Day_Return'], X_train['stdSentiment'], color=colors[2])
    plt_maxS = plt.scatter(Y_train['Next_Day_Return'], X_train['maxSentiment'], color=colors[3])
    plt_minS = plt.scatter(Y_train['Next_Day_Return'], X_train['minSentiment'], color=colors[4])
    plt.legend((plt_pdR, plt_avgS, plt_stdS, plt_maxS, plt_minS),
               ('Previous_Day_Return', 'avgSentiment', 'stdSentiment', 'maxSentiment', 'minSentiment'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)

    plt.show()

    # separate scatter plot for article count, otherwise y-axis scale is too inaccurate
    plt.figure()
    plt.plot(X_plot, X_plot * results_articleCount.params[1] + results_articleCount.params[0])
    plt_artC = plt.scatter(Y_train['Next_Day_Return'], X_train['articleCount'], color=colors[4])
    plt.legend((plt_artC),
               ('articleCount'),
               scatterpoints=1,
               loc='lower left',
               ncol=1,
               fontsize=8)

    plt.show()

    plt.plot(X_train['articleCount'], label='AVG Sentiment')

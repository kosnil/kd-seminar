def cumulate_returns(finance_data, portfolio_data):
    finance_data = finance_data.copy()
    #
    # portfolio_data = pd.read_csv(modell,sep='\t')
    tp_portfolio_data = portfolio_data.copy()
    tn_portfolio_data = portfolio_data.copy()

    # Both Ascending by Time
    finance_data = finance_data.sort_values('Timestamp')
    tp_portfolio_data = tp_portfolio_data.sort_values('Timestamp')
    tn_portfolio_data = tn_portfolio_data.sort_values('Timestamp')

    # Set Index
    finance_data = finance_data.set_index('Timestamp')
    tp_portfolio_data = tp_portfolio_data.set_index('Timestamp')
    tn_portfolio_data = tn_portfolio_data.set_index('Timestamp')

    stocks = finance_data.columns

    ########################################

    # Set Both Datasets on same time range
    start = max(tp_portfolio_data.index.min(), finance_data.index.min(), tn_portfolio_data.index.min())
    end = min(tp_portfolio_data.index.max(), finance_data.index.max(), tn_portfolio_data.index.max())
    finance_data = finance_data.loc[start:end]
    tp_portfolio_data = tp_portfolio_data.loc[start:end]
    tn_portfolio_data = tn_portfolio_data.loc[start:end]

    # Check has to be 0
    tp_portfolio_data.shape[0] - finance_data.shape[0]

    ########################################

    # Shift portfolio binary variable to day where investment get paid out
    # Set 1 to day when return is realized or 0 when return is not realized
    # shift > 0 --> Shift nach unten
    # shift < 0 --> shift nach oben
    for stock in stocks:
        tp_portfolio_data[stock] = tp_portfolio_data[stock].shift(1)
        tn_portfolio_data[stock] = tn_portfolio_data[stock].shift(1)

    # Cant invest before the first day
    tp_portfolio_data = tp_portfolio_data.fillna(0)
    tn_portfolio_data = tn_portfolio_data.fillna(0)

    ########################################

    # Calculate Mean return for equal weighted strategy
    equal_weighted_mean = finance_data.mean(1, True)

    ########################################

    # Copy DF for filter matrix
    # ptf_mask gets filled with 0 from tn_portfolio data
    # ptf_mask gets filled with 1 from tp_portfolio_data

    ptf_mask = tp_portfolio_data.copy()
    ptf_mask.loc[:] = np.nan

    ########################################

    # Reset Index
    # Damit .loc[] funktioniert
    finance_data = finance_data.reset_index()

    tp_portfolio_data = tp_portfolio_data.reset_index()
    tn_portfolio_data = tn_portfolio_data.reset_index()

    ptf_mask = ptf_mask.reset_index()

    ########################################


    # Fill ptf_mask with 0/1


    for stock in stocks:
        buy_indices = np.where(tp_portfolio_data[stock] == 1)[0]
        sell_indices = np.where(tn_portfolio_data[stock] == 0)[0]
        nan_indices = np.intersect1d(buy_indices, sell_indices)

        ptf_mask.loc[buy_indices, [stock]] = 1
        ptf_mask.loc[sell_indices, [stock]] = 0
        ptf_mask.loc[nan_indices, [stock]] = np.nan

    ########################################

    tp_finance_data = finance_data.copy()
    tn_finance_data = finance_data.copy()

    ########################################


    # TPs
    # Clear all return entries, where we do not want to invest
    for stock in stocks:
        tp_finance_data.loc[np.where(ptf_mask[stock] != 1)[0], [stock]] = np.nan

    ########################################

    # Long
    # Set all Nans to 0, where we do not invest
    tp_finance_data['Model_Mean_Long'] = tp_finance_data.mean(1, True)
    tp_finance_data.loc[np.isnan(tp_finance_data['Model_Mean_Long']), 'Model_Mean_Long'] = 0

    ########################################


    # TNs
    # Clear all return entries, where we do not want to invest
    for stock in stocks:
        tn_finance_data.loc[np.where(ptf_mask[stock] != 0)[0], [stock]] = np.nan

    ########################################


    # Shorting -> *-1
    # Set all Nans to 0, where we do not invest
    tn_finance_data['Model_Mean_Short'] = tn_finance_data.mean(1, True)
    tn_finance_data.loc[np.isnan(tn_finance_data['Model_Mean_Short']), 'Model_Mean_Short'] = 0

    ########################################

    return [tn_finance_data['Model_Mean_Short'].sum() * -1, tp_finance_data['Model_Mean_Long'].sum(), \
            equal_weighted_mean.sum()]
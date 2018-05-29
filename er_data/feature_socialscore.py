def getSocialScore(social_value):
    facebook_count = 0
    pinterest_count = 0

    if social_value:
        for element in social_value:
            if 'facebook' in social_value:
                facebook_count = social_value['facebook']
            if 'pinterest' in social_value:
                pinterest_count = social_value['pinterest']
    social_df = [facebook_count, pinterest_count]

    return social_df
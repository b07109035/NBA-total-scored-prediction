import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from correlation import drop_first_game
from math import fabs, sqrt


df = pd.read_csv("data\efficiency_with_O_and_D_0602.csv")


# season_avg = []
# season_yr = []
# for season, season_df in df.groupby("season_year"):
#     season_avg.append(season_df["fg3a"].mean())
#     season_yr.append(season)
# plt.scatter(season_yr, season_avg)
# plt.show()

variable_list = ['possesions', 'possessions_oppo', 'offensive_rating', 'defensive_rating', 'offensive_rating_oppo', 'defensive_rating_oppo', 'is_home_true', 'back_to_back_true', 'total_bet']
# variable_list = ['total_bet']
# , 
#                 'back_to_back_true', 'is_home_true', 'is_home_false', 'back_to_back_false', 'rank', 'rank_oppo',
#                 'season_year', 'fgm_cumulative', 'fga_cumulative',
#                 'fg_pct_cumulative', 'fg3m_cumulative',
#                 'fg3a_cumulative', 'fg3_pct_cumulative', 'ftm_cumulative', 'fta_cumulative', 'ft_pct_cumulative', 
#                 'oreb_cumulative', 'dreb_cumulative', 'reb_cumulative', 
#                 'ast_cumulative', 'stl_cumulative', 'blk_cumulative', 
#                 'tov_cumulative','pf_cumulative'] #, 'pts_per_game', 'lost_pts_per_game', 'oppo_pts_per_game', 'oppo_lost_pts_per_game']

# for season, season_df in df.groupby('season_year'):

#     X = season_df[variable_list]
#     y = season_df["total_real"]
#     X = sm.add_constant(X)

#     model = sm.OLS(y, X).fit()
#     print(season, model.rsquared)
    

# print("---------------")
# print("\n")
# team_name_list = df["matchup"].str[:3].unique()
# df = df[df["min"] == 240]
# for i in range(10, 51, 5):
#     df = drop_first_game(df, i, False)
#     df.to_csv(f"data\\first_{i}.csv")
# df = df.drop_duplicates(subset=['game_id'])

# df = pd.read_csv("data\efficiency_with_O_and_D_0602.csv.csv")

for team, team_df in df.groupby(df["matchup"].str[:3]):
    team_df = team_df[team_df["min"] == 240]
    # print(team_df)
    # print(train_df)
    if True:
        try:
            for year in range(2006, 2018):
                _df = team_df[(team_df["season_year"] == year) | (team_df["season_year"] == year + 1)]
                # train_df = train_df[train_df['rank_diff'] == 20]
                train_df = _df.head(30)
                test_df = _df.drop(train_df.index)

                X = train_df[variable_list]
                y = train_df["total_real"]
                # print(X)
                # print(y)
                X_test = test_df[variable_list]
                y_test = test_df["total_real"]
                
                X = sm.add_constant(X)
                X_test = sm.add_constant(X_test)

                model = sm.OLS(y, X).fit()
                model_pred = model.predict(X_test)
                mse = np.mean((y_test - model_pred) ** 2)
                # mse = np.mean((y - model.predict(X)) ** 2)            
                ssr = np.sum((model_pred - y_test) ** 2)
                sst = np.sum((y_test - np.mean(y_test)) ** 2)
                r2 = 1 - (ssr / sst)
                if model.rsquared > 0.5:
                    print(model.summary())

                # spread_avg = 0
                # for i in team_df['spread']:x1
                #     spread_avg += fabs(i)            
                # spread_avg /= len(team_df['spread'])
                # if spread_avg > sqrt(mse):
                #     print('666666666666')

                # print(year, team, model.rsquared, sqrt(mse))
                # print(year)
                # print(test_df['total_bet'])
                # print(y_test)
                # print(model_pred)
                # a = pd.DataFrame(columns = ['total_bet', 'total_real', 'predicted'])
                # a['total_bet'] = test_df['total_bet']
                # a['total_real'] = y_test
                # a['predicted'] = model_pred
                # a.to_csv(f"data\\{year}.csv")

                if r2 > 0.5:
                    print('====================LETSGOOOO====================')
                    # print(team, model.summary())

                # for i, j in zip(y_test, model_pred):
                #     print(i, j)



        except:
            print(f"{team} has no data before 2010")


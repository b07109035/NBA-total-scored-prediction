from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import math
from sklearn.metrics import r2_score 

df = pd.read_csv("data\efficiency_with_O_and_D_0602.csv")
df.dropna(inplace = True)


def drop_pts_per_game_is_zero(df):
  
  df = df[df ["pts_per_game"] != 0] # drop the first game of each team
  return df



def drop_first_row_of_each_team(df, drop_rows: int): # filt out the rows that don't have enough sample to calculate the average


  filted_df = pd.DataFrame(columns = df.columns)
  for season, season_df in df.groupby("season"):
    for team, team_df in season_df.groupby(df["matchup"].str[:4]):
      team_df.drop(team_df.index[:drop_rows], inplace = True)
      filted_df["back_to_back"] = filted_df["back_to_back"].astype(bool)
      filted_df["is_home"] = filted_df["is_home"].astype(bool)
      team_df["back_to_back"] = team_df["back_to_back"].astype(bool)
      team_df["is_home"] = team_df["is_home"].astype(bool)
      filted_df = pd.concat([filted_df, team_df])
  return filted_df

def drop_first_game(df: pd.DataFrame, drop_rows: int, drop_place: bool): # filt out the rows that don't have enough sample to calculate the average
  filted_df = pd.DataFrame(columns = df.columns)
  for season, season_df in df.groupby("season"):
    for team, team_df in season_df.groupby(df['matchup'].str[:4]):
      print(len(team_df))
      if drop_place == True:  
        team_df.drop(team_df.index[:drop_rows], inplace = True)
      if drop_place == False:
        team_df.drop(team_df.index[drop_rows:], inplace = True)
      try:
        filted_df = pd.concat([filted_df, team_df])
      except:
        pass
  return filted_df


def drop_playoff(df):
  
  df = df[df["season_type"] == "Regular Season"]
  return df
  


def draw_correlation():

  """
  """
  print(df.columns)

  for season, season_df in df.groupby("season"):
    print(f"--------------------{season}--------------------")
    pts_per_game = season_df["pts_per_game"]
    lost_pts_per_game = season_df["lost_pts_per_game"]
    oppo_pts_per_game = season_df["oppo_pts_per_game"]
    oppo_lost_pts_per_game = season_df["oppo_lost_pts_per_game"]
    total_predict = season_df["pts_per_game"] + season_df["oppo_lost_pts_per_game"] - season_df["lost_pts_per_game"] - season_df["oppo_pts_per_game"]
    total_real = season_df["total_real"]



  # correlation
    pts_per_game_corr, _ = pearsonr(pts_per_game, total_real)
    lost_pts_per_game_corr, _ = pearsonr(lost_pts_per_game, total_real)
    oppo_pts_per_game_corr, _ = pearsonr(oppo_pts_per_game, total_real)
    oppo_lost_pts_per_game_corr, _ = pearsonr(oppo_lost_pts_per_game, total_real)
    total_predict_corr, _ = pearsonr(total_predict, total_real)
    print(f"pts_per_game_corr: {pts_per_game_corr}")
    print(f"lost_pts_per_game_corr: {lost_pts_per_game_corr}")        
    print(f"oppo_pts_per_game_corr: {oppo_pts_per_game_corr}")
    print(f"oppo_lost_pts_per_game_corr: {oppo_lost_pts_per_game_corr}")
    print(f"total_predict_corr: {total_predict_corr}")




def regression_model(df):

  model = LinearRegression()
  X = df[['pts_per_game', 'lost_pts_per_game', 'oppo_pts_per_game', 'oppo_lost_pts_per_game']]
  y = df['total_real']


def train_test_split(df, split_ratio: float):
  train_df = pd.DataFrame(columns = df.columns)
  test_df = pd.DataFrame(columns = df.columns)

  # split by season
  for season, season_df in df.groupby("season_year"):
    if season == 2006 or season == 2007 or season == 2008:
      train_df = pd.concat([train_df, season_df])
    if season == 2009 or season == 2010:
      test_df = pd.concat([test_df, season_df])
  train_df.to_csv("data\\train_data_by_year.csv", index = False)
  test_df.to_csv("data\\test_data_by_year.csv", index = False)
  
  # split by team
  # for team, team_df in df.groupby(df['matchup'].str[:4]):
  #   train_df = pd.concat([train_df, team_df.iloc[:int(len(team_df) * split_ratio)]])
  #   test_df = pd.concat([test_df, team_df.iloc[int(len(team_df) * split_ratio):]])
  # train_df.to_csv("data\\train_data.csv", index = False)
  # test_df.to_csv("data\\test_data.csv", index = False)
  
  
  # return train_df, test_df

if __name__ == "__main__":

  # df = pd.read_csv("data/efficiency_with_O_and_D_0602.csv")
  # df = df[df['min'] == 240]
  # train_test_split(df, 0.8)


  # pd.set_option('display.max_rows', None)
  # filt data
  # df = drop_first_row_of_each_team(df,1)
  # df = drop_playoff(df)
  
  # ouput final data
  # df.to_csv("data\\final_data.csv", index = False)


  # df = pd.read_csv("data\efficiency_with_O_and_D_0601.csv")
  # season_avg_pts_list = [] 
  # season_avg_pts_list = [] 
  # season_avg_pts_list = [] 
  # season_avg_pts_list = [] 
  # season_avg_pts_list = [] 
  # season_avg_pts_list = [] 
  # season_avg_pts_list = [] 
  # for season, season_df in df.groupby("season"):
  #   season_avg_pts_list.append(season_df["pts_per_game"].mean())
    
  # plt.scatter(df["season"].unique(), season_avg_pts_list)
  # plt.show()

  # test and predict
  # draw_correlation()

  # regression_model(df)



  # 輸出NBA每年平均得分
  # for season, season_df in df.groupby("season_year"):
  #   print(season)
  #   # print(season_df)
  #   season_df["game_date"] = pd.to_datetime(season_df["game_date"], format = "%m/%d/%Y")
  #   season_df = season_df.sort_values("game_date")
  #   # print(season_df["accu_total_real"].iloc[-1])
  #   plt.scatter(season, season_df["accu_total_real"].iloc[-1], s = 100, c = "red", alpha = 0.5)
  # plt.xlabel("years")
  # plt.ylabel("mean points")
  # plt.title("total_point_scored_mean")
  # plt.show()


  # assumption1: 雙方rank差距越大，越容易出現outlier
  """
  df = pd.read_csv("data/efficiency_with_O_and_D_0602.csv")
  df = drop_first_row_of_each_team(df,30)
  print(len(df))
  print(df.head())
  
  for season, season_df in df.groupby("season_year"):
    print(season)
    num = 0

    # higher ranking diff
    for team, team_df in season_df.groupby(season_df["matchup"].str[:4]):
      if [i for i in team_df["rank"]][-1] > 5 and [i for i in team_df["rank"]][-1] < 25:
        continue
      plt.plot(np.array([i for i in range(16)]), np.array([0 for i in range(16)]), c = "gray", linewidth = 1)
      team_df = team_df[((team_df["rank"] - team_df["rank_oppo"]) < 5) & ((team_df["rank"] - team_df["rank_oppo"]) > -5)]
      num += 1
      ax = plt.subplot(6, 5, num)
      x_point_low_diff = np.array([i for i in range(len(team_df))])      
      y_point_low_diff = np.array([i for i in (team_df["total_real"] - team_df["accu_total_real"])])
      plt.scatter(x_point_low_diff, y_point_low_diff, c = 'blue', s = 5, alpha = 0.5)

    # lower ranking diff 
    num = 0
    for team, team_df in season_df.groupby(season_df["matchup"].str[:4]):
      if [i for i in team_df["rank"]][-1] > 5 and [i for i in team_df["rank"]][-1] < 25:
        continue
      team_df = team_df[((team_df["rank"] - team_df["rank_oppo"]) > 15) | ((team_df["rank"] - team_df["rank_oppo"]) < -15)]
      num += 1
      ax = plt.subplot(6, 5, num)
      x_point_low_diff = np.array([i for i in range(len(team_df))])      
      y_point_low_diff = np.array([i for i in (team_df["total_real"] - team_df["accu_total_real"])])
      plt.scatter(x_point_low_diff, y_point_low_diff, c = 'red', s = 5, alpha = 0.5)
      ax.set_title(f"{team}", fontsize = 6, pad = 10)
      ax.set_ylim(-50, 50)
      ax.set_xlim(0, 15)
    plt.show()
  """
  


  # visulize data which can compare the data's variance between good team and bad team
  """
  df = pd.read_csv("data/efficiency_with_O_and_D_0602.csv")
  # df = drop_first_row_of_each_team(df,30)
  df = df[df["min"] == 240]
  print(len(df))
  print(df.head())
  
  for season, season_df in df.groupby("season_year"):
    print(season)
    low_num = 0
    high_num = 0 #4
    # higher ranking diff
    for team, team_df in season_df.groupby(season_df["matchup"].str[:4]):
      rank = [i for i in team_df["rank"]][-1]
      if rank < 5:
        print(team, rank)
        low_num += 1
        ax = plt.subplot(3, 4, low_num)
        x_point_low_diff = np.array([i for i in range(len(team_df))])      
        y_point_low_diff = np.array([i for i in (team_df["total_real"] - team_df["accu_total_real"])])
        var = sum([x ** 2 for x in y_point_low_diff]) / len(y_point_low_diff)
        avg_line = np.array([team_df["total_real"].mean() for i in range(len(team_df))])
        plt.plot(x_point_low_diff, avg_line, c = "gray", linewidth = 1)
        plt.scatter(x_point_low_diff, y_point_low_diff, c = 'blue', s = 5, alpha = 0.8)
        ax.set_title(f"{rank} / {team} / {var}", fontsize = 6, pad = 10)
        ax.title.set_position([.5, 1.05])

      if rank > 25: # team_df = team_df[((team_df["rank"] - team_df["rank_oppo"]) > 15) | ((team_df["rank"] - team_df["rank_oppo"]) < -15)]
        print(team, rank)
        high_num += 1
        ax = plt.subplot(5, 6, high_num)
        x_point_high_diff = np.array([i for i in range(len(team_df))])      
        y_point_high_diff = np.array([i for i in (team_df["total_real"] - team_df["accu_total_real"])])
        var = sum([x ** 2 for x in y_point_high_diff]) / len(y_point_high_diff)
        avg_line = np.array([team_df["total_real"].mean() for i in range(len(team_df))])
        plt.plot(x_point_high_diff, avg_line, c = "gray", linewidth = 1)
        plt.scatter(x_point_high_diff, y_point_high_diff, c = 'red', s = 5, alpha = 0.8)
        ax.set_title(f"{rank} / {team} / {var}", fontsize = 6, pad = 10)
      
      
        ax.set_ylim(-80, 80)
        ax.set_xlim(0, 45)
        plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
    plt.show()
  """

  # plt that analyze the variance of each team
  # df = pd.read_csv("data/efficiency_with_O_and_D_0602.csv")
  # df = drop_first_row_of_each_team(df, 30)
  # num = 0
  # for season, season_df in df.groupby("season_year"):
  #   num += 1
  #   print(season, end = "\n")
       
    # var_list = []
    # rank_list = []
    # for team, team_df in season_df.groupby(season_df["matchup"].str[:4]):
    #   rank = [i for i in team_df["rank"]][-1]
    #   var = team_df["total_real"].var()
    #   var_list.append(var)
    #   rank_list.append(rank)
    # plt.scatter(rank_list, var_list, s = 10, c = "blue", alpha = 0.5)  
    # plt.show()


  # plt that analyze the variance of high-rank team v.s low_rank team
  #   rank_diff_level_list = []
  #   one_ten_diff_list = []
  #   ten_twenty_diff_list = []
  #   twenty_thirty_diff_list = []
    
  #   season_df["rank_diff"] = season_df["rank"] - season_df["rank_oppo"]
  #   for i in season_df["rank_diff"]:
  #     rank_diff_level_list.append(math.fabs(i))
        
  #   season_df["rank_diff_level"] = rank_diff_level_list
  #   var_list = []
  #   level_list = []
  #   for rank_diff_level, rank_diff_level_df in season_df.groupby("rank_diff_level"):
  #     level_list.append(rank_diff_level)
  #     var_list.append(rank_diff_level_df["total_real"].var())
  #   ax = plt.subplot(4, 3, num)
  #   plt.scatter(level_list, var_list, s = 100, c = "blue", alpha = 0.5)
  # plt.savefig(f"C:\\Users\\jack\\Desktop\\output_image\\rank_var.png")
  # plt.show()
    

  # overview the asymtotic of the data
  # for season, season_df in df.groupby("season_year"):
  #   print(df.columns)

  #   input_variable = ['pts_per_game', 'lost_pts_per_game',
  #      'oppo_pts_per_game', 'oppo_lost_pts_per_game', 'total_bet','possesions', 'possessions_oppo', 'offensive_rating',
  #      'defensive_rating', 'rank', 'rank_oppo','fgm_cumulative', 'fga_cumulative',
  #      'fg_pct_cumulative','fg3m_cumulative',
  #      'fg3a_cumulative', 'fg3_pct_cumulative',
  #      'ftm_cumulative', 'fta_cumulative', 
  #      'ft_pct_cumulative', 'oreb_cumulative', 
  #      'dreb_cumulative', 'reb_cumulative', 'ast_cumulative',
  #      'stl_cumulative', 'blk_cumulative',
  #      'tov_cumulative', 'pf_cumulative']
  #   for team, team_df in season_df.groupby(season_df["matchup"].str[:4]):
  #     print(team)
  #     if team == "ATL ":
  #       num = 0
  #       for variable in input_variable:
  #         num += 1
  #         ax = plt.subplot(4, 7, num)
  #         plt.plot(np.array([i for i in range(len(team_df))]), np.array([j for j in team_df[variable]]), c = "gray", linewidth = 1)
  #         ax.set_title(f"{variable}", fontsize = 10, pad = -10)
  #         plt.subplots_adjust(hspace = 0.5, wspace = 0.3)

  #   plt.show()          
  #   break


  # data overview
  # df = pd.read_csv("data/efficiency_with_O_and_D_0602.csv")
  # df = df[df['min'] == 240]
  # train_test_split(df, 0.8)

  # df = df.drop_duplicates(subset = ["game_id"], keep = "last") 

  # num = 0
  # # for season, season_df in df.groupby("season_year"):  
  # #   plt.figure()
  # # descriptive statistics 1
  # for team, team_df in df.groupby(df["matchup"].str[:4]):
  #   num += 1
  #   ax = plt.subplot(5, 8, num)
  #   ax.set_title(f"{team}", fontsize = 6, pad = 10)
  #   plt.hist(team_df["fgm"], bins = 20, alpha = 0.8)
  #   plt.subplots_adjust(hspace = 0.8, wspace = 0.3)
  # plt.show()


  # discriptive staststics 2
  # num = 0
  # plt.figure()
  # for season, season_df in df.groupby('season_year'):
  #   num += 1
  #   ax = plt.subplot(3, 4, num)
  #   ax.set_title(f"{season}")
  #   plt.hist(season_df['fgm'], bins = 20, alpha = 0.3)
  #   plt.subplots_adjust(hspace = 0.8, wspace = 0.8)
  # plt.show()


  # descriptive statistics 3
  # for team, team_df in df.groupby(df["matchup"].str[:4]):
  #   # team_df = team_df[['pts_per_game', 'lost_pts_per_game']]
  #   # sns.heatmap(team_df.corr(), annot = True)
  #   # plt.show()
  #   team_df = team_df[['possesions', 'offensive_rating', 'defensive_rating']]
  #   sns.heatmap(team_df.corr(), annot = True)
  #   plt.show()


  # model-------------------------------------------------------------------------------------

  variable_list = ['possesions', 'possessions_oppo', 'offensive_rating', 'defensive_rating', 'total_bet', 
                   'back_to_back_true', 'is_home_true']
                  #  'season_year', 'fgm_cumulative', 'fga_cumulative']
                    # 'fg_pct_cumulative', 'fg3m_cumulative',
                    # 'fg3a_cumulative', 'fg3_pct_cumulative', 'ftm_cumulative', 'fta_cumulative', 'ft_pct_cumulative', 
                    # 'oreb_cumulative', 'dreb_cumulative', 'reb_cumulative', 
                    # 'ast_cumulative', 'stl_cumulative', 'blk_cumulative', 
                    # 'tov_cumulative','pf_cumulative']


  train_df = pd.read_csv("data//train_data_by_year.csv")
  train_df = pd.read_csv("data//efficiency_with_O_and_D_0602.csv")
  train_df.drop_duplicates(subset = ["game_id"], inplace = True)
  test_df = pd.read_csv("data//test_data_by_year.csv")
  X_train = train_df[variable_list].astype(float)
  X_train = sm.add_constant(X_train)

  X_test = test_df[variable_list].astype(float)
  X_test = sm.add_constant(X_test)
  
  y_train_real = train_df['total_real'].astype(float)
  y_test_real = test_df['total_real'].astype(float)

  model = sm.OLS(y_train_real, X_train).fit()

  y_train_pred = model.predict(X_train)
  y_test_pred = model.predict(X_test)
  # plt.scatter(y_train, y_train_pred)
  plt.scatter(y_test_real, y_test_pred, s = 20, alpha = 0.5)
  # plt.show()
  print(model.summary())
  ssr = np.sum((y_test_pred - y_test_real) ** 2)
  sst = np.sum((y_test_real - np.mean(y_test_real)) ** 2)
  rsquared = 1 - (ssr / sst)
  print(rsquared)
  sns.heatmap(train_df[variable_list].corr(), annot = True)
  plt.show()














  """
  team_name_list = df["matchup"].str[:3].unique()
  num = 0
  for team_name in team_name_list:
    team_df = df[df['matchup'].str[:3] == team_name]
    team_df_test = test_df[test_df['matchup'].str[:3] == team_name]
    num += 1
    # before feature selection
    X = team_df[['pts_per_game', 'lost_pts_per_game',
      'oppo_pts_per_game', 'oppo_lost_pts_per_game', 'fgm_cumulative', 'fga_cumulative',
      'fg_pct_cumulative', 'fg3m_cumulative',
      'fg3a_cumulative', 'fg3_pct_cumulative', 'ftm_cumulative', 'fta_cumulative', 
      'ft_pct_cumulative', 'oreb_cumulative', 
      'dreb_cumulative', 'reb_cumulative', 'ast_cumulative',
      'stl_cumulative', 'blk_cumulative', 
      'tov_cumulative','pf_cumulative', 'total_bet', 'season_year', 'back_to_back_false', 'back_to_back_true', 'is_home_false', 'is_home_true']].astype(float)

    X = sm.add_constant(X)
    y = team_df['total_real'].astype(float)
    model = sm.OLS(y, X).fit()
    
    # print(f"{team_name} / {model.rsquared} / Before feature selection")
    # print(model.summary())

    # after_feature_selection
    X_2 = team_df[['fgm_cumulative', 'fga_cumulative',
      'fg_pct_cumulative', 'fg3m_cumulative',
      'fg3a_cumulative', 'fg3_pct_cumulative', 'ftm_cumulative', 'fta_cumulative', 
      'ft_pct_cumulative', 'oreb_cumulative', 
      'dreb_cumulative', 'reb_cumulative', 'ast_cumulative',
      'stl_cumulative', 'blk_cumulative', 
      'tov_cumulative','pf_cumulative', 'possesions', 'possessions_oppo', 'offensive_rating',
      'defensive_rating', 'total_bet', 'season_year', 'offensive_rating_oppo', 'defensive_rating_oppo', 'back_to_back_false', 'back_to_back_true', 'is_home_false', 'is_home_true']].astype(float)
    X_2 = sm.add_constant(X_2)

    X_2_test = team_df_test[['fgm_cumulative', 'fga_cumulative',
      'fg_pct_cumulative', 'fg3m_cumulative',
      'fg3a_cumulative', 'fg3_pct_cumulative', 'ftm_cumulative', 'fta_cumulative', 
      'ft_pct_cumulative', 'oreb_cumulative', 
      'dreb_cumulative', 'reb_cumulative', 'ast_cumulative',
      'stl_cumulative', 'blk_cumulative', 
      'tov_cumulative','pf_cumulative', 'possesions', 'possessions_oppo', 'offensive_rating',
      'defensive_rating', 'total_bet', 'season_year', 'offensive_rating_oppo', 'defensive_rating_oppo', 'back_to_back_false', 'back_to_back_true', 'is_home_false', 'is_home_true']].astype(float)
    X_2_test = sm.add_constant(X_2_test)
    
    y_train = team_df['total_real'].astype(float)
    y_test = test_df['total_real'].astype(float)

    model_2 = sm.OLS(y_train, X_2).fit()
    y_pred = model_2.predict(X_2_test)
    print(y_pred)
    print(y_test)
    # test_r2 = r2_score(y_test, y_pred)

    print(f"{team_name} / Training R-Squared: {model_2.rsquared}")
    print(f"{team_name} / Testing R-Squared: {test_r2}")
    # print(model.summary())
  """
  """
  # VIF Calculattion
  for team, team_df in df.groupby('season_year'): #df.groupby(df['matchup'].str[:4]):
    X = team_df[['pts_per_game', 'lost_pts_per_game',
       'oppo_pts_per_game', 'oppo_lost_pts_per_game'
      'fgm_cumulative', 'fga_cumulative',
       'fg_pct_cumulative', 'fg3m_cumulative',
       'fg3a_cumulative', 'fg3_pct_cumulative', 'ftm_cumulative', 'fta_cumulative', 
       'ft_pct_cumulative', 'oreb_cumulative', 
       'dreb_cumulative', 'reb_cumulative', 'ast_cumulative',
       'stl_cumulative', 'blk_cumulative', 
       'tov_cumulative','pf_cumulative']]
    y = team_df['total_real']
  
    X = sm.add_constant(X)

    model = sm.OLS(y, X)
    result = model.fit()
    # print(result.summary())

    vif = pd.DataFrame()
    vif['feature'] = X.columns
    vif['VIF Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif.to_csv("data/vif{}.csv", index = False)
    print(vif)

  print('-------------------')
  for team, team_df in df.groupby(df['matchup'].str[:4]):
    X = team_df[['possesions', 'possessions_oppo', 'offensive_rating', 'defensive_rating', 'offensive_rating_oppo', 'defensive_rating_oppo']]
    y = team_df['total_real']
  
    X = sm.add_constant(X)

    model = sm.OLS(y, X)
    result = model.fit()
    # print(result.summary())
    vif = pd.DataFrame()
    vif['feature'] = X.columns
    vif['VIF Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif)
  """ 

  # point rank correlation
  """
  df1 = df[df['rank_diff'] < 10]
  df2 = df[(df['rank_diff'] >= 10) & (df['rank_diff'] < 20)]
  df3 = df[df['rank_diff'] >= 20]
  plt.figure()

  ax = plt.subplot(311)
  plt.hist(df1['total_real'], bins = 20, alpha = 0.5)
  ax.set_title('rank_diff < 10')
  plt.xticks(np.arange(140, 280, 20))

  ax = plt.subplot(312)
  plt.hist(df2['total_real'], bins = 20, alpha = 0.5)
  ax.set_title('rank_diff 10 - 20')
  plt.xticks(np.arange(140, 280, 20))

  ax = plt.subplot(313)  
  plt.hist(df3['total_real'], bins = 20, alpha = 0.5)
  ax.set_title('rank_diff > 20')
  plt.xticks(np.arange(140, 280, 20))
  plt.subplots_adjust(hspace = 0.5)
  plt.show()
  """


  # point and lost_point correlation
  # sns.heatmap(df[['pts_per_game', 'lost_pts_per_game', 'oppo_pts_per_game', 'oppo_lost_pts_per_game']].corr(), annot = True)
  # plt.show()
  # sns.heatmap(df[['possesions, possessions_oppo, offensive_rating, defensive_rating, offensive_rating_oppo, defensive_rating_oppo', 'total_real']].corr(), annot = True)
  # plt.show()


  # sns.heatmap(df[['pts', 'pts_oppo']].corr(), annot = True)
  
  # for season, season_df in df.groupby("season_year"):
  #   plt.scatter(season_df["pts"], season_df["pts_oppo"], c = "blue", alpha = 0.5)
  #   plt.show()




  # variable correlation heatmap
  """
  print(df.columns)
  df = df[df["season_year"] <= 2010]
  for teams in df["matchup"].str[:4].unique():
    df = pd.read_csv("data/efficiency_with_O_and_D_0602.csv")
    df = df[df["matchup"].str[:4] == teams]
    df = df[['season_year',
            'pts_per_game', 
            'lost_pts_per_game',
            'oppo_pts_per_game', 
            'oppo_lost_pts_per_game',
            # 'matchup', 
            #  'pts', 
            #  'pts_oppo', 
            #  'avg_outter', 
            #  'oppo_avg_outter', 
            #  'outter',
            #  'accu_total_real', 
            'total_real', 
            #  'total_bet', 
            #  'spread',
            #  'possesions', 
            #  'possessions_oppo',
            #  'offensive_rating', 
            #  'defensive_rating', 
            #  'wl', 
            #  'w', 
            #  'l', 
            #  'w_pct',
            #  'fgm_cumulative', 
            #  'fga_cumulative', 
            #  'fg_pct_cumulative', 
            #  'fg3m_cumulative', 
            #  'fg3a_cumulative'
            #  'fg3_pct_cumulative', 
            #  'ftm_cumulative', 
            #  'fta_cumulative', 
            #  'ft_pct_cumulative', 'oreb_cumulative', 
            #  'dreb_cumulative', 'reb_cumulative', 'ast_cumulative',
            #  'stl_cumulative', 'blk_cumulative', 
            #  'tov_cumulative', 'pf_cumulative'
            ]]
    # print(df.columns)
    # for season, season_df in df.groupby("season_year"):
    plt.figure(figsize = (15, 15))
    plt.suptitle(f"{teams}")
    sns.heatmap(df.corr(), annot = True)
    plt.show()
  """
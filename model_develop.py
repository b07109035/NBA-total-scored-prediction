from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv("data/efficiency_with_O_and_D_0602.csv")
df = df[df["pts_per_game"] != 0] # drop the first game of each team
df = df[df["oppo_pts_per_game"] != 0]
df = df[df["oppo_lost_pts_per_game"] != 0]
df = df[df["season_type"] == "Regular Season"]
df.dropna(inplace = True)
df["offense"] = df["pts_per_game"] + df["oppo_lost_pts_per_game"]
df["defense"] = df["lost_pts_per_game"] + df["oppo_pts_per_game"]


input_variables_list = ['lost_pts_per_game', "offense", "defense", "avg_outter", "oppo_avg_outter", 
       'oppo_pts_per_game',
       'total_bet','w_pct','fgm_cumulative',
       'fga_cumulative', 
       'fg3m_cumulative', 'fg3a_cumulative', 
       'ftm_cumulative', 'fta_cumulative',
       'oreb_cumulative',
       'dreb_cumulative', 'reb_cumulative',
       'tov_cumulative', 'pf_cumulative', "possesions", "offensive_rating", "defensive_rating", "back_to_back_false", "is_home_true"] #'pts_per_game', 'oppo_lost_pts_per_game',  'ast_cumulative','stl_cumulative', 'blk_cumulative',

input_variables_list = ['pts_per_game', 'lost_pts_per_game',
       'oppo_pts_per_game', 'oppo_lost_pts_per_game', 
       'rank', 'rank_oppo', 'fgm', 'fga',
       'fg3m', 'fg3a', 'ftm',
       'fta', 'oreb', 'dreb',
       'reb', 'ast',
       'stl', 'blk', 'tov',
       'pf']



def dummy_back_to_back(df):
  dummy_df = pd.get_dummies(df["back_to_back"])
  print(dummy_df.columns)  
  # df = pd.concat([df, dummy_df], axis = 1)
  df["back_to_back_false"] = dummy_df[False]
  df["back_to_back_true"] = dummy_df[True]

  pd.set_option("expand_frame_repr", False)
  print(df.head(1))
  df.to_csv("data/efficiency_with_O_and_D_0531.csv")

def dummy_is_home(df):
  dummy_df = pd.get_dummies(df["is_home"])
  df["is_home_false"] = dummy_df["f"]
  df["is_home_true"] = dummy_df["t"]
  print(df.head(1))
  df.to_csv("data/efficiency_with_O_and_D_0531.csv")


def data_visulize_overview():
  for season, season_df in df.groupby("season"):
      
      rnum = 0
      cnum = 0
      num = 0
      team_list = []
      fig = plt.figure()
      for variable in input_variables_list:
          
          num += 1
          # rnum += 1 if rnum < 5 else 0
          # cnum += 1 if rnum == 5 else 0
          # rnum = 0 if rnum == 5 else rnum
          ax = plt.subplot(5, 5, num)
          x_point = np.array([i for i in season_df[variable]])
          y_point = np.array([int(i) for i in season_df["total_real"]])
          plt.scatter(x_point, y_point, s = 1, c = "blue")
          print(x_point)
          print(y_point)
          ax.set_title(f"{variable} vs total_real", fontsize = 6, pad = 10)


      print(team_list)
      plt.show()


def data_visulize_one_over_years(variable:str):
  for team, team_df in df.groupby(df.matchup.str[:4]):
    print(team)
    num = 0
    for season, season_df in team_df.groupby("season"):
      num += 1
      ax = plt.subplot(4, 3, num)
      x_point = np.array([i for i in season_df[f"{variable}"]])
      y_point = np.array([int(i) for i in season_df["total_real"]])
      plt.scatter(x_point, y_point, s = 1, c = "blue")
      ax.set_title(f"{season}/{variable} vs total_real", fontsize = 6, pad = 10)
    plt.show()


def data_visulize_one_over_teams(variable:str):
  for season, season_df in df.groupby("season"):
    print(season)
    num = 0
    for team, team_df in season_df.groupby(df.matchup.str[:4]):
      num += 1
      ax = plt.subplot(6, 5, num)
      x_point = np.array([i for i in team_df[f"{variable}"]])
      y_point = np.array([int(i) for i in team_df["total_real"]])
      plt.scatter(x_point, y_point, s = 1, c = "blue")
      ax.set_title(f"{team}/{variable} vs total_real", fontsize = 6, pad = 10)
    plt.show()



def train():
   
  for season, season_df in df.groupby("season_year"):
    model = LinearRegression()
    x = season_df[input_variables_list]
    y = season_df["total_real"]
    x2 = sm.add_constant(x)
    est = sm.OLS(y, x2).fit()
    print(est.summary())
    residuals = est.resid
    print(residuals)
    sm.qqplot(residuals, line = "s")
    _, p_value = stats.shapiro(residuals)
    plt.title(f"{season}qqplot/ p_value = {p_value}")
    plt.show()  

  # plt.show()
    # plt.savefig(f"qqplot_{season}.png")
    # model.fit(x, y)
    # y_predict = model.predict(x)
    # r2 = r2_score(y, y_predict)
    # ax = plt.subplot(1, 2, 1)
    # plt.scatter(y, y_predict, s = 1, c = "blue")
    # ax.set_title(f"{season} / predict")
    # plt.plot()
    # ax = plt.subplot(1, 2, 2)
    # plt.scatter([i for i in season_df["total_real"]], [i for i in season_df["total_bet"]], s = 1, c = "blue")
    # plt.show()

    # print(season)
    # print(r2)

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





if __name__ == "__main__":


  # df = drop_first_row_of_each_team(df, 10)
  # print(df.columns)
  # dummy_back_to_back(df)
  # dummy_is_home(df)

  # data_visulize_overview()
  # data_visulize_one_over_years("total_bet")
  # data_visulize_one_over_teams("outter")
  
  df.drop_duplicates(subset = ["game_id"], inplace = True)
  train()





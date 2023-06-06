import pandas as pd
import numpy as np
import csv


df = pd.read_csv("data\efficiency_with_O_and_D_0531.csv")
df.drop(["Unnamed: 0.1", "Unnamed: 0"], inplace = True, axis = 1)
def add_avg_point(df):
    
    final_df = pd.DataFrame(columns = ['game_id', 'game_date', 'matchup', 'pts_per_game', 'lost_pts_per_game',
       'oppo_pts_per_game', 'oppo_lost_pts_per_game', 'pts', 'pts_oppo',
       'accu_total_real', 'total_real', 'total_bet', 'spread', 'team_id',
       'back_to_back', 'is_home', 'possesions', 'possessions_oppo',
       'offensive_rating', 'defensive_rating', 'wl', 'w', 'l', 'w_pct', 'min',
       'fgm', 'fgm_cumulative', 'fga', 'fga_cumulative', 'fg_pct',
       'fg_pct_cumulative', 'fg3m', 'fg3m_cumulative', 'fg3a',
       'fg3a_cumulative', 'fg3_pct', 'fg3_pct_cumulative', 'ftm',
       'ftm_cumulative', 'fta', 'fta_cumulative', 'ft_pct',
       'ft_pct_cumulative', 'oreb', 'oreb_cumulative', 'dreb',
       'dreb_cumulative', 'reb', 'reb_cumulative', 'ast', 'ast_cumulative',
       'stl', 'stl_cumulative', 'blk', 'blk_cumulative', 'tov',
       'tov_cumulative', 'pf', 'pf_cumulative', 'a_team_id', 'season_year',
       'season_type', 'season', 'book_name', 'price1', 'price2',
       'back_to_back_false', 'back_to_back_true', 'is_home_false',
       'is_home_true'])
  
    df["game_date"] = pd.to_datetime(df["game_date"], format = "%m/%d/%Y")
    df = df.sort_values("game_date")
    print(df.head(10))
    
    for season_year, season_df in df.groupby("season_year"):
      print(season_year)
      print(season_df.head())
      accu_total_real = 0
      accu_total_real_list = [0]
      count = 0
      for total_real in season_df["total_real"]:
        accu_total_real += total_real
        count += 1
        accu_total_real_list.append(accu_total_real / count)        
      season_df.insert(9, "accu_total_real", accu_total_real_list[:-1])
      print(season_df.columns)
      final_df = pd.concat([final_df, season_df], axis = 0)
      # print(final_df.head())
    final_df.to_csv("data\efficiency_with_O_and_D_0601.csv")







if __name__ == "__main__":
  add_avg_point(df)





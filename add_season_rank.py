import pandas as pd
import numpy as np



df = pd.read_csv("data\efficiency_with_O_and_D_0601.csv")
df["game_date"] = pd.to_datetime(df["game_date"], format = "%m/%d/%Y")
df = df.sort_values("game_date")
print(df.head())


def add_season_rank():
  final_df = pd.DataFrame(columns = df.columns)
  final_df["rank"] = np.nan
  for season, season_df in df.groupby("season_year"):
    team_rank = []
    for x in season_df["matchup"].str[:3].unique():
      team_rank.append([x, 0])
    for game_date,  game_date_df in season_df.groupby("game_date"):
      print(game_date)
      print(game_date_df)
      team_df_by_date = game_date_df["matchup"].str[6:]
      wlp_df_by_date = game_date_df["w_pct"]

      buffer_ranking_df = pd.DataFrame(team_rank, columns = ["team_name", "wlp"])
      for team_name, wlp in zip(team_df_by_date, wlp_df_by_date):
        team_name = team_name[2:] if len(team_name) > 3 else team_name
        print(len(team_name), team_name, wlp, sep = "/")
        print(team_rank)
        for i in team_rank:
          if i[0] == team_name:
            print("-------------------------------")
            i[1] = wlp
        buffer_ranking_df["wlp"] = [i[1] for i in team_rank]
        buffer_ranking_df["rank"] = buffer_ranking_df["wlp"].rank(method = "first", ascending = False)
        # print("-------------------------------")
        
        print(buffer_ranking_df)
      
      team_rank_final = []
      for team_name in team_df_by_date:
        team_name = team_name[2:] if len(team_name) > 3 else team_name
        for buffer_team_name, buffer_team_rank in zip(buffer_ranking_df["team_name"], buffer_ranking_df["rank"]):
          if team_name == buffer_team_name:
            team_rank_final.append([team_name, buffer_team_rank])
      ranking_df = pd.DataFrame(team_rank_final, columns = ["team_name", "rank"])
      print(ranking_df)
      game_date_df["rank"] = [i for i in ranking_df["rank"]]
      game_date_df = game_date_df.astype({"rank": "int32"})
      game_date_df.drop(["Unnamed: 0"], inplace = True, axis = 1)
      
      print(game_date_df)
      final_df = pd.concat([final_df, game_date_df])
      print(final_df)      
  final_df.to_csv("data\efficiency_with_O_and_D_0602.csv", index = False)

if __name__ == "__main__":
  add_season_rank()









import pandas as pd
import numpy as np
import csv

df = pd.read_csv("data\efficiency_with_O_and_D_and_without_key_player.csv")
print(df.columns)
df.drop(["Unnamed: 0"], axis = 1, 	inplace = True)
def seperate_df_by_season_and_calculate_points_per_game():
		final_df = pd.DataFrame({
			'game_id' : [0], 
			'game_date' : [0], 
			'matchup' : [0], 
			'pts_per_game' : [0], 
			'lost_pts_per_game' : [0], 
			'team_id' : [0], 
			'is_home' : [0], 
			'wl' : [0], 
			'w' : [0], 
			'l' : [0],
      'w_pct' : [0], 
			'min' : [0], 
			'fgm' : [0], 
			'fga' : [0], 
			'fg_pct' : [0], 
			'fg3m' : [0], 
			'fg3a' : [0], 
			'fg3_pct' : [0],
      'ftm' : [0], 
			'fta' : [0], 
			'ft_pct' : [0], 
			'oreb' : [0], 
			'dreb' : [0], 
			'reb' : [0], 
			'ast' : [0], 
			'stl' : [0], 
			'blk' : [0],
      'tov' : [0], 
			'pf' : [0], 
			'pts' : [0], 
			'pts_oppo' : [0], 
			'a_team_id' : [0], 
			'season_year' : [0],
      'season_type' : [0], 
			'season' : [0], 
			'book_name' : [0], 
			'total' : [0], 
			'price1' : [0], 
			'price2' : [0]
			})
		

		groups_by_season = df.groupby("season")
		for season, season_group in groups_by_season:
			print(f"----------------{season}------------")
			print(season_group.head())

			team_groups = season_group.groupby(df["matchup"].str[:4])
			for team, team_group in team_groups:
				print(team)
				print(team_group)
				pts_per_game_list = []
				lost_pts_per_game_list = []
				point_sum = 0
				lost_point_sum = 0
				loop_count = 1
				for pts, lost_pts in zip(team_group["pts"], team_group["pts_oppo"]):
					pts_per_game_list.append(point_sum / loop_count)
					lost_pts_per_game_list.append(lost_point_sum / loop_count)
					loop_count += 1 if point_sum != 0 else 0
					point_sum += pts
					lost_point_sum += lost_pts
				team_group.insert(3, "pts_per_game", pts_per_game_list)
				team_group.insert(4, "lost_pts_per_game", lost_pts_per_game_list)
				print(team_group)
				final_df = pd.concat([final_df, team_group])
				print(final_df)

		# final_df.to_csv("data\\efficiency_with_O_and_D_0519.csv", index = False)
		"""
				for home_status, host_group in team_group.groupby("is_home"):
					# print(home_status)
					# print(host_group)

					pts_per_game_list = []
					lost_pts_per_game_list = []
					loop_count = 0
					point_sum = 0
					lost_point_sum = 0
					for pts, lost_pts in zip(host_group["pts"], host_group["pts_oppo"]):
						# print(pts)
						loop_count += 1
						point_sum += pts
						lost_point_sum += lost_pts
						pts_per_game_list.append(point_sum / loop_count)
						lost_pts_per_game_list.append(lost_point_sum / loop_count)
					# print(pts_per_game_list)
					# team_group["pts_per_game"] 
					host_group.insert(3, "pts_per_game_new", pts_per_game_list)
					host_group.insert(4, "lost_pts_per_game_new", lost_pts_per_game_list)
					print(host_group)
					print("-----------------------------------------")
					final_df = pd.concat([final_df, host_group], axis = 0)
					print(final_df)	
					break
		# final_df.to_csv("data\\test.csv", index = False)
			"""
		

def add_back_to_back_column():

	df = pd.read_csv("data\efficiency_with_O_and_D.csv")
	# print(df.head())
	df['game_date'] = pd.to_datetime(df['game_date'], format='%m/%d/%Y')
	df.insert(6, "back_to_back", False)
	# print(df.loc[1, "game_date"].day)
	for i in range(1, len(df)):
		if (df.loc[i, "game_date"] - df.loc[i - 1, "game_date"]).days == 1:
			df.loc[i, "back_to_back"] = True
	print(df)
	df.to_csv("data\\efficiency_with_O_and_D_and_back_to_back.csv", index = False)





if __name__ == "__main__":

	# add_back_to_back_column()
	seperate_df_by_season_and_calculate_points_per_game()








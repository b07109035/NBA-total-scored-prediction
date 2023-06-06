import pandas as pd
import csv
import time
import math

"""
def output_via_season():

	df = pd.read_csv("data\games_details.csv")
	for season in df["SEASON"].unique():

		df[df["SEASON"] == season].to_csv("data\games_details" + str(season) + ".csv")

def output_game_details_via_season():
	

	for season in range(2003, 2023):
		season_df_details = pd.DataFrame()
		df = pd.read_csv(f"data\games{season}.csv")
		print(season)
		df_details = pd.read_csv(f"data\games_details.csv")
		for i in df["GAME_ID"]:
			# for j in df_details["GAME_ID"]:
			# 	if j == i:
			# print(i)
			# print(df_details[df_details["GAME_ID"] == i])
			season_df_details = pd.concat([season_df_details, df_details[df_details["GAME_ID"] == i]])
		season_df_details.to_csv(f"data\games_details{season}.csv")
		
		print("------------------------")


def calculate_each_player_total_attendance_and_min():

	# for season in range(2003, 2023):
		
		# print(season)
		# df = pd.read_csv(f"data\games_details{season}.csv")
		# df["MIN"] = df["MIN"].str.split(":").str[0]
		# df = df.dropna(subset = ["MIN"])

		# player_name_list = df["PLAYER_NAME"].unique() # list
		# min_list = []
		# attend_count_list = []
		# for i in range(len(player_name_list)):
		# 	min_list.append(0)
		# 	attend_count_list.append(0)
		# pd.DataFrame({"PLAYER_NAME" : player_name_list, "MIN" : min_list, "attend" : attend_count_list}).to_csv(f"data\games_player_min_{season}.csv")
	

	for season in range(2018, 2022):

		df = pd.read_csv(f"data\games_details{season}.csv")
		df["MIN"] = df["MIN"].str.split(":").str[0]
		df = df.dropna(subset = ["MIN"])

		processed_df = pd.read_csv(f"data\games_player_min_{season}.csv")
		print(processed_df)
		raw_df = df[["PLAYER_NAME", "MIN"]]
		for player in processed_df["PLAYER_NAME"]:
			print(player)
			for j, k in zip(raw_df["PLAYER_NAME"], raw_df["MIN"]):
				print(f"name: {j}, min: {k}")
				if j == player:
					print(f"{player} -----------------------------")
					boolean_index = processed_df["PLAYER_NAME"] == player
					boolean_index = boolean_index.reset_index(drop = True)
					processed_df.loc[boolean_index, "MIN"] += float(k)
					processed_df.loc[boolean_index, "attend"] += 1
					# print(processed_df)
					# time.sleep(10)
		processed_df.to_csv(f"data\games_player_min_{season}.csv")
		print("------------------------")



def add_avg_time_into_games_player_min():

	for season in range(2003, 2023):
		df = pd.read_csv(f"data\games_player_min_{season}.csv")
		df["avg_time"] = df["MIN"] / df["attend"]
		df.to_csv(f"data\games_player_min_{season}.csv")


"""
def drop_minutes_less_than_a_constant(constant: int):

	for season in range(2003, 2023):
		df = pd.read_csv(f"data\games_player_min_{season}.csv")
		df = df[df["avg_time"] >= constant]
		df.to_csv(f"data/games_player_min_filted_{season}.csv")
		print(season)
		print(df)

def ouput_the_key_players_each_season():
	
	for season in range(2003, 2022):
		
		filt_df = pd.read_csv(f"data/games_player_min_filted_{season}.csv")
		season_key_player_list = []
		for i in filt_df["PLAYER_NAME"]:
			season_key_player_list.append(i)
		print(f"----------{season}----------")
		# print(season_key_player_list)

		df = pd.read_csv(f"data/games_details{season}.csv")

		team_name_list = df["TEAM_ABBREVIATION"].unique()
		team_key_player_df = pd.DataFrame(columns = team_name_list)
		print(team_key_player_df)
		for team_name, roster_df in df.groupby("TEAM_ABBREVIATION"):
			print(f"----------{team_name}----------")
			# print(roster_df.head())
			key_player_list = []
			for game_id, team_roster_df in roster_df.groupby("GAME_ID"):
				print(game_id)
				print(team_roster_df)
				for player in team_roster_df["PLAYER_NAME"]:
					if player in season_key_player_list:
						key_player_list.append(player)
				key_player_series = pd.Series(key_player_list)
				team_key_player_df[team_name] = key_player_series
				print(team_key_player_df)
				team_key_player_df.to_csv(f"data/team_key_player_{season}.csv")
				break


def output_game_ids_without_key_player(): # could be traded or DNP

	game_with_diff_key_player = [] # could be traded or DNP
	player_dff = []
	game_with_diff_key_player_df = pd.DataFrame(columns = ["season", "game_id", "PLAYER_NAME"])
	for season in range(2006, 2007): # range(2003, 2022)

		print(f"----------{season}----------")
		key_palyer_df = pd.read_csv(f"data/team_key_player_{season}.csv")
		# print(key_palyer_df)

		game_detail_df = pd.read_csv(f"data/games_details{season}.csv")
		for game_id, game_df in game_detail_df.groupby("GAME_ID"):
			for team_name in game_df["TEAM_ABBREVIATION"].unique():
				team_roster_df = game_df[game_df["TEAM_ABBREVIATION"] == team_name]
				for player in team_roster_df["PLAYER_NAME"]:
					try:
						if (player in key_palyer_df[team_name].unique()) and (math.isnan(team_roster_df.loc[team_roster_df["PLAYER_NAME"] == player, "MIN"].values[0]) == True):
							game_with_diff_key_player.append(game_id)
							player_dff.append(player)
							break
					except:
						pass
				break

		# print(game_with_diff_key_player)
		# print(player_dff)
		game_with_diff_key_player_df = pd.DataFrame({"season" : pd.Series([season] * len(game_with_diff_key_player)), "game_id" : pd.Series(game_with_diff_key_player), "PLAYER_NAME" : pd.Series(player_dff)})
		pd.set_option('display.max_rows', None)
		print(game_with_diff_key_player_df)

	efficiency_df = pd.read_csv("data\efficiency_with_O_and_D_and_back_to_back.csv")
	print(efficiency_df.head(30))
	for game_id in game_with_diff_key_player:
		efficiency_df = efficiency_df[efficiency_df["game_id"] != game_id]
	print(efficiency_df.head(30))
	efficiency_df.to_csv("data\efficiency_with_O_and_D_and_without_key_player.csv")
	





if __name__ == "__main__":

	# output_via_season(df)
	# output_game_details_via_season()
	# calculate_each_player_total_attendance_and_min()	
	# add_avg_time_into_games_player_min()
	# drop_minutes_less_than_a_constant(15)
	# ouput_the_key_players_each_season()
	output_game_ids_without_key_player()


	# print(df)






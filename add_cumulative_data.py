import pandas as pd
import numpy as np


df = pd.read_csv("data\efficiency_with_O_and_D_0519.csv")
print(df.head())

def add_cumulative_data(df):
    

    final_df = pd.DataFrame(columns = ['game_id', 'game_date', 'matchup', 'pts_per_game', 'lost_pts_per_game',
       'oppo_pts_per_game', 'oppo_lost_pts_per_game', 'pts', 'pts_oppo',
       'total_real', 'total_bet', 'team_id', 'back_to_back', 'is_home', 'wl',
       'w', 'l', 'w_pct', 'min', 'fgm', 'fgm_cumulative', 'fga',
       'fga_cumulative', 'fg_pct', 'fg_pct_cumulative', 'fg3m',
       'fg3m_cumulative', 'fg3a', 'fg3a_cumulative', 'fg3_pct',
       'fg3_pct_cumulative', 'ftm', 'ftm_cumulative', 'fta', 'fta_cumulative',
       'ft_pct', 'ft_pct_cumulative', 'oreb', 'oreb_cumulative', 'dreb',
       'dreb_cumulative', 'reb', 'reb_cumulative', 'ast', 'ast_cumulative',
       'stl', 'stl_cumulative', 'blk', 'blk_cumulative', 'tov',
       'tov_cumulative', 'pf', 'pf_cumulative', 'a_team_id', 'season_year',
       'season_type', 'season', 'book_name', 'price1', 'price2'])
    for season, season_df in df.groupby("season"):
      print(f"--------------------{season}--------------------")
      # print(season, season_df.head(), sep = "\n")

      for team, team_df in season_df.groupby(df["matchup"].str[:4]):
        print(team, team_df.head(), sep = "\n") 
        count = 0
        cul_fgm = 0
        cul_fga = 0
        cul_fg_pct = 0
        cul_fg3m = 0
        cul_fg3a = 0
        cul_fg3_pct = 0
        cul_ftm = 0
        cul_fta = 0
        cul_ft_pct = 0
        cul_oreb = 0
        cul_dreb = 0
        cul_reb = 0
        cul_ast = 0
        cul_stl = 0
        cul_blk = 0
        cul_tov = 0
        cul_pf = 0
        fgm_list = [0]
        fga_list = [0]
        fg_pct_list = [0]
        fg3m_list = [0]
        fg3a_list = [0]
        fg3_pct_list = [0]
        ftm_list = [0]
        fta_list = [0]
        ft_pct_list = [0]
        oreb_list = [0]
        dreb_list = [0]
        reb_list = [0]
        ast_list = [0]
        stl_list = [0]
        blk_list = [0]
        tov_list = [0]
        pf_list = [0]
        for fgm, fga, fg_pct, fg3m, fg3a, fg3_pct, ftm, fta, ft_pct, oreb, dreb, reb, ast, stl, blk, tov, pf in zip(team_df.fgm, team_df.fga, team_df.fg_pct, team_df.fg3m, team_df.fg3a, team_df.fg3_pct, team_df.ftm, team_df.fta, team_df.ft_pct, team_df.oreb, team_df.dreb, team_df.reb, team_df.ast, team_df.stl, team_df.blk, team_df.tov, team_df.pf): #
          count += 1
          cul_fgm += fgm
          cul_fga += fga
          cul_fg_pct += fg_pct
          cul_fg3m += fg3m
          cul_fg3a += fg3a
          cul_fg3_pct += fg3_pct
          cul_ftm += ftm
          cul_fta += fta
          cul_ft_pct += ft_pct
          cul_oreb += oreb
          cul_dreb += dreb 
          cul_reb += reb 
          cul_ast += ast
          cul_stl += stl
          cul_blk += blk
          cul_tov += tov
          cul_pf += pf

          if count != len(team_df.fgm):
            fgm_list.append(cul_fgm / count)    
            fga_list.append(cul_fga / count)
            fg_pct_list.append(cul_fg_pct / count)
            fg3m_list.append(cul_fg3m / count)
            fg3a_list.append(cul_fg3a / count)
            fg3_pct_list.append(cul_fg3_pct / count)
            ftm_list.append(cul_ftm / count)
            fta_list.append(cul_fta / count)
            ft_pct_list.append(cul_ft_pct / count)
            oreb_list.append(cul_oreb / count)
            dreb_list.append(cul_dreb / count)
            reb_list.append(cul_reb / count)
            ast_list.append(cul_ast / count)
            stl_list.append(cul_stl / count)
            blk_list.append(cul_blk / count)
            tov_list.append(cul_tov / count)
            pf_list.append(cul_pf / count)

        team_df.insert(20, "fgm_cumulative", fgm_list)
        team_df.insert(23, "fga_cumulative", fga_list)
        team_df.insert(25, "fg_pct_cumulative", fg_pct_list)
        team_df.insert(27, "fg3m_cumulative", fg3m_list)
        team_df.insert(29, "fg3a_cumulative", fg3a_list)
        team_df.insert(31, "fg3_pct_cumulative", fg3_pct_list)
        team_df.insert(33, "ftm_cumulative", ftm_list)
        team_df.insert(35, "fta_cumulative", fta_list)
        team_df.insert(37, "ft_pct_cumulative", ft_pct_list)
        team_df.insert(39, "oreb_cumulative", oreb_list)
        team_df.insert(41, "dreb_cumulative", dreb_list)
        team_df.insert(43, "reb_cumulative", reb_list)
        team_df.insert(45, "ast_cumulative", ast_list)
        team_df.insert(47, "stl_cumulative", stl_list)
        team_df.insert(49, "blk_cumulative", blk_list)
        team_df.insert(51, "tov_cumulative", tov_list)
        team_df.insert(53, "pf_cumulative", pf_list)
        team_df.drop(columns = ["Unnamed: 20", "Unnamed: 42"], inplace = True)
        final_df = pd.concat([final_df, team_df])
      print(team_df.columns)
      print(final_df)
      # pd.set_option("display.expand_frame_repr", None)
      # print(df.head(1))


    final_df.to_csv("data\efficiency_with_O_and_D_0531.csv", index = False)





if __name__ == "__main__":
    add_cumulative_data(df)
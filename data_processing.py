import pandas as pd
import numpy as np


class game_df():
    
    def __init__(self, data_path: str):

        self.data = pd.read_csv(data_path)
  
    
    def drop_unused_columns(self):

        self.data.drop(columns = ["GAME_ID", "TEAMID_ID_home", ""], inplace = True)


    def preview(self):

        print(self.data.head())

    def output(self):

        self.data.to_csv("data\games_cleaned.csv")


if __name__ == "__main__":
    
    data_path = "data\games.csv"
    df = game_df(data_path)
    df.drop_unused_columns()

    # df.preview()



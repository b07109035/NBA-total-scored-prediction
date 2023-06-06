import pandas as pd

df = pd.read_csv('data\efficiency_with_O_and_D_0602.csv')
for team, team_df in df.groupby(df['matchup'].str[:4]):

  for year in range(2006, 2018):
    _team_df = team_df[team_df['season_year'] == year]
    train_df = _team_df.head(10)
    test_df  = _team_df.drop(train_df.index)
    print(train_df)
    print(test_df)
    break















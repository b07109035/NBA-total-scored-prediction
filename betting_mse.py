import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import fabs, sqrt


df = pd.read_csv('data\efficiency.csv')
for season, season_df in df.groupby('season_year'):
  avg = 0
  for i in season_df['spread']:
    avg += fabs(i)
  avg = avg / len(season_df['spread'])
  print(season, avg)

for team, team_df in df.groupby(df['matchup'].str[:4]):
  avg = 0
  for i in team_df['spread']:
    avg += fabs(i)
  avg = avg / len(team_df['spread'])
  print(team, avg)

avg = 0
for i in df['spread']:
  avg += fabs(i)
avg = avg / len(df['spread'])
print(avg)


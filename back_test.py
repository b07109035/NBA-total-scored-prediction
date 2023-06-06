import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data\efficiency_with_O_and_D_0602.csv')
predicted_list = []
for betting_price, total_real in zip(df['total_bet'], df['total_real']):
    if betting_price > total_real:
        predicted_list.append('under')
    else:
        predicted_list.append('over')
df['predicted'] = predicted_list
df.to_csv('data\efficiency_with_O_and_D_0602.csv')

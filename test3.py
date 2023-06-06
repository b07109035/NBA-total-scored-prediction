import pandas as pd
import matplotlib.pyplot as plt

equity = 1000
equity_list = []
for yr in range(2007, 2017):

    '''
    result = []
    df = pd.read_csv(f'data\{yr}.csv')
    for i, j, k in zip(df['total_bet'], df['total_real'], df['predicted']):
      if j > i and k > i:
        result.append('win')
      elif j < i and k < i:
        result.append('win')
      else:
        result.append('lose')

    df['result'] = result 
    if yr != 2016:
      df.drop(df.tail(20).index, inplace=True)
    # print(df['result'].value_counts())
    print(f'===================={yr}=======================')
    # print(df)
    # df.to_csv(f'data\{yr}.csv')
    '''

    df = pd.read_csv(f'data\{yr}.csv')
    if yr != 2014 and yr != 2015:
      for i in df['result']:
        if i== 'win':
          equity += 91
        if i == 'lose':
          equity -= 100
        equity_list.append(equity)
      # print(equity_list)
      # print([i for i in range(1, len(equity_list) + 1)])
      # break
      print(df['result'].value_counts())

plt.plot([i for i in range(1, len(equity_list) + 1)], equity_list, label = yr)
plt.show()
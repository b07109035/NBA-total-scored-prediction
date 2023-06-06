import pandas as pd

def dta_to_csv(path: str):
  data = pd.read_stata(f"{path}.dta", convert_categoricals = False)
  data.to_csv(f"{path}.csv")

if __name__ == "__main__":
  dta_to_csv()
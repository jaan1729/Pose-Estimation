from DataGen import transform_and_save
import pandas as pd

def main():
  path = "data/train"
  train = pd.read_csv("data/train.csv")
  transform_and_save(path, train)
  print('completed saving input files')


main()

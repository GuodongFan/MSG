import pandas as pd
import numpy as np


data_df = pd.read_csv('./data/relation2.csv')
data_df.replace

train_df = data_df.sample(frac=0.8,random_state=0,axis=0)
other_df = data_df[~data_df.index.isin(train_df.index)]

train_df.to_csv('./train.csv')
other_df.to_csv('./test.csv')


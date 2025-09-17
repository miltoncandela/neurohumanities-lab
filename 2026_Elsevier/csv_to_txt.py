import os
import pandas as pd

# 	Timestamp	Channel_1	Channel_2	Channel_3	Channel_4	Channel_5	Channel_6	Channel_7	Channel_8	Acc_1	Acc_2	Acc_3
for file in os.listdir('data/csv'):
    df = pd.read_csv('data/csv/{}'.format(file), index_col=0)
    df = df.drop(['Timestamp', 'Acc_1', 'Acc_2', 'Acc_3'], axis=1).dropna(axis=0)

    df.to_csv('data/txt/{}.txt'.format(file[:-4]), header=False, index=False)

import os, sys
import pandas as pd
from model_pred import LabelHarassment

def main():
    file = sys.argv[1]
    dirname, path = os.path.split(file)
    print("fsdfsdf", dirname, path)
    df = pd.read_csv(file, names=[])
    new_df = LabelHarassment(df)
    new_df.to_csv(f'{path}_labeled', index=False)


if __name__ == '__main__':
    file = sys.argv[1]
    print(len(sys.argv))
    dirname, path = os.path.split(file)
    print(dirname, path)

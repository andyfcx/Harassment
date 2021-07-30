import os
import pandas as pd
from utils import context_clean, reason_normalize, reserve
from xgboost import XGBClassifier

def chg_sep(file):
    print(f"[start] ',' -> ';' {file}")
    try:
        with open(f"./data/to_predict/{file}") as f:
            txt = f.read()
        txt = txt.replace("','","';'")

        with open(f"./data/to_predict/{file}", 'w') as f:
            f.write(txt)
    except Exception as e:
        print(f"[fail] {file}, {e}")
        pass
    
# [fail] 司法院－刑事補償.csv, 'utf-8' codec can't decode byte 0x8b in position 1: invalid start byte


class ProcessCsv():

    def __init__(self, file_path):
        df = pd.read_csv(file_path, sep=';', names=['court', 'datetime', 'case_number', 'accuse_', 'reason_'])
        df['reason'] = df.reason_.apply(context_clean)
        df.accuse_.apply(reason_normalize).apply(reserve) # Convert accuse into 0, 0.5, 1
        self.df_new = df.drop(['court', 'datetime', 'case_number', 'acccuse_', 'reason'])

        return self.df_new

    def word_seg(self):
        pass

    def sentiment(self):
        

    def load_xgbc_model(self):
        self.xgbc = XGBClassifier()
        self.xgbc.load_model(self.xgbc_model_path)

    def fit(self):
        return self.xgbc.fit(self.df_new[['reason']])

def main():
    csv_files = os.listdir("./data/to_predict")
    for csv in csv_files:
        chg_sep(csv)

if __name__ == "__main__":
    # csv_files = os.listdir("./data/to_predict")
    # print(csv_files)
    main()
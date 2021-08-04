import os
import pandas as pd
from utils import clean_context, reason_normalize, reserve, get_sentiments, get_summary
from xgboost import XGBClassifier


def chg_sep(file):
    print(f"[start] ',' -> ';' {file}")
    try:
        with open(f"./data/to_predict/{file}") as f:
            txt = f.read()
        txt = txt.replace("','", "';'")

        with open(f"./data/to_predict/{file}", 'w') as f:
            f.write(txt)
    except Exception as e:
        print(f"[fail] {file}, {e}")
        pass

# [fail] 司法院－刑事補償.csv, 'utf-8' codec can't decode byte 0x8b in position 1: invalid start byte


class LabelHarassment():
    # sentence/ : accuse,reason,sentence
    # to_predict/ : 'court', 'datetime', 'case_number', 'accuse_', 'reason_'
    def __init__(self, file_path):
        _, fname = os.path.split(file_path)
        self.xgbc = XGBClassifier()
        self.xgbc_model_path = "trained_model/m1.model"
        self.load_xgbc_model()
        # df = pd.read_csv(file_path, sep=';', names=['court', 'datetime', 'case_number', 'accuse_', 'reason_'])
        self.df = pd.read_csv(file_path, sep=',')
        self.prefit()
        self.predict(fname)

        # return self.df_new

    def load_xgbc_model(self):
        self.xgbc.load_model(self.xgbc_model_path)

    def prefit(self):
        self.df['label'] = self.df.accuse.apply(lambda x: 0 if x == 0 else x)

    def predict(self, fname):
        for _, row in self.df_new[self.df_new.label != 0].iterrows():
            row['label'] = self.xgbc.predict(row[['accused', 'combined_sentiments']])

        self.df.to_csv(f"result/{fname}", index=False)

def run_all():
    csv_files = os.listdir("./data/sentence")
    for csv in csv_files:
        LabelHarassment(csv)

if __name__ == "__main__":
    main()

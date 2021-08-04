import os
import pandas as pd
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


def accuse_quick_conversion(accuse):
    if accuse==0:
        return 0
    elif accuse==1:
        return 1
    else:
        return accuse

class LabelHarassment():
    # sentence/ : accuse,reason,sentence
    # to_predict/ : 'court', 'datetime', 'case_number', 'accuse_', 'reason_'
    def __init__(self, file_path):
        _, fname = os.path.split(file_path)
        self.xgbc = XGBClassifier()
        self.xgbc_model_path = "trained_model/m1.model"
        self.load_xgbc_model()
        self.df = pd.read_csv(file_path, sep=',')
        self.prefit()
        self.predict(fname)

    # @classmethod
    # def dfinit(cls, df, fname):
    #     cls.file_path = fname
    #     cls.df = df
    #     cls.__init__(fname)


    def load_xgbc_model(self):
        self.xgbc.load_model(self.xgbc_model_path)

    def prefit(self):
        """
        If accused is 0, apply it to label, else, remain label undetermined
        """
        self.df['label'] = self.df.accused.apply(accuse_quick_conversion)

    def predict(self, fname):
        """Predict using XGBClassifier, exclude pre-labeled 0 and 1"""
        self.df.loc[(self.df.label != 0) & (self.df.label != 1), 'label'] = self.xgbc.predict(self.df[(self.df.label != 0) & (self.df.label != 1)][['accused', 'combined_sentiments']])

        self.df.to_csv(f"result/{fname}", index=False)
        print(f"[Pred] Saved to result/{fname}")

def run_all():
    csv_files = os.listdir("./data/preprocessed")
    for csv in csv_files:
        LabelHarassment(csv)

# df1.loc[:,'label'][df1.label != 0] = xgbc.predict(df1[df1.label!=0][['accused', 'combined_sentiments']])
# df.loc[df.label != 0 & df.label != 1 , 'label'] = xgbc.predict(df[df.label != 0 & df.label != 1][['accused', 'combined_sentiments']])

if __name__ == "__main__":
    n = LabelHarassment("data/preprocessed/臺灣澎湖地方法院.csv")
import os
import pandas as pd
from xgboost import XGBClassifier
from configs import XGB_MODEL_FILE, RAW_DATA_DIR, OUTPUT_DIR

def chg_sep(file):
    print(f"[start] ',' -> ';' {file}")
    try:
        with open(f"./{RAW_DATA_DIR}{file}") as f:
            txt = f.read()
        txt = txt.replace("','", "';'")

        with open(f"./{RAW_DATA_DIR}{file}", 'w') as f:
            f.write(txt)
    except Exception as e:
        print(f"[fail] {file}, {e}")
        pass

# [fail] 司法院－刑事補償.csv, 'utf-8' codec can't decode byte 0x8b in position 1: invalid start byte


def accuse_quick_labeling(accuse):
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
        print(f"ORIGINAL INIT")
        _, self.fname = os.path.split(file_path)
        self.xgbc = XGBClassifier()
        self.xgbc_model_path = XGB_MODEL_FILE
        self.load_xgbc_model()
        if file_path:
            self.df = pd.read_csv(file_path, sep=',')
        self.prefit()
        self.predict(self.fname)

    @classmethod
    def init_from_df(cls, df, file_path):
        lh = cls.__new__(cls)
        lh.xgbc = XGBClassifier()
        lh.xgbc_model_path = XGB_MODEL_FILE
        lh.load_xgbc_model()
        _, fname = os.path.split(file_path)
        lh.df = df
        lh.prefit()
        lh.predict(fname)

    def load_xgbc_model(self):
        self.xgbc.load_model(self.xgbc_model_path)

    def prefit(self):
        """
        If accused is 0, apply it to label, else, remain label undetermined
        """
        self.df['label'] = self.df.accused.apply(accuse_quick_labeling)

    def predict(self, fname):
        """Predict using XGBClassifier, exclude pre-labeled 0 and 1"""
        self.df.loc[(self.df.label != 0) & (self.df.label != 1), 'label'] = self.xgbc.predict(self.df[(self.df.label != 0) & (self.df.label != 1)][['accused', 'combined_sentiments']])

        self.df.to_csv(f"{OUTPUT_DIR}{fname}", index=False)
        print(f"[Pred] Saved to {OUTPUT_DIR}{fname}")

def run_all():
    csv_files = os.listdir(RAW_DATA_DIR)
    for csv in csv_files:
        LabelHarassment(csv)

# df1.loc[:,'label'][df1.label != 0] = xgbc.predict(df1[df1.label!=0][['accused', 'combined_sentiments']])
# df.loc[df.label != 0 & df.label != 1 , 'label'] = xgbc.predict(df[df.label != 0 & df.label != 1][['accused', 'combined_sentiments']])

if __name__ == "__main__":
    n = LabelHarassment(f"{RAW_DATA_DIR}臺灣澎湖地方法院.csv")
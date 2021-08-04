from file_preprocessing import read_csv
from model_pred import LabelHarassment

read_csv("data/to_predict/臺灣橋頭地方法院.csv")
LabelHarassment("data/preprocessed/臺灣橋頭地方法院.csv")
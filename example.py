from file_preprocessing import read_csv
from model_pred import LabelHarassment
from configs import RAW_DATA_DIR, PREPROCESSED_DIR

df = read_csv(f"{RAW_DATA_DIR}福建金門地方法院.csv")
# LabelHarassment(f"{PREPROCESSED_DIR}臺灣南投地方法院.csv")
LabelHarassment.init_from_df(df, file_path="福建金門地方法院.csv")
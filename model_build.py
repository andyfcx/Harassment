from sklearn.model_selection import train_test_split, KFold
# from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import pandas as pd
# from ckiptagger import WS
from snownlp import SnowNLP, sentiment
from utils import conv_yn, reason_normalize, reserve, context_clean

# ws = WS("./data")

def ckip_words(text):
    ws_results = ws([text])
    return ws_results[0]

def get_sentiments(s):
    sentiments_of_emptystring = 0.5262327818078083
    try:
        ss = SnowNLP(s)
        sen = ss.sentiments
    except ZeroDivisionError:
        sen = sentiments_of_emptystring
    return (sen - sentiments_of_emptystring)/(1-sentiments_of_emptystring)

def get_summary(s):
    try:
        ss = SnowNLP(s)
        return ss.summary(3)
    except ZeroDivisionError:
        return []


def preprocessing():
    df = pd.read_csv("data/follow.csv")
    df.label_.fillna('N', inplace=True)
    df.犯罪事實.fillna(" ", inplace=True)
    df.理由.fillna(" ", inplace=True)
    #犯罪事實 理由要filter出關鍵字？
    df['judge'] = df.主文.apply(get_sentiments)
    df['facts'] = df.犯罪事實.apply(get_sentiments)
    df['reason'] = df.理由.apply(get_sentiments)
    df['label'] = df.label_.apply(conv_yn)
    df['accused'] = df.案由.apply(reason_normalize).apply(reserve)
    df.drop(columns=['主文','犯罪事實','理由','案例','案由','法院','案號','label_'], axis=1, inplace=True)
    return df


# x_train, x_test, y_train, y_test = train_test_split( ,df['label'], test_size=0.25, random_state=42)

# xgbc = XGBClassifier()
# xgbc.fit(x_train, y_train)
# xgbc.save_model()


# cv = KFold(n_splits=5, shuffle=True, random_state=100)
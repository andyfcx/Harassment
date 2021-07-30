from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBClassifier
import pandas as pd
# from ckiptagger import WS
from snownlp import SnowNLP, sentiment
from utils import conv_yn, reason_normalize, reserve, context_clean


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


def annotation(s):
    if (s == 'N' or not s):
        return 0

    else:
        return 1


def preprocessing():
    """Process file for training"""
    df = pd.read_csv("data/follow.csv")
    df.label_.fillna('N', inplace=True)
    df.犯罪事實.fillna(" ", inplace=True)
    df.理由.fillna(" ", inplace=True)

    df['judge'] = df.主文.apply(get_sentiments)
    df['facts'] = df.犯罪事實.apply(get_sentiments)
    df['reason'] = df.理由.apply(get_sentiments)
    df['label'] = df.label_.apply(conv_yn)
    df['accused'] = df.案由.apply(reason_normalize).apply(reserve)
    df['annotation'] = df.備註.apply(annotation)

    df_parsed = df.drop(columns=['主文', '犯罪事實', '理由', '案例',
                                 '案由', '法院', '案號', 'label_', '備註'],
                        axis=1)
    df_parsed.to_csv("data/follow_matrix.csv", index=False)
    return df_parsed


def preprocessing_combined():
    """Process file for training"""
    df = pd.read_csv("data/follow.csv")
    df.label_.fillna('N', inplace=True)
    df.犯罪事實.fillna(" ", inplace=True)
    df.理由.fillna(" ", inplace=True)

    df['combine'] = df.apply(
        lambda x: f"{x.主文} {x.犯罪事實} {x.理由}", axis=1).apply(context_clean).apply(get_sentiments)
    df['label'] = df.label_.apply(conv_yn)
    df['accused'] = df.案由.apply(reason_normalize).apply(reserve)

    df['annotation'] = df.備註.apply(annotation)

    df_parsed = df.drop(columns=['主文', '犯罪事實', '理由', '案例',
                                 '案由', '法院', '案號', 'label_', '備註'],
                        axis=1)
    df_parsed.to_csv("data/follow_matrix.csv", index=False)
    return df_parsed


df_parsed = preprocessing_combined()
# x_train, x_test, y_train, y_test = train_test_split(df_parsed[['judge', 'facts', 'reason', 'accused']],
#                                                     df_parsed['label'], test_size=0.25, random_state=42)

x_train, x_test, y_train, y_test = train_test_split(df_parsed[['combine', 'accused']],
                                                    df_parsed['label'], test_size=0.25, random_state=42)


xgbc = XGBClassifier(learning_rate=0.5)
xgbc.fit(x_train, y_train)
# xgbc.save_model()


# cv = KFold(n_splits=5, shuffle=True, random_state=100)
print(xgbc.score(x_test, y_test))

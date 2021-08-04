from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
from file_preprocessing import get_sentence
# from ckiptagger import WS
from utils import conv_yn, reason_normalize, reserve, clean_context, get_sentiments


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
    df['annotation'] = df.備註.apply(annotation) # 此為無效指標

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

    # 將主文、犯罪事實、理由放在一起
    df['combined_sentiments'] = df.apply(
        lambda x: f"{x.主文} {x.犯罪事實} {x.理由}", axis=1).apply(clean_context).apply(get_sentiments)
    df['label'] = df.label_.apply(conv_yn)
    df['accused'] = df.案由.apply(reason_normalize).apply(reserve)

    df['annotation'] = df.備註.apply(annotation)

    df_parsed = df.drop(columns=['主文', '犯罪事實', '理由', '案例',
                                 '案由', '法院', '案號', 'label_', '備註'],
                        axis=1)
    df_parsed['sentences'] = df.apply(
        lambda x: f"{x.主文} {x.犯罪事實} {x.理由}", axis=1).apply(clean_context).apply(get_sentence)
    df_parsed.to_csv("data/follow_matrix.csv", index=False)
    return df_parsed


df_parsed = preprocessing_combined()
# x_train, x_test, y_train, y_test = train_test_split(df_parsed[['judge', 'facts', 'reason', 'accused']],
#                                                     df_parsed['label'], test_size=0.25, random_state=42)

x_train, x_test, y_train, y_test = train_test_split(df_parsed[['combined_sentiments', 'accused']],
                                                    df_parsed['label'], test_size=0.25, random_state=42)

df_parsed.to_csv("data/follow_combined.csv", index=False)
xgbc = XGBClassifier(learning_rate=0.6)
xgbc.fit(x_train, y_train)
xgbc.save_model('m1.model')


# cv = KFold(n_splits=5, shuffle=True, random_state=100)
print(xgbc.score(x_test, y_test))

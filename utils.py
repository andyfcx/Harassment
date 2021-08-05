import re
import pandas as pd
from snownlp import SnowNLP
from zhon.hanzi import punctuation
from gensim.models.word2vec import Word2Vec
import jieba

with open("data/reserved") as f:
    reserved = [l.rstrip("\n") for l in f]
with open("data/excluded") as f:
    excluded = [l.rstrip("\n") for l in f]
with open("data/excluded_reason") as f:
    excluded_reason = [l.rstrip("\n") for l in f]

model = Word2Vec.load("data/wiki_w2v_100/wiki_verdict_add_words.model")


def read_csv_to_fit(file_path):
    df = pd.read_csv(file_path, names=['court', 'datetime', 'case_number', 'accuse_', 'reason_'])
    df['reason'] = df.reason_.apply(clean_context)  # Clean text

    df['accuse'] = df.accuse_.apply(reason_normalize).apply(reserve)  # Convert accuse into 0, 0.5, 1

    df.drop(columns=['court, datetime, case_number, accuse_', 'reason_'], axis=1, inplace=True)
    return df


def clean_context(s: str):
    return re.sub('</br>|\s{1,}|\n|\'', '', s)


def clean_punctuation(s):
    return re.sub(f"[{punctuation}]", ' ', s)


def reason_normalize(s: str):
    return re.sub("\n|等$|違反|罪$", '', s)


def conv_yn(s):
    if s == 'Y':
        return 1
    elif s == 'N':
        return 0


def reserve(s):
    if s in reserved:
        return 1
    elif s in excluded:
        return 0
    else:
        for r in reserved:
            if r in s:
                return 1
        for e in excluded:
            if e in s:
                return 0
        return max_similarity(s)


def max_similarity(s):
    """Calculate the max similarity in word_list, if the word is too long, seg it."""
    word_list = reserved  # 目前僅先處理保留字
    if len(s) > max(len(item) for item in word_list):
        return max(max_similarity(w) for w in seg(s))

    if model.wv.key_to_index.get(s, ''):
        try:
            return max([model.wv.similarity(e, s) for e in word_list])
        except:
            return 0.3
    else:
        return 0.3


def seg(s):
    return jieba.cut(s, cut_all=False)


def get_sentiments(s):
    sentiments_of_emptystring = 0.5262327818078083
    try:
        ss = SnowNLP(s)
        sen = ss.sentiments
    except ZeroDivisionError:
        sen = sentiments_of_emptystring
    return (sen - sentiments_of_emptystring) / (1 - sentiments_of_emptystring)


def get_summary(s):
    try:
        ss = SnowNLP(s)
        return ss.summary(3)
    except ZeroDivisionError:
        return []
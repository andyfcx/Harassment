import pandas as pd
import os, sys
from gensim.models.word2vec import Word2Vec
from snownlp import SnowNLP
from time import time
from utils import clean_context, get_sentiments, reason_normalize, reserve

with open("data/excluded") as f:
    excluded = [l.rstrip("\n") for l in f]

with open("data/reserved") as f:
    reserved = [l.rstrip("\n") for l in f]
vocab_list = excluded + reserved

def clean_words(lst):
    cleaned = []
    for w in lst:
        if len(w) > 1:
            try:
                int(w[0])
            except ValueError:
                cleaned.append(w)
    return cleaned


def parse_follow():
    df = pd.read_csv("./data/follow_ws.csv")
    df.crime_kw = df.crime_kw.apply(eval).apply(clean_words)
    df.reason_kw = df.reason_kw.apply(eval).apply(clean_words)

    df.crime_kw = df.crime_kw.apply(clean_words)
    df.reason_kw = df.reason_kw.apply(clean_words)
    return df


# corpus = pd.concat([df.crime_kw, df.reason_kw])

def get_sentence(paragraph):
    try:
        ss = SnowNLP(paragraph)
        return ss.sentences[1:-1]
    except ZeroDivisionError:
        return []


def read_csv(file_path):
    """Read csv as df and get all sentences, add to outer global variable sentences"""
    t0 = time()
    _, fn = os.path.split(file_path)
    print(f"[File] Loading {file_path}, {os.path.getsize(file_path)/1024/1024} MB")
    df = pd.read_csv(file_path, names=['court', 'datetime', 'case_number', 'accused_', 'reason_'], sep=';')
    df.reason_.fillna(" ", inplace=True)
    df['reason'] = df.reason_.apply(clean_context)  # Clean text
    df['accused'] = df.accused_.apply(reason_normalize).apply(reserve) # 案由數字化處理
    df['combined_sentiments'] = df.reason.apply(get_sentiments)
    # df.drop(columns=['court', 'datetime', 'case_number', 'reason_'], axis=1, inplace=True) # Do not drop accuse
    df.drop(columns=['accused_','reason_'], axis=1, inplace=True)  # Do not drop accuse, case_number
    # df['sentence'] = df.reason.apply(get_sentence)
    print(df.columns)
    df.to_csv(f"./data/preprocessed/{fn}", index=False)
    print(f"[File] Saved to data/preprocessed/{fn} by read_csv()")
    t1 = time()
    print(f"[Time] Used {t1 - t0} sec")
    return df
    # return df.sentence.sum()


def try_one():
    read_csv('./data/to_predict/臺灣澎湖地方法院.csv')

def train_new(vocab_list, sentences):
    t0 = time()
    model_save_path = "./data/wiki_w2v_100/wiki_verdict.model"
    print("[W2V] Loading pre-trained model")
    model = Word2Vec.load("./data/wiki_w2v_100/wiki2019tw_word2vec_Skip-gram_d100.model")
    
    print("[W2V] Start training ...")
    model.min_count = 1
    model.build_vocab([vocab_list], update=True)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

    t1 = time()
    print(f"[W2V] Training finished :{t1 - t0}. Save model to: {model_save_path}")

    model.save(model_save_path)


def main():

    csv_file = os.listdir('./data/to_predict') # sep=';'
    sentences = []
    for f in csv_file:
        read_csv(f'./data/to_predict/{f}')
        # sentences += s
        print(f"[data] Sentences is now of {len(sentences)}, Mem: {sys.getsizeof(sentences)/1024/1024} MB")

    # train_new(vocab_list, sentences)


if __name__ == "__main__":
    # main()
    try_one()
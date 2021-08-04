from gensim.models.word2vec import Word2Vec
from typing import List

def check_sim(lst: List[str]):
    v = sum([model.wv[word] for word in lst])
    print(model.wv.most_similar(positive=[v]))
    s = [f'model.wv["{word}"]' for word in lst]
    print('+'.join(s))
    # return v

# https://drive.google.com/uc?export=download&id=1ZJui-EuBUx-NLy9NjQy_peWo5NX2dxCY
model = Word2Vec.load("data/wiki_w2v_100/wiki2019tw_word2vec_Skip-gram_d100.model")

word_list = {
    "妨害自由": model.wv["拘束"] + model.wv["強制罪"],
    "妨害名譽": model.wv['侮辱'],
    "社會秩序維護法": model.wv["社會"] + model.wv["秩序"] + model.wv["維護"] + model.wv["法律"],
    "性騷擾防治法": model.wv["性騷擾"] + model.wv["防治"] + model.wv["法律"],
    "性別工作平等法": model.wv["性別"] + model.wv["平等"] + model.wv["工作"] + model.wv["法律"],
    "性別平等教育法": model.wv["性別"] + model.wv["平等"] + model.wv["教育"] + model.wv["法律"],
    "損害賠償": model.wv["損害"] + model.wv["賠償"],
    "侵權行為損害賠償": model.wv["侵權"] + model.wv["損害"] + model.wv["賠償"],
    "保護令": model.wv["家暴"] + model.wv["防治"] + model.wv["保護"],
    "家庭暴力": model.wv['家暴'],
    "家庭暴力防治法": model.wv['家暴'],
    "偽造文書": model.wv["虛假"] + model.wv["文書"],
    "妨害電腦使用": model.wv['黑客'],
    "給付資遣費": model.wv['資遣費'],
    "個人資料保護法": model.wv['個資法'],
    "妨害性自主": model.wv['性侵'],
    "竊佔": model.wv['偷竊'],
    "強制猥褻": model.wv['性侵'],
    "妨害公務": model.wv["妨害"] + model.wv["公共"] + model.wv["勤務"],
    "毒品危害防制條例": model.wv['毒品'],
    "偽證": model.wv["虛假"] + model.wv["證據"],
    "殺人未遂": model.wv["未遂"] + model.wv["殺人"],
    "違反政府採購法": model.wv["政府"] + model.wv["採購法"],
    "公共危險": model.wv["公共"] + model.wv['危險']}

for k, v in word_list.items():
    try:
        model.wv.add_vector(k, v)
    except AttributeError:
        # 此處遇到 https://github.com/RaRe-Technologies/gensim/issues/3114 gensim版本不相容問題
        from gensim.models import keyedvectors
        model.wv.save_word2vec_format("data/w2v_port_3.8")
        wv = keyedvectors.load_word2vec_format("data/w2v_port_3.8")
        wv.add_vector(k, v)
        model.wv = wv

model.save("data/wiki_w2v_100/wiki_verdict_add_words.model")

import re
import pandas as pd

def read_csv_to_fit(file_path):
    """Preprocessing the csv files, make them """
    df = pd.read_csv(file_path, names=['court', 'datetime', 'case_number', 'accuse_', 'reason_'])
    df['reason'] = df.reason_.apply(context_clean) # Clean text
    
    df['accuse'] = df.accuse_.apply(reason_normalize).apply(reserve) # Convert accuse into 0, 0.5, 1

    df.drop(columns=['court, datetime, case_number, accuse_', 'reason_'], axis=1, inplace=True)
    return df

def context_clean(s:str):
    return re.sub('</br>|\s{1,}|\n','', s)

def reason_normalize(s:str):
    return re.sub("\n|等$|違反|罪$", '', s)

def conv_yn(s):
    if s=='Y':
        return 1
    elif s=='N':
        return 0

def to_num(g):
    u = ['壹','貳','參','肆','伍','陸','柒','捌','玖','拾','廿','卅','一','二','三','四','五','六','七','八','九','十',0,'１','２','３','４','５','６','７','８','９','０','佰','仟','萬','百','千']
    n = [1,2,3,4,5,6,7,8,9,10,20,30,1,2,3,4,5,6,7,8,9,10,0,1,2,3,4,5,6,7,8,9,0,100,1000,10000,100,1000]
    cv = dict(zip(u,n))
    return cv[g]

def reserve(s):
    with open("data/reserved") as f:
        reserved = [l.rstrip("\n") for l in f]
    with open("data/excluded") as f:
        excluded = [l.rstrip("\n") for l in f]

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
        return 0.5 

def nds(y):
    if bool(re.search('[壹貳參肆伍陸柒捌玖拾廿卅佰百仟千萬一二三四五六七八九十１２３４５６７８９０]', y)):
        if len(y) == 1:
            return to_num(y)

        elif len(y) == 2:
            if y[1]=='拾':
                return to_num(y[0])*to_num(y[1])

            elif y[0] =='拾' or y[0] =='廿' or y[0]=='卅':
                return to_num(y[0])+to_num(y[1])

            else:
                return to_num(y[0])*10 + to_num(y[1])

        elif len(y) == 3:
            return to_num(y[0])*to_num(y[1])+to_num(y[2])

        else:
            return to_num(y)
    else:
        return 0

def to_days(x):
    if '年' in x or '月' in x or '日' in x:
        try:
            y = x.index("年")
        except:
            y = 0
        try:
            m = x.index("月")
        except:
            m = 0
        try:
            d = x.index("日")
        except:
            d = 0

        if y>0:
            year = x[0:y]

            if m>0:
                month = x[0+y+1:m]

                if d>0:
                    day = x[0+m+1:d]
                else:
                    day = ""
            else:
                month = ""

                if d>0:
                    day = x[0+y+1:d]
                else:
                    day = ""
        else:
            year = ""

            if m>0:
                month = x[0:m]

                if d>0:
                    day = x[0+m+1:d]

                else:
                    day =""
            else:
                month = ""
                day = x[0:d]

        res = nds(year)*365+nds(month)*30+nds(day)
        return res

    else:
        return '＊'+x

def to_money(x):
    x = x.replace('千','仟').replace('百','佰')
    if '萬' in x or '仟' in x or '佰' in x:
        try:
            w = x.index("萬")
        except:
            w = 0
        try:
            t = x.index("仟")
        except:
            t = 0
        try:
            h = x.index("佰")
        except:
            h = 0

        if w>0:
            wan = x[0:w]

            if t>0:
                tho = x[0+w+1:t]

                if h>0:
                    hun = x[0+t+1:h]
                else:
                    hun = ""
            else:
                tho = ""

                if h>0:
                    hun = x[0+w+1:t]
                else:
                    hun = ""
        else:
            wan = ""

            if t>0:
                tho = x[0:t]

                if h>0:
                    hun = x[0+t+1:h]

                else:
                    hun =""
            else:
                tho = ""
                hun = x[0:h]

        res = nds(wan)*10000+nds(tho)*1000+nds(hun)*100
        return res
    else:
        return '＊'+x
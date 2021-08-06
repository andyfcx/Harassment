## Quick start

`python3 process_csv.py data/to_predict/臺灣澎湖地方法院.csv`
Or check `example.py`

## 使用工具
 - SnowNLP: 提供情感分析、文章摘要、切詞
 - XGBoost: 分類
 - gensim: Word2Vec
    1. 使用預訓練過的模型：由此下載 https://drive.google.com/uc?export=download&id=1ZJui-EuBUx-NLy9NjQy_peWo5NX2dxCY
    2. finetune_model.py 將沒有在模型中但必要的字詞加進去，因為主要使用相似值，故向量本身的norm不影響

## 欄位名稱說明
 - 案由: accused_, 數值化處理後為accused
 - 案號: case_number
 - 判決理由: reason
 - 犯罪事實: facts
 - 判決書情義分數: combined_sentiments 
 - 是否與跟騷有關: label

## 新增feature的處理
如果需要新增欄位作為新的feature，需要處理：
 - model_train.py : preprocessing_combined
 - model_pred.py: class LabelHarassment
加上相對應的新欄位重新訓練XGBClassifier，然後用新的model去predict

## 檔案目錄
 - data/to_predict 原始資料
 - data/preprocessed 預處理後中間存檔，將','換成';'並且用;來當csv分隔符號
 - result/ 加上label後的結果
 
## 判斷流程參考
1. 先 filter 案由
    1. 可保留的：
        1. 妨害自由
        2. 妨害名譽
        3. 社會秩序維護法
        4. 性騷擾防治法
        5. 性別工作平等法
        6. 性別平等教育法
        7. 損害賠償
        8. 侵權行為損害賠償
        9. 違反保護令
        10. 家庭暴力罪
        11. 家庭暴力防治法
        12. 偽造文書
        13. 恐嚇
        14. 妨害電腦使用
        15. 解聘
        16. 退學
        17. 給付資遣費
        18. 個人資料保護法
            
    2. 可過濾掉的
        1. 妨害性自主
        2. 傷害
        3. 竊佔
        4. 強制猥褻
        5. 賭博
        6. 詐欺
        7. 竊盜
        8. 妨害公務
        9. 毒品危害防制條例
        10. 偽證
        11. 殺人
        12. 殺人未遂
        13. 違反政府採購法等
        14. 毀損
        15. 公共危險
            
2. 再 filter 事實以及理由，可能要強調哪些行為（關鍵字）被辨識出來時是要保留的，需要討論一下怎麼做才能符合需求


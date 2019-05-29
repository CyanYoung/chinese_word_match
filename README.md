## Chinese Word Match 2018-7

#### 1.preprocess

prepare() 将按类文件保存的数据汇总、去重，去除停用词、统一替换特殊词

#### 2.build

link_fit() 建立 word_sent 字典，freq_fit() 通过 bow、svd 建立 sent_vec 字典

#### 3.match

通过 word_sent 按字查找共现的句索引，根据同音、同义字典扩充

edit_predict() 使用拼音的编辑距离比、cos_predict() 使用余弦距离进行匹配

## Chinese Intent Match 2018-7

#### 1.preprocess

prepare() 将按类文件保存的数据汇总、去重，去除停用词、统一替换特殊词

#### 2.build

link_fit() 建立 class2word 字典，实现类、字、句索引的两层映射

freq_fit() 训练各类的 tfidf 模型，建立 ind2vec 字典，实现句索引、句向量的映射

#### 3.match

通过 class2word 按字查找共现的句索引，包括同音、同义字

edit_predict() 定义编辑距离系数 edit_dist / len(phon)，将字转换为拼音

cos_predict() 使用余弦相关系数，通过各类的 tfidf 模型分别得到句向量进行匹配
### 任务
1. 熟悉数据集
2. 数据清洗
3. 构建词典

#### data_clean.py:
1. get_symbols: 返回特殊符号集合
2. remove_words: 返回去除'|', '[', ']', '图片', '语音', ' '（空格）后的输入
3. segment：分词
4. parse_data：读取csv文件，去除Report字段缺失的样本、用空字符串填充其余缺失字段、拼接Question和Dialogue字段
5. save_data: 将4的结果进行分词、清洗，转换成空格分隔的字符串输出到相应文件

#### getDictionary.py:
1. save_word_dict：将单词、序号写入文件
2. read_data：读取data_clean.py中得到的3个文件
3. build_dictionary：按词频或原始顺序创建单词/序号字典
# Word2Vec

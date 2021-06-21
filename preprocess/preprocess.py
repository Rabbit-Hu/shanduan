import json
import os
import re
import glob
import torch
from tqdm import tqdm

# 感兴趣的标点符号列表：
punctuations_list = list("，。、？！：；")
# 要忽略（变为空字符）的字符列表：
ignore_characters = ['[', ']', '（', '）', '\n', '\r', ' ', '「', '」', '『', '』', ' ', '　', '”', '“']
# 要转换的字符：
character_trans_list = []
character_trans = str.maketrans(dict(character_trans_list + [(c, '') for c in ignore_characters]))
remove_punc_trans = str.maketrans(dict([(c, '') for c in punctuations_list]))

def preprocess(split):

    data_root = "./data" # TODO: change back!

    input_root = os.path.join(data_root, "txt_files/", split)
    text = ""
    for file_path in glob.glob(os.path.join(input_root, '*.txt')):
        with open(file_path, 'r') as f:
            file_content = f.read()
            print(f"length of {file_path.split('/')[-1]}: {len(file_content)}")
            text += file_content
    print("total length of text:", len(text))
    # return
    # text = "尧曰：「谁可顺此事？」放齐曰：「嗣子丹朱（开明）。」尧曰：『吁！顽凶，不用。』尧又曰：「谁可者？」欢兜曰：「共工旁聚布功，可用。」"
    # print(text[:1000], end='\n\n')
    text = text.translate(character_trans)
    # print(text[:5000])
    
    ''' 统计所有的标点组合种类 '''
    puncs_count = {}
    last_puncs = ""
    for c in text:
        if c in punctuations_list:
            last_puncs += c
        elif last_puncs != "":
            puncs_count[last_puncs] = puncs_count.get(last_puncs, 0) + 1
            last_puncs = ""
    if last_puncs != "":
        puncs_count[last_puncs] = puncs_count.get(last_puncs, 0) + 1
    # print("All punctuations:", puncs_count)
        
    ''' 多重标点只保留第一个（如：史记[24352:] 有两个连续的句号，为删除右引号所致。） '''
    for s in puncs_count:
        if len(s) > 1:
            text = text.replace(s, s[0])

    ''' 按照除顿号、逗号以外的标点拆分句子 '''
    sentences = re.split(r"([。？！：；])", text)
    sentences = ["".join(i) for i in zip(sentences[0::2],sentences[1::2])]

    # longest_sentence = max(sentences, key=lambda s: len(s))
    # print("top lengths:", sorted((len(s) for s in sentences))[-10:])
    # print("longest sentence (len = %d): %s" % (len(longest_sentence), longest_sentence))
    
    ''' 去除大于maxlen的句子，将过短的句子拼成长句，尽可能长，但不要超过maxlen '''
    maxlen = 254
    new_sentences = []
    cur = ""
    for s in sentences:
        if len(s) > maxlen:
            continue
        if len(cur) + len(s) > maxlen:
            new_sentences.append(cur)
            cur = s
        else:
            cur += s
    if cur != "":
        new_sentences.append(cur)
    
    sentences = new_sentences
    del new_sentences
    print("number of sentences:", len(sentences))

    ''' 为了测试甲言，输出sentences到txt文件中，每行一个 '''
    sentences_output_root = os.path.join(data_root, "sentences_txt_files/", split)
    os.makedirs(sentences_output_root, exist_ok=True)
    sentences_output_filename = split + ".txt"
    with open(os.path.join(sentences_output_root, sentences_output_filename), 'w') as f:
        for sentence in sentences:
            f.write(sentence + '\n')

    
    ''' 制作puncless_text和label(punc, punc_type) ，并输出到json中; 同时记录标点/非标点出现次数'''
    output_root = os.path.join(data_root, "json_files")
    output_filename = split + ".json"
    os.makedirs(output_root, exist_ok=True)

    punc_cnt = torch.zeros((2,))
    punc_type_cnt = torch.zeros((8,))

    with open(os.path.join(output_root, output_filename), 'w') as f:
        f.write('{\n    "data": [\n')
        for i, s in enumerate(tqdm(sentences)):
            f.write('        {\n')

            puncless_text = []
            punc = []
            punc_type = []
            for c in s:
                if c not in punctuations_list:
                    puncless_text.append(c)
                    punc.append(0)
                    punc_type.append(0)

                    punc_cnt[0] += 1
                    punc_type_cnt[0] += 1
                else:
                    punc[-1] = 1
                    punc_type[-1] = punctuations_list.index(c) + 1

                    punc_cnt[0] -= 1
                    punc_type_cnt[0] -= 1

                    punc_cnt[1] += 1
                    punc_type_cnt[punc_type[-1]] += 1


            f.write('            "tokens": ' + json.dumps(list(puncless_text)) + ',\n')
            f.write('            "punc": ' + json.dumps(punc) + ',\n')
            f.write('            "punc_type": ' + json.dumps(punc_type) + '\n')

            # print(len(puncless_text))
            # print(len(punc))
            # exit()

            f.write('        }')

            if i != len(sentences) - 1:
                f.write(',')
            f.write('\n')
        f.write('    ]\n}\n')

    print("punc_cnt:", punc_cnt)
    print("punc_type_cnt:", punc_type_cnt)


if __name__ == '__main__':
    preprocess('train')
    preprocess('val')
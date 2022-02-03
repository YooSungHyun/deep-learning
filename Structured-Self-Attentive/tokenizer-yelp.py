from __future__ import print_function
import argparse
import json
import random
from util import Dictionary
import pandas as pd

'''
    데이터를 전처리하기 위함입니다. train.py를 수정한다면, 꼭 필요한 절차 및 소스는 아닙니다만,
    빠른 검증 및 실무 적용을 위해 최대한 적게 수정하여 사용합니다.
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tokenizer')
    parser.add_argument('--input', type=str, default='', help='input file')
    parser.add_argument('--output', type=str, default='', help='output file')
    parser.add_argument('--dict', type=str, default='', help='dictionary file')
    parser.add_argument('--label_list', type=str, default='', help='label category file')
    args = parser.parse_args()

    # CSV를 읽어와서 필요한 컬럼을 뽑아서 사용하는 형태로 됩니다.
    # 즉, csv로 읽어진 pandas에서 필요하신 컬럼을 잘 선택하셔서 활용하세요.
    csv_df = pd.read_csv(args.input,sep=',', encoding='utf-8')
    tot_x = list()
    '''
        해당 경로의 csv 컬럼을 꼭 사용할 컬럼으로 변환하여야 합니다.
    '''
    for idx, text in enumerate(csv_df['wordpiece'].values):
        tot_x.append(text.split())
    tot_y = csv_df['cate2'].values

    dictionary = Dictionary()
    dictionary.add_word('<pad>')  # add padding word
    # 학습에 사용될 데이터가 처리됩니다.
    with open(args.output, 'w') as fout:
        csv_df = csv_df.sample(frac=1).reset_index(drop=True)
        for i in range(len(csv_df)):
            data = {
                'label': tot_y[i],
                'text': tot_x[i]
            }
            fout.write(json.dumps(data) + '\n')
            for item in data['text']:
                dictionary.add_word(item)
            if i % 100 == 99:
                print('%d/%d files done, dictionary size: %d' %
                      (i + 1, len(csv_df), len(dictionary)))
        fout.close()
    
    # 벡터 임베딩을 시키기 위한 단어 투 벡터에 사용될 딕셔너리를 만듭니다. (딕셔너리의 인덱스가 단어의 벡터가 됨)
    with open(args.output, 'rb') as fout:
        for i in range(len(fout.readlines())):
            data = {
                'label': tot_y[i],
                'text': tot_x[i]
            }
            for item in data['text']:
                dictionary.add_word(item)
            if i % 100 == 99:
                print('%d/%d files done, dictionary size: %d' %
                      (i + 1, len(csv_df), len(dictionary)))
        fout.close()
    
    # 딕셔너리 출력
    with open(args.dict, 'w') as fout:  # save dictionary for fast next process
        fout.write(json.dumps(dictionary.idx2word) + '\n')
        fout.close()

    # 라벨로 사용할 카테고리들이 저장됩니다.
    with open(args.label_list, 'w') as fout:
        set_y = list(set(tot_y))
        fout.write(json.dumps(set_y) + '\n')
        fout.close()
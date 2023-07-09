import pandas as pd
from pathlib import Path

import gzip

import sys

from bert_dataset import BertDataset
from transformers import AutoTokenizer

import glob
from tqdm import tqdm


ds_dir = sys.argv[1]
assert ds_dir in ['2017/test', '2017/train', '2018/test', '2019/dta/test', '2019/intervention/test', '2019/intervention/train']

topics_dir = [dir_ for dir_ in glob.glob(f'./clef_info/{ds_dir}/*')]

path_or_name = [('bert-base-cased', 'org_bert-base')]

for t_dir in tqdm(topics_dir):
    raw_text = pd.read_csv(f'{t_dir}/raw_text.csv')
    rel_info = pd.read_pickle(f'{t_dir}/rel_info.pkl')

    ds_name = ds_dir.replace('/', '_')
    topic = t_dir[-8:]
    run_name = f'clef{ds_name}_{topic}'

    for name, outname in path_or_name:

        tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./hf_cache')

        print(f"{name} tokenizing...")
        full_ds = BertDataset.from_raw_text( raw_text.raw, rel_info, tokenizer, max_length=512, verbose=True )

        print("saving...")
        Path('./cache_new').mkdir(exist_ok=True)
        full_ds.save_encoded_features(gzip.open(f'./cache_new/clef/{run_name}_{outname}.512.pkl.gz', 'wb'))



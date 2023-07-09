import pandas as pd
from pathlib import Path
import gzip

import sys

from bert_dataset import BertDataset
from transformers import AutoTokenizer


ds_name = sys.argv[1]
assert ds_name in ['rcv1', 'jb']

sampled_mask = pd.read_pickle(f'./{ds_name}_info/sampled-for-bertal.pkl')

if ds_name == 'rcv1':
    raw_text = pd.read_csv('./rcv1_info/raw_text.csv').iloc[sampled_mask]
    rel_info = pd.read_pickle('./rcv1_info/rel_info.pkl').iloc[sampled_mask]
elif ds_name == 'jb':
    raw_text = pd.read_pickle('./jb_info/raw_text_cleaned.pkl').iloc[sampled_mask]
    rel_info = pd.read_pickle('./jb_info/rel_info.pkl').iloc[sampled_mask]


dummy_topic = {'rcv1': 'TAIWAN', 'jb': '415'}  # original authors' setting
run_name = {'rcv1': 'rcv1-20sub', 'jb': 'jb50sub'}  # original authors' setting

## BERTs
path_or_name = [('bert-base-cased', 'org_bert-base')]

for name, outname in path_or_name:

    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./hf_cache')

    print(f"{name} tokenizing...")
    full_ds = BertDataset.from_raw_text(raw_text.raw, rel_info[dummy_topic[ds_name]], tokenizer, max_length=512, verbose=True )

    print("saving...")
    Path('./cache_new').mkdir(exist_ok=True)
    full_ds.save_encoded_features( gzip.open(f'./cache_new/{run_name[ds_name]}_{outname}.512.pkl.gz', 'wb') )



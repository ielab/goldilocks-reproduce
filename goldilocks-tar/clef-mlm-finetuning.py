# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from transformers.trainer_utils import is_main_process

import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import gzip
from datetime import datetime

import torch
import glob

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class DumpCallback(TrainerCallback):

    def on_epoch_begin(self, args, state, control, logs=None, model=None, **kwargs):
        pass

    def on_epoch_end(self, args, state, control, logs=None, model=None, **kwargs):
        path = f'./mlm-finetune/clef/{ds_name[:4]}/bert-base_clef_{ds_name}_{topic}_ep{int(state.epoch)}/'
        # path = f'./mlm-finetune/biolink/clef/{ds_name[:4]}/biolink_bert-base_clef_{ds_name}_{topic}_ep{int(state.epoch)}/'
        print(f"Saving after epoch {state.epoch} to {path}")
        model.save_pretrained(path)
        torch.save(args, os.path.join(path, "training_args.bin"))

ds_dir = sys.argv[1]
assert ds_dir in ['2017/test', '2017/train', '2018/test', '2019/dta/test', '2019/intervention/test', '2019/intervention/train']

topics_dir = [dir_ for dir_ in glob.glob(f'./clef_info/{ds_dir}/*')]

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

set_seed(12345)

model_path_or_name = 'bert-base-cased'  # michiyasunaga/BioLinkBERT-base

config = AutoConfig.from_pretrained(model_path_or_name, cache_dir='./hf_cache/')

tokenizer = AutoTokenizer.from_pretrained(
    model_path_or_name, cache_dir='./hf_cache/', use_fast=False, max_length=512,
)


def tokenize_function(t):
    return tokenizer(
        t,
        padding=True,
        truncation=True,
        max_length=512,
        return_special_tokens_mask=True,
    )


for t_dir in tqdm(topics_dir):

    ds_name = ds_dir.replace('/', '_')
    topic = t_dir[-8:]
    run_name = f'clef{ds_name}_{topic}'

    tokenized_fn = f'./mlm-finetune/clef/{ds_name[:4]}/clef_{ds_name}_{topic}_bert_tokenized_cached.pkl.gz'
    output_dir = f'./mlm-finetune/clef/{ds_name[:4]}/bert-base_clef_{ds_name}_{topic}_final/'
    # tokenized_fn = f'./mlm-finetune/biolink/clef/{ds_name[:4]}/clef_{ds_name}_{topic}_bert_tokenized_cached.pkl.gz'
    # output_dir = f'./mlm-finetune/biolink/clef/{ds_name[:4]}/biolink_bert-base_clef_{ds_name}_{topic}_final/'

    if os.path.exists(tokenized_fn):
        tokenized_dataset = pd.read_pickle(tokenized_fn)
    else:
        raw_text = pd.read_csv(f'{t_dir}/raw_text.csv')

        tokenized_dataset = [
            tokenize_function(t) for t in tqdm(raw_text.raw)
        ]
        pickle.dump(tokenized_dataset, gzip.open(tokenized_fn, 'wb'))

    model = AutoModelForMaskedLM.from_pretrained(
        model_path_or_name, cache_dir='./hf_cache/',
        from_tf=False,
        config=config,
    )

    model.resize_token_embeddings(len(tokenizer))

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=10,
            per_device_train_batch_size=20,
            overwrite_output_dir=False,
            save_steps=1000000,
            save_total_limit=10,
        ),
        train_dataset=tokenized_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[DumpCallback]
    )

    # train_result = trainer.train(model_path=model_path_or_name)
    trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload




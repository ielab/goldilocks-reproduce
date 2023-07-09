"""
This module includes a class for interfacing huggingface transformer model 
"""
import numpy as np
from scipy.special import softmax

from tqdm import tqdm

# TODO: use Sean's logger

import torch

from transformers import BertForSequenceClassification, AutoTokenizer
from transformers import PreTrainedModel, Trainer, TrainingArguments
from transformers import TrainerCallback

from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from bert_dataset import BertDataset
#from longformer_dataset import LongformerDataset

from libact.base.interfaces import ProbabilisticModel, ContinuousModel
from sklearn.exceptions import NotFittedError

_default_training_args = {
    'num_train_epochs': 10,                  # total # of training epochs
    'learning_rate': 2e-3,
    'per_device_train_batch_size': 32,  # batch size per device during training
    'per_device_eval_batch_size': 1024,   # batch size for evaluation
    'warmup_steps': 50,                 # number of warmup steps for learning rate scheduler
    'weight_decay': 0.01,               # strength of weight decay
    'logging_dir': './logs',            # directory for storing logs
    'save_steps': 2000,
    'save_total_limit': 5,
    'disable_tqdm': True,
    'overwrite_output_dir': True,
    'fp16': True,                        # we enable for speed-up

}

class CustomeTrainer(Trainer):
    # custom loss function is done by overwriting the method
    # def compute_loss(self, model, inputs):
    #     labels = inputs.pop("labels")
    #     outputs = models(**inputs)
    #     logits = outputs[0]
    #     return my_custom_loss(logits, labels)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            trainable = ['classifier.weight', 'classifier.bais', 'pooler']
            optimizer_grouped_parameters = [
                {
                    'params': p, 
                    'weight_dacay': 0.0 if any(nd in n for nd in no_decay) else self.args.weight_decay,
                    'lr': self.args.learning_rate if any(nd in n for nd in trainable) else 0.0
                }
                for n, p in self.model.named_parameters()
            ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )

        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )


class TransformerModel(ContinuousModel):

    def __init__(self, model_path, collator, **kwargs):
        if isinstance(model_path, PreTrainedModel):
            self.model = model_path
        else:
            self.model = BertForSequenceClassification.from_pretrained(model_path, cache_dir='./hf_cache/')

        self.collator = collator

        training_args = _default_training_args.copy()
        training_args.update(kwargs)
        print(kwargs)
        self.training_args = training_args

        self.trainer = None
    
    def to_cpu(self):
        self.model = self.model.cpu()

    def train(self, train_dataset, eval_dataset=None):
        #LongformerDataset
        if not isinstance(train_dataset, (BertDataset)):
            raise TypeError("dataset should be a BertDataset.")
        
        eval_dataset = eval_dataset or train_dataset
        if train_dataset.__class__ != eval_dataset.__class__:
            raise TypeError("eval dataset and train dataset needs to be "
                            "the same class")
        
        if train_dataset.len_labeled() == 0:
            # TODO: should throw warning
            return 

        print(self.training_args)

        self.trainer = CustomeTrainer(
            model=self.model,
            args=TrainingArguments(**self.training_args),
            data_collator=self.collator,
            train_dataset=train_dataset.get_labeled_features(),
            eval_dataset=eval_dataset.get_labeled_features()
        )

        self.trainer.train()
    
    def save_model(self, path):
        self.trainer.save_model(str(path))

    def predict(self, features):
        return self.predict_real(features).argmax(axis=-1)

    def predict_real(self, features):
        if self.trainer is None:
            raise NotFittedError()
        logits, gold, _ = self.trainer.predict(features)
        return logits

    def score(self, testing_dataset):
        pass

    def predict_proba(self, features):
        #softmax(self.predict_real(features))[:, -1]
        return softmax(self.predict_real(features), axis=-1)
        
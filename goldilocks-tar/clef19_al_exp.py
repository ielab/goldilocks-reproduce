import numpy as np
import pandas as pd

from ir_datasets import log

_logger = log.easy()

import argparse
from tqdm import tqdm
import pickle, gzip, json
import shutil
from datetime import datetime
from pathlib import Path
from hashlib import md5

from itertools import product

from libact.base.dataset import Dataset
from libact.labelers import IdealLabeler

from libact.query_strategies import UncertaintySampling, RandomSampling, RelevanceFeedbackSampling
from al_record import DefaultMetrics, HeuristicsStoppingInfo

from bert_dataset import BertDataset
from bert_model import TransformerModel

import torch
from transformers import BertForSequenceClassification
from transformers import LongformerForSequenceClassification

# postrun
from scipy.special import softmax



def main(args):
    rel_info = pd.read_pickle(args.dataset_path / 'rel_info.pkl')

    hashs = pd.Series(rel_info.index.astype(str).map(lambda x: md5(x.encode()).hexdigest()),
                      index=rel_info.index, name='md5')

    rel_info = rel_info.assign(md5=hashs).reset_index(drop=True)
    sorted_rel = rel_info.sort_values('md5')

    for topic, qsname in product(args.topic, args.sampling_strategy):
        tag = f"clef19_intervention_train_{topic}_{args.iseed}_{qsname}"

        if (args.output_path / f"{tag}.results.pkl").exists():
            if args.resume_runs:
                # print("--RESUME_RUNS HAS NOT IMPLEMENTED YET -- SKIPPING FOR NOW")
                try:
                    metrics, cost, infos = pd.read_pickle(args.output_path / f"{tag}.results.pkl")

                    it_start = len(metrics)
                    if it_start >= args.niter:
                        _logger.info(f"Run {tag} is done based on the number of it, skipped")
                        continue

                    resume_known = np.concatenate([e['ask_id'] for e in infos])
                    logits = infos[-1]['logits']

                    if (args.output_path / tag / f"it_{it_start - 1}_snapshot").exists():
                        args.model_path = str(args.output_path / tag / f"it_{it_start - 1}_snapshot")
                        _logger.info(f"resume and loaded model from {args.model_path}")
                    else:
                        _logger.warn(f"Model file does not exists -- start from scratch")
                        metrics, cost, infos = [], [], []
                        it_start = 0
                except:
                    _logger.warn("File " + str(
                        args.output_path / f"{tag}.results.pkl") + " failed to open -- start from scratch")
                    metrics, cost, infos = [], [], []
                    it_start = 0
            else:
                _logger.warn(f"{tag} exists in {args.output_path}, skipped")
                continue
        else:
            metrics, cost, infos = [], [], []
            it_start = 0

        _logger.info(f"{tag} starting from {it_start}")

        Y = rel_info[topic].astype(int)
        seedset = [sorted_rel.index[sorted_rel[topic]][args.iseed]]  # one positive

        _logger.info(f"loading cached dataset from {str(args.cached_dataset)}")
        if args.model_type == 'Bert':
            full_ds = BertDataset.from_cache_file(args.cached_dataset, Y)
            torch_model = BertForSequenceClassification.from_pretrained(args.model_path, cache_dir='./hf_cache/')
            collator = BertDataset.collator
        else:
            raise ValueError(f"Unknown model type {args.model_type}")

        trn_ds = full_ds.spawn_dataset([None] * Y.size)
        labeler = IdealLabeler(full_ds)

        if it_start > 0:
            trn_ds.update(resume_known, labeler.label_by_id(resume_known))

        # path_or_name
        model = TransformerModel(torch_model, collator,
                                 disable_tqdm=args.disable_tqdm,
                                 learning_rate=args.lr,
                                 num_train_epochs=args.train_epochs,
                                 per_device_train_batch_size=100,  # 26 fixed for rtx GPU
                                 per_device_eval_batch_size=1000,  # 600 fixed for rtx GPU
                                 output_dir=str(args.output_path / tag))

        if qsname == 'relevance':
            qs = RelevanceFeedbackSampling(trn_ds, model=model)
        elif qsname == 'uncertainty':
            qs = UncertaintySampling(trn_ds, model=model)
        else:
            raise ValueError(f'Unknown sampling strategy {qsname}')

        # met_recording = DefaultMetrics(trn_ds, labeler, post_training_cost=1, dummy_last=0)
        cost_recording = HeuristicsStoppingInfo(trn_ds, labeler,
                                                target_recall=args.target_recall,
                                                dummy_last=0)

        # time_stat = []
        for i in range(it_start, args.niter):
            # time_start = time.time()
            _logger.debug(f"it={i} querying")
            if i == 0:
                ask_id = seedset
            else:
                ask_id = qs.make_query(n_ask=args.batch_size, retrain=False, dvalue=logits)

            _logger.debug(f"it={i} update known")
            trn_ds.update(ask_id, labeler.label_by_id(ask_id))
            # check dataset updating
            # print("dataset size:", len(trn_ds))

            _logger.debug(f"it={i} training -- train on all known = {not args.train_only_on_new}")
            if args.train_only_on_new:
                model.train(trn_ds.spawn_dataset(labels=labeler.label_by_id(ask_id), idx=ask_id))
            else:
                model.train(trn_ds)


            # always save the latest version
            model.save_model(args.output_path / tag / f"it_{i}_snapshot")
            if not args.save_it_models:
                # delete past model
                for f in (args.output_path / tag).glob("it_[0-9]*_snapshot"):
                    if f.name != f"it_{i}_snapshot":
                        shutil.rmtree(f)

            _logger.debug(f"it={i} predicting")

            logits = model.predict_real(full_ds.get_labeled_features())
            probabilities = softmax(logits, axis=-1)

            _logger.debug(f"it={i} evaluate")
            # saving runs
            run_name = f"{tag}_ep{i}.csv"
            run_path = f"{args.output_path}/{run_name}"
            met_recording = DefaultMetrics(trn_ds, labeler, run_path=run_path, post_training_cost=1, dummy_last=0)
            # metrics.append(met_recording.evaluate(logits[:, -1]))
            # cost.append(cost_recording.evaluate(logits[:, -1])[0])

            metrics.append(met_recording.evaluate(probabilities[:, -1], logits[:, -1]))  # logits
            cost.append(cost_recording.evaluate(probabilities[:, -1], logits[:, -1])[0])  # logits

            # infos.append({
            #     'ask_id': ask_id,
            #     'logits': logits
            # })

            infos.append({
                'ask_id': ask_id,
                'probabilities': probabilities,
                'logits': logits
            })  # logits


            _logger.info(f"it={i} done -- R-P={metrics[-1]['R-P']}, DFR0.8={metrics[-1]['DFR@0.8']}, "  # Try 0.95 
                         f"asked_pos={sum(labeler.label_by_id(ask_id))} ]")


            if i % args.dump_freq == 0:
                _logger.info(f"it={i} dumpping info")
                pickle.dump((metrics, cost, infos), (args.output_path / f"{tag}.results.pkl.saving").open('wb'))
                (args.output_path / f"{tag}.results.pkl.saving").rename((args.output_path / f"{tag}.results.pkl"))

        _logger.info(f"{tag} finished -- dumpping")
        pickle.dump((metrics, cost, infos), (args.output_path / f"{tag}.results.pkl.saving").open('wb'))
        (args.output_path / f"{tag}.results.pkl.saving").rename((args.output_path / f"{tag}.results.pkl"))

        # remove model files
        if not args.save_it_models:
            shutil.rmtree(args.output_path / tag)

        # clean up GPU
        model.to_cpu()
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # active learning setting
    parser.add_argument('--topic', type=str, nargs='+', default=['CD007431'])
    parser.add_argument('--sampling_strategy', type=str, nargs='+', default=['relevance', 'uncertainty'])
    parser.add_argument('--iseed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--niter', type=int, default=20)
    parser.add_argument('--target_recall', type=float, nargs='+', default=[0.95, 0.9, 0.85, 0.8, 0.7])

    parser.add_argument('--model_type', choices=['Bert'], default='Bert')
    parser.add_argument('--model_path', type=str, default='bert-base-cased')
    parser.add_argument('--cached_dataset', type=Path, default='./cache_new/clef/clef2017_test_CD007431_org_bert-base.512.pkl.gz')
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--train_only_on_new', action='store_true', default=False)
    parser.add_argument('--disable_tqdm', action='store_true', default=False)

    parser.add_argument('--dataset_path', type=Path, default='./clef_info/2017/CD007431')

    parser.add_argument('--output_path', type=Path, default='./results/')
    parser.add_argument('--resume_runs', action='store_true', default=False)
    parser.add_argument('--save_it_models', action='store_true', default=False)
    parser.add_argument('--dump_freq', type=int, default=1)

    args = parser.parse_args()

    main(args)

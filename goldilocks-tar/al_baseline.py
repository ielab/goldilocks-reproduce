import pathlib

import numpy as np
import pandas as pd
import os

from ir_datasets import log

_logger = log.easy()

import argparse
from tqdm import tqdm
import pickle, gzip, json
import shutil
from time import time
from pathlib import Path
from hashlib import md5

from itertools import product

from libact.base.dataset import Dataset
from libact.labelers import IdealLabeler

from libact.query_strategies import UncertaintySampling, RelevanceFeedbackSampling
from clef_record import DefaultMetrics, HeuristicsStoppingInfo

# from sklearn.linear_model import LogisticRegression
from libact.models import LogisticRegression
from joblib import dump, load
import scipy.sparse as sp

from lr_dataset import LrDataset


def main(args):
    # Sample one positive seed document
    sampled_mask = pd.read_pickle(args.dataset_path / 'sampled-for-bertal.pkl')
    rel_info = pd.read_pickle(args.dataset_path / 'rel_info.pkl')

    hashs = pd.Series(rel_info.index.astype(str).map(lambda x: md5(x.encode()).hexdigest()),
                      index=rel_info.index, name='md5')

    rel_info = rel_info.assign(md5=hashs).iloc[sampled_mask].reset_index(drop=True)
    sorted_rel = rel_info.sort_values('md5')

    # main logic
    for cate, qsname in product(args.category, args.sampling_strategy):
        tag = f"{cate}_{args.iseed}_{qsname}"

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

        Y = rel_info[cate].astype(int)
        seedset = [sorted_rel.index[sorted_rel[cate]][args.iseed]]  # one positive

        _logger.info(f"loading cached dataset from {str(args.cached_dataset)}")
        if args.model_type == 'Logistic Regression':
            # ds_features = pickle.load(open(args.cached_dataset, 'rb'))
            full_ds = LrDataset.from_cache_file(args.cached_dataset, Y)
            model = LogisticRegression(random_state=0, penalty='l2', C=1.0)

        else:
            raise ValueError(f"Unknown model type {args.model_type}")

        trn_ds = full_ds.spawn_dataset([None] * Y.size)
        labeler = IdealLabeler(full_ds)

        if it_start > 0:
            trn_ds.update(resume_known, labeler.label_by_id(resume_known))

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
            # assert X.shape[0] == len(y)

            trn_t = time()
            if args.train_only_on_new:
                model.train(trn_ds.spawn_dataset(labels=labeler.label_by_id(ask_id), idx=ask_id))
            else:
                # print(trn_ds.get_labeled_features())
                # fix one positive class for lr training
                trn_seed = trn_ds.get_labeled_features()
                X = trn_seed[0][0]
                y = trn_seed[0][1]
                assert X.shape[0] == len([y])
                # add dummy negative  ([0,...,0],0)
                if np.unique(y).size == 1 and y == 1:
                    X = sp.vstack([X, sp.csr_matrix((1, X.shape[1]))])
                    y = np.array([y, 0])
                    trn_ds_0 = Dataset(X, y)
                    model.train(trn_ds_0)
                else:
                    model.train(trn_ds)
            print(f'training time: {time() - trn_t}')

            # always save the latest version
            if not os.path.isdir(args.output_path / tag):
                os.makedirs(args.output_path / tag)
            dump(model, args.output_path / tag / f"it_{i}_snapshot")

            if not args.save_it_models:
                # delete past model
                for f in (args.output_path / tag).glob("it_[0-9]*_snapshot"):
                    if f.name != f"it_{i}_snapshot":
                        pathlib.Path(f).unlink()

            _logger.debug(f"it={i} predicting")
            eval_t = time()
            logits = model.predict_proba(full_ds.format_sklearn()[0])
            print(f'evaluation time: {time() - eval_t}')

            _logger.debug(f"it={i} evaluate")
            # saving runs
            run_name = f"{tag}_ep{i}.csv"
            run_folder = f"{args.output_path}/{tag}_runs/"

            isExists = os.path.exists(run_folder)
            if not isExists:
                os.makedirs(run_folder)
            else:
                pass
            run_path = f"{run_folder}/{run_name}"
            met_recording = DefaultMetrics(trn_ds, labeler, run_path=run_path, post_training_cost=1, dummy_last=0)
            metrics.append(met_recording.evaluate(logits[:, -1]))
            cost.append(cost_recording.evaluate(logits[:, -1])[0])
            infos.append({
                'ask_id': ask_id,
                'logits': logits
            })

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # active learning setting
    # industries_I35102, regions_ALG, topics_E12, regions_POL
    parser.add_argument('--category', type=str, nargs='+', default=['ALG'])
    parser.add_argument('--sampling_strategy', type=str, nargs='+', default=['relevance', 'uncertainty'])
    parser.add_argument('--iseed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--niter', type=int, default=20)
    parser.add_argument('--target_recall', type=float, nargs='+', default=[0.95, 0.9, 0.8, 0.7, 0.6])

    parser.add_argument('--model_type', choices=['Logistic Regression'], default='Logistic Regression')
    parser.add_argument('--cached_dataset', type=Path, default='./rcv1_info/rcv1_sampled_features.pkl')
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--train_only_on_new', action='store_true', default=False)
    parser.add_argument('--disable_tqdm', action='store_true', default=False)

    parser.add_argument('--dataset_path', type=Path, default='./rcv1_info')

    parser.add_argument('--output_path', type=Path, default='./results/baseline/rcv1')
    parser.add_argument('--resume_runs', action='store_true', default=False)
    parser.add_argument('--save_it_models', action='store_true', default=False)
    parser.add_argument('--dump_freq', type=int, default=1)

    args = parser.parse_args()

    main(args)
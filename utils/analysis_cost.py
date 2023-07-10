import pickle
import glob
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, f
from statsmodels.sandbox.stats.multicomp import multipletests

# RCV1-v2
with open('./rcv1-v2/categories_dict.pkl', 'rb') as file:
    categories_dict = pickle.load(file)


def min_cost(result, cost_structure, recall=0.8):
    """Return recorded minimal cost among 20 active learning iterations for each category."""
    costs = []
    for epoch in range(20):
        costs.append(result[1][epoch][(recall, cost_structure)])
    return min(costs)


def costs_rcv1_bert(categories, strategy, cost_structure, pre_train_epoch):
    """Report minimal cost under difficulty hierarchy."""
    costs_rcv1_bert = {}

    for difficulty, topics in categories.items():
        for topic in topics:
            result = pd.read_pickle(f"./results/rcv1/ep{pre_train_epoch}/{topic}_0_{strategy}.results.pkl")
            if difficulty not in costs_rcv1_bert.keys():
                costs_rcv1_bert[difficulty] = {topic: min_cost(result=result, cost_structure=cost_structure)}
            else:
                costs_rcv1_bert[difficulty].update({topic: min_cost(result=result, cost_structure=cost_structure)})
    return costs_rcv1_bert


def optimal_review_cost(cost_structure, result, target_recall=0.8):
    """Compute review cost under uniform cost and expensive training structure for target recall."""

    if cost_structure == 'uni-cost':
        cost_matrix = [1, 1, 1, 1]
    elif cost_structure == 'exp-train':
        cost_matrix = [10, 10, 1, 1]
    else:
        raise ("undefined cost structure.")

    costs = []
    for epoch in range(20):
        a_p = result[1][epoch][(target_recall, 'training', 'pos')]
        a_n = result[1][epoch][(target_recall, 'training', 'neg')]
        b_p = result[1][epoch][(target_recall, 'post-training-above', 'pos')]
        b_n = result[1][epoch][(target_recall, 'post-training-above', 'neg')]
        record = (a_p, a_n, b_p, b_n)
        cost = sum(num * ucost for num, ucost in zip(record, cost_matrix))
        costs.append(cost)
    return costs


def costs_rcv1_baseline(categories, strategy, cost_structure):
    costs_rcv1 = {}

    for difficulty, topics in categories.items():
        for topic in topics:
            result = pd.read_pickle(f"./results/baseline/rcv1/{topic}_0_{strategy}.results.pkl")
            if difficulty not in costs_rcv1.keys():
                costs_rcv1[difficulty] = {topic: min(optimal_review_cost(cost_structure, result))}
            else:
                costs_rcv1[difficulty].update({topic: min(optimal_review_cost(cost_structure, result))})
    return costs_rcv1


def costs_ratio_rcv1(categories_dict, strategy, cost_structure):
    """
    categories_dict: dict of rcv1-v2 categories with difficulty hierarchy
    strategy: relevance, uncertainty
    cost-structure: uni-cost, exp-train
    """
    # compute costs of baseline logistic regression.
    print(f'cost_structure: {cost_structure}, strategy:{strategy}')
    costs_rcv1_baseline_dict = costs_rcv1_baseline(categories_dict, strategy=strategy, cost_structure=cost_structure)
    avg_costs_rcv1_baseline = 0

    for difficulty in costs_rcv1_baseline_dict.keys():
        avg_costs_rcv1_baseline += np.sum(list(costs_rcv1_baseline_dict[difficulty].values()))
    avg_costs_rcv1_baseline = avg_costs_rcv1_baseline / 45

    # compute costs of bert with further pre-training settings.
    for epoch in [0, 1, 2, 5, 10]:
        costs_rcv1_bert_dict = costs_rcv1_bert(categories_dict, strategy=strategy, cost_structure=cost_structure,
                                               pre_train_epoch=epoch)
        avg_costs_rcv1_bert = 0

        for difficulty in costs_rcv1_bert_dict.keys():
            avg_costs_rcv1_bert += np.sum(list(costs_rcv1_bert_dict[difficulty].values()))

        avg_costs_rcv1_bert = avg_costs_rcv1_bert / 45
        cost_ratio = avg_costs_rcv1_bert / avg_costs_rcv1_baseline

        print(f'pre-training epoch:{epoch}, relative cost:{cost_ratio}')

costs_ratio_rcv1(categories_dict, strategy='uncertainty', cost_structure='uni-cost')


# Jeb Bush
categories = list(pd.read_csv("../jeb-bush/jeb_bush_label.csv", header=None)[0])


def costs_jb_bert(categories, strategy, cost_structure, pre_train_epoch):
    """Report recorded minimal cost for each category."""
    costs_jb_bert = {}

    for topic in categories:
        result = pd.read_pickle(f"./results/jb/ep{pre_train_epoch}/{topic}_0_{strategy}.results.pkl")
        costs_jb_bert[topic] = min_cost(result=result,
                                        cost_structure=cost_structure)  # min(optimal_review_cost(cost_structure, result))

    return costs_jb_bert


def costs_jb_baseline(categories, strategy, cost_structure):
    """Report minimal total cost during 20 al iterations"""

    costs_jb_dict = {}

    for topic in categories:
        result = pd.read_pickle(f"./baseline/jb/{topic}_0_{strategy}.results.pkl")
        costs_jb_dict[topic] = min(optimal_review_cost(cost_structure, result))

    return costs_jb_dict


def costs_ratio_jb(categories, strategy, cost_structure):
    """
    categories_dict: dict of jeb bush categories
    strategy: relevance, uncertainty
    cost-structure: uni-cost, exp-train
    """
    # compute costs of baseline logistic regression.
    print(f'cost_structure: {cost_structure}, strategy:{strategy}')
    costs_jb_baseline_dict = costs_jb_baseline(categories, strategy=strategy, cost_structure=cost_structure)
    avg_costs_jb_baseline = np.mean(list(costs_jb_baseline_dict.values()))

    # compute costs of bert with further pre-training settings.
    for epoch in [0, 1, 2, 5, 10]:
        costs_jb_bert_dict = costs_jb_bert(categories, strategy=strategy, cost_structure=cost_structure,
                                           pre_train_epoch=epoch)
        avg_costs_jb_bert = np.mean(list(costs_jb_bert_dict.values()))

        cost_ratio = avg_costs_jb_bert / avg_costs_jb_baseline

        print(f'pre-training epoch:{epoch}, relative cost:{cost_ratio}')

costs_ratio_jb(categories, strategy='relevance', cost_structure='exp-train')


# CLEF collections

def costs_clef_bert(clef_yr, split_name, categories, strategy, cost_structure, pre_train_epoch, target_recall=0.8):
    """Report recorded minimal cost for each topic."""
    costs_clef_bert = {}

    for topic in categories:
        result = pd.read_pickle(
            f'./results/clef/ep{pre_train_epoch}/clef{clef_yr}_{split_name}/clef{clef_yr}_{split_name}_{topic}_0_{strategy}.results.pkl')
        costs_clef_bert[topic] = min_cost(result=result, cost_structure=cost_structure, target_recall=target_recall)

    return costs_clef_bert


def costs_clef_baseline(clef_yr, split_name, categories, strategy, cost_structure, target_recall=0.8):
    """Report minimal total cost during 20 al iterations"""
    costs_clef_dict = {}

    for topic in categories:
        result = pd.read_pickle(
            f"./results/baseline/clef/clef{clef_yr}_{split_name}/clef20{clef_yr}_{split_name}_{topic}_0_{strategy}.results.pkl")
        costs_clef_dict[topic] = min(optimal_review_cost(cost_structure, result, target_recall=target_recall))

    return costs_clef_dict


def costs_clef_biolink(clef_yr, split_name, categories, strategy, cost_structure, target_recall=0.8):
    """Report recorded minimal cost for each topic."""
    costs_clef_biolink = {}

    for topic in categories:
        result = pd.read_pickle(
            f'./biolink/ep0/clef{clef_yr}_{split_name}/clef{clef_yr}_{split_name}_{topic}_0_{strategy}.results.pkl')
        costs_clef_biolink[topic] = min(optimal_review_cost(cost_structure, result, target_recall=target_recall))

    return costs_clef_biolink


def costs_ratio_clef(clef_yr, split_name, categories, strategy, cost_structure, target_recall=0.8):
    """
    categories_dict: dict of clef topics
    strategy: relevance, uncertainty
    cost-structure: uni-cost, exp-train
    """
    # compute costs of baseline logistic regression.
    print(f'cost_structure: {cost_structure}, strategy:{strategy}')
    costs_clef_baseline_dict = costs_clef_baseline(clef_yr, split_name, categories, strategy=strategy,
                                                   cost_structure=cost_structure, target_recall=target_recall)
    avg_costs_clef_baseline = np.mean(list(costs_clef_baseline_dict.values()))

    # compute costs of bert with further pre-training settings.
    for epoch in [0, 1, 2, 5, 10]:
        costs_clef_bert_dict = costs_clef_bert(clef_yr, split_name, categories, strategy=strategy,
                                               cost_structure=cost_structure, pre_train_epoch=epoch,
                                               target_recall=target_recall)
        avg_costs_clef_bert = np.mean(list(costs_clef_bert_dict.values()))

        cost_ratio = avg_costs_clef_bert / avg_costs_clef_baseline

        print(f'pre-training epoch:{epoch}, relative cost:{cost_ratio}')


collections = ['2017/train', '2017/test', '2018/test','2019/dta/test','2019/intervention/train', '2019/intervention/test']
for col in collections:
    print(col)
    topics_clef = [dir_[-8:]for dir_ in glob.glob(f'./clef/{col}/*')]
    if col[2:4] == '19':
        split_name = col[5:].replace('/', '_')
    else:
        split_name = col[5:]
    costs_ratio_clef(clef_yr=col[2:4], split_name=split_name, categories=topics_clef, strategy='uncertainty', cost_structure='uni-cost', target_recall=0.8)
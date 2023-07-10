import pandas as pd
from analysis_cost import *


categories = pd.read_pickle("./rcv1-v2/categories_dict.pkl")
difficulties = list(categories.keys())


def costs_rcv1_avgbin_bert(categories, strategy, pre_train_epoch, cost_structure):
    """Report averaged cost for each bin with 5 categories, for each category, its cost is the minimal cost during 20 active learning iterations"""
    costs_rcv1 = {}
    costs_rcv1_avgbin = {}

    for difficulty, topics in categories.items():
        for topic in topics:
            result = pd.read_pickle(f"./results/rcv1/ep{pre_train_epoch}/{topic}_0_{strategy}.results.pkl")
            if difficulty not in costs_rcv1.keys():
                costs_rcv1[difficulty] = {topic: min(optimal_review_cost(cost_structure, result))}
            else:
                costs_rcv1[difficulty].update({topic: min(optimal_review_cost(cost_structure, result))})
        costs_rcv1_avgbin[difficulty] = np.mean(list(costs_rcv1[difficulty].values()))
    return costs_rcv1_avgbin


def costs_rcv1_baseline_avgbin(categories, strategy, cost_structure):
    """Report minimal total cost during 20 active learning iterations"""
    costs_rcv1 = {}
    costs_rcv1_avgbin = {}

    for difficulty, topics in categories.items():
        for topic in topics:
            result = pd.read_pickle(f"./results/baseline/rcv1/{topic}_0_{strategy}.results.pkl")
            if difficulty not in costs_rcv1.keys():
                costs_rcv1[difficulty] = {topic: min(optimal_review_cost(cost_structure, result))}
            else:
                costs_rcv1[difficulty].update({topic: min(optimal_review_cost(cost_structure, result))})
        costs_rcv1_avgbin[difficulty] = np.mean(list(costs_rcv1[difficulty].values()))
    return costs_rcv1_avgbin

relative_cost = {}
costs_avgbin_bert = costs_rcv1_avgbin_bert(categories, 'relevance', pre_train_epoch=10, cost_structure='exp-train')
costs_avgbin_baseline = costs_rcv1_baseline_avgbin(categories, 'relevance', cost_structure='exp-train')

for difficulty in difficulties:
    relative_cost[difficulty] = costs_avgbin_bert[difficulty]/costs_avgbin_baseline[difficulty]

pd.DataFrame.from_dict(relative_cost, orient='index')


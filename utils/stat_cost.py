import numpy as np
from scipy.stats import ttest_rel, f
from statsmodels.sandbox.stats.multicomp import multipletests
from analysis_cost import *


# RCV1-v2
def stat_test_rcv1(categories_dict, strategy, cost_structure):
    """
    """
    print(f'cost_structure: {cost_structure}, strategy:{strategy}')
    costs_rcv1_baseline_dict = costs_rcv1_baseline(categories_dict, strategy=strategy, cost_structure=cost_structure)
    costs_rcv_baseline_list = []

    for difficulty in costs_rcv1_baseline_dict.keys():
        for cost in costs_rcv1_baseline_dict[difficulty].values():
            costs_rcv_baseline_list.append(cost)
    # plt.bar(range(len(costs_rcv_baseline_list)), costs_rcv_baseline_list)

    p_values = []
    for epoch in [0, 1, 2, 5, 10]:
        count = 0
        costs_rcv1_bert_dict = costs_rcv1_bert(categories_dict, strategy=strategy, cost_structure=cost_structure,
                                               pre_train_epoch=epoch)
        costs_rcv_bert_list = []
        for difficulty in costs_rcv1_bert_dict.keys():
            for cost in costs_rcv1_bert_dict[difficulty].values():
                costs_rcv_bert_list.append(cost)

        p = ttest_rel(costs_rcv_baseline_list, costs_rcv_bert_list)[1]
        p_values.append(p)
        # plt.figure()
        # plt.bar(range(len(costs_rcv_bert_list)), np.subtract(costs_rcv_baseline_list,costs_rcv_bert_list))
        # plt.tight_layout()

    corrected_p = multipletests(p_values, method='bonferroni')
    print(f'p-values:{p_values}')
    print(f'significance after bonferroni:{corrected_p[0]}')

stat_test_rcv1(categories_dict, strategy='uncertainty', cost_structure='uni-cost')


# Jeb Bush
def stat_test_jb(categories, strategy, cost_structure):
    """
    """
    print(f'cost_structure: {cost_structure}, strategy:{strategy}')
    costs_jb_baseline_dict = costs_jb_baseline(categories=categories, strategy=strategy, cost_structure=cost_structure)
    costs_jb_baseline_list = list(costs_jb_baseline_dict.values())

    p_values = []
    for epoch in [0, 1, 2, 5, 10]:
        costs_jb_bert_dict = costs_jb_bert(categories=categories, strategy=strategy, cost_structure=cost_structure,
                                           pre_train_epoch=epoch)
        costs_jb_bert_list = list(costs_jb_bert_dict.values())

        p = ttest_rel(costs_jb_baseline_list, costs_jb_bert_list)[1]
        p_values.append(p)

    corrected_p = multipletests(p_values, method='bonferroni')
    print(f'pre-training epoch:{epoch}, significance:{corrected_p[0]}')

stat_test_jb(categories, strategy='uncertainty', cost_structure='uni-cost')


# CLEF collections
# CLEF 17 train as an example
def stat_test_clef(clef_yr, split_name, categories, strategy, cost_structure, expand=False):
    """
    Statistical significance test with cost structure, between baseline and bert runs, biolink included
    """
    print(f'cost_structure: {cost_structure}, strategy:{strategy}')
    costs_clef_baseline_dict = costs_clef_baseline(clef_yr, split_name, categories, strategy=strategy,
                                                   cost_structure=cost_structure)
    costs_clef_baseline_list = list(costs_clef_baseline_dict.values())
    if expand == True:
        costs_clef_biolink_dict = costs_clef_biolink(clef_yr, split_name, categories, strategy=strategy,
                                                     cost_structure=cost_structure)
        costs_clef_biolink_list = list(costs_clef_biolink_dict.values())

    p_values = []
    for epoch in [0, 1, 2, 5, 10]:
        costs_clef_bert_dict = costs_clef_bert(clef_yr, split_name, categories, strategy=strategy,
                                               cost_structure=cost_structure, pre_train_epoch=epoch)
        costs_clef_bert_list = list(costs_clef_bert_dict.values())

        p = ttest_rel(costs_clef_baseline_list, costs_clef_bert_list)[1]
        p_values.append(p)

    if expand == True:
        p_biolink = ttest_rel(costs_clef_baseline_list, costs_clef_biolink_list)[1]
        if p_biolink < 0.05:
            print("True for biolink and baseline.")
        p_values.append(p_biolink)

    corrected_p = multipletests(p_values, method='bonferroni')
    print(f'p-values:{p_values}')
    print(f'significance after bonferroni:{corrected_p[0]}')

topics_clef = [dir_[-8:]for dir_ in glob.glob('./clef/2017/train/*')]
stat_test_clef(clef_yr='17', split_name='train', categories=topics_clef, strategy='relevance', cost_structure='uni-cost', expand=True)
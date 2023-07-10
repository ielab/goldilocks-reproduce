from scipy.stats import ttest_rel, f
from statsmodels.sandbox.stats.multicomp import multipletests
from analysis_RP import *

# RCV1-v2
def get_results_baseline(dataset, categories, strategy, measure):
    results = []
    for topic in categories:
        topic_rel = read_pkl(f'./results/baseline/{dataset}/{topic}_0_{strategy}.results.pkl')
        assert len(topic_rel[0]) == 20
        results.append(topic_rel[0][-1][measure])
    return results

def get_results_bert(dataset, categories, epoch, strategy, measure):
    results = []
    for topic in categories:
        topic_rel = read_pkl(f'./results/{dataset}/{epoch}/{topic}_0_{strategy}.results.pkl')
        assert len(topic_rel[0]) == 20
        results.append(topic_rel[0][-1][measure])
    return results

# change dataset for 'rcv1' and 'jb'
dataset = 'jb'
strategy = 'uncertainty'
categories = []
# rcv1
# with open("../reuters/categories.csv")as f:
#     for line in f:
#         categories.append(line.split("_")[1].strip("\n"))
# jb
with open("../jeb-bush/jeb_bush_label.csv") as f:
    for line in f:
        categories.append(line.split(",")[0].strip("\n"))
measure = 'R-P'

goldilocks_baseline = get_results_baseline(dataset=dataset, categories=categories, strategy=strategy, measure=measure)

p_values = []
for ep in ['ep0', 'ep1', 'ep2', 'ep5', 'ep10']:
    goldilocks_bert_ep = get_results_bert(dataset=dataset, categories=categories, epoch=ep, strategy=strategy,
                                          measure=measure)
    p = ttest_rel(goldilocks_baseline, goldilocks_bert_ep)[1]
    p_values.append(p)

corrected_p = multipletests(p_values, method='bonferroni')
corrected_p


# CLEF collections
# CLEF18 test as an example
clef_yr = '18'
split_name= 'test'
strategy = 'relevance'
topics_clef = [dir_[-8:]for dir_ in glob.glob(f'./clef/2018/test/*')]

def get_results_baseline(clef_yr, split_name, categories, strategy, measure):
    results = []
    for topic in categories:
        topic_rel = read_pkl(f'./results/baseline/clef/clef{clef_yr}_{split_name}/clef20{clef_yr}_{split_name}_{topic}_0_{strategy}.results.pkl')
        assert len(topic_rel[0]) == 20
        results.append(topic_rel[0][-1][measure])
    return results

def get_results_bert(clef_yr, split_name, categories, epoch, strategy, measure):
    results = []
    for topic in categories:
        topic_rel = read_pkl(f'./results/clef/{epoch}/clef{clef_yr}_{split_name}/clef{clef_yr}_{split_name}_{topic}_0_{strategy}.results.pkl')
        assert len(topic_rel[0]) == 20
        results.append(topic_rel[0][-1][measure])
    return results

def get_results_biolink(clef_yr, split_name, categories, strategy, measure):
    results = []
    for topic in categories:
        topic_rel = pd.read_pickle(f'./results/biolink/ep0/clef{clef_yr}_{split_name}/clef{clef_yr}_{split_name}_{topic}_0_{strategy}.results.pkl')
        assert len(topic_rel[0]) == 20
        results.append(topic_rel[0][-1][measure])
    return results


clef_baseline = get_results_baseline(clef_yr=clef_yr, split_name=split_name, categories=topics_clef, strategy=strategy,
                                     measure='R-P')
clef_bioink_ep0 = get_results_biolink(clef_yr=clef_yr, split_name=split_name, categories=topics_clef, strategy=strategy,
                                      measure='R-P')

p_values = []
for ep in ['ep0', 'ep1', 'ep2', 'ep5', 'ep10']:
    clef_bert_ep = get_results_bert(clef_yr=clef_yr, split_name=split_name, categories=topics_clef, epoch=ep,
                                    strategy=strategy, measure='R-P')
    p = ttest_rel(clef_baseline, clef_bert_ep)[1]
    p_values.append(p)

p_bio = ttest_rel(clef_baseline, clef_bioink_ep0)[1]
if p_bio < 0.05:
    print(print("True for biolink and baseline."))

p_values.append(p_bio)

corrected_p = multipletests(p_values, method='bonferroni')
corrected_p
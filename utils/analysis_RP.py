import pickle
import glob
import pandas as pd


# RCV1-v2
def read_pkl(data_dir):
    res = open(data_dir, 'rb')
    result = pickle.load(res)
    res.close()
    return result


def avg_measure(categories, strategy, epoch, measure):
    result = 0
    for topic in categories:
        topic_rel = read_pkl(f'./results/rcv1_3/{epoch}/{topic}_0_{strategy}.results.pkl')
        assert len(topic_rel[0]) == 20
        result += topic_rel[0][-1][measure]
    return result / len(categories)


categories_rcv1 = []
with open("./rcv1-v2/categories.csv") as f:
    for line in f:
        categories_rcv1.append(line.split("_")[1].strip("\n"))

# strategy: relevance, uncertainty
# epoch: ep0, ep1, ep2, ep5, ep10
# measure: R-P
avg_measure(categories_rcv1, strategy='relevance', epoch='ep5', measure='R-P')

# Jeb Bush
categories_jb = []
with open("./jeb-bush/jeb_bush_label.csv") as f:
    for line in f:
        categories_jb.append(line.split(",")[0].strip("\n"))


def avg_measure_jb(categories, strategy, epoch, measure):
    result = 0
    for topic in categories:
        topic_rel = read_pkl(f'./results/jb/{epoch}/{topic}_0_{strategy}.results.pkl')
        assert len(topic_rel[0]) == 20
        result += topic_rel[0][-1][measure]
    return result / len(categories)


avg_measure_jb(categories=categories_jb, strategy='relevance', epoch='ep0', measure='R-P')

# CLEF collections
# CLEF17 train as an example
topics_clef = [dir_[-8:] for dir_ in glob.glob('./clef/2017/train/*')]

def avg_measure_clef(clef_yr, split_name, topic, strategy, pre_train_epoch, measure):
    result = 0
    topic_rel = read_pkl(f'./results/clef/{pre_train_epoch}/clef{clef_yr}_{split_name}/clef{clef_yr}_{split_name}_{topic}_0_{strategy}.results.pkl')
    assert len(topic_rel[0]) == 20  # check completeness
    result += topic_rel[0][-1][measure]  # final result
    return result

def measure_per_topic(topics, epoch, metric='R-P'):
    result = []
    for topic in topics:
        result.append(avg_measure_clef('17', 'train', topic, 'relevance', epoch, metric)) # relevance  uncertainty
    return result

def measure_clef_base(clef_yr, split_name, topic, strategy, measure):
    result = 0
    topic_rel = read_pkl(f'./results/baseline/clef/clef{clef_yr}_{split_name}/clef20{clef_yr}_{split_name}_{topic}_0_{strategy}.results.pkl')
    assert len(topic_rel[0]) == 20  # check completeness
    result += topic_rel[0][-1][measure]  # final result
    return result

def measure_per_topic_base(topics, metric='R-P'):
    result = []
    for topic in topics:
        result.append(measure_clef_base('17', 'train', topic, 'relevance', 'R-P'))
    return result


clef17_table = {'topic': list(topics_clef), 'baseline': measure_per_topic_base(topics_clef),'ep0': measure_per_topic(topics_clef, 'ep0'), 'ep1':measure_per_topic(topics_clef, 'ep1'),'ep2':measure_per_topic(topics_clef, 'ep2'), 'ep5': measure_per_topic(topics_clef, 'ep5'), 'ep10': measure_per_topic(topics_clef, 'ep10')}
pd.DataFrame(clef17_table)


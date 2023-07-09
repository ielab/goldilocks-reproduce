import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import kendalltau, spearmanr

class ActiveLearningRecord(object):
    def __init__(self, dataset, labeler, dummy_last=0):
        self.dataset = dataset
        self.lbr = labeler

        # TODO: change it to int -- could be 0 1 2
        self.dummy_last = int(dummy_last)
        self.npos = self.gold.sum()
    
    @property
    def N(self):
        return len(self.dataset) - self.dummy_last
    
    @property
    def rich(self):
        return self.npos / self.N

    @property
    def gold(self):
        return self.lbr.y
    
    @property
    def known(self):
        if self.dummy_last > 0:
            return self.dataset.get_labeled_mask()[:-self.dummy_last]
        return self.dataset.get_labeled_mask()
    
    def evaluate(self, score, *args, **kwargs):
        raise NotImplementedError()



class DefaultMetrics(ActiveLearningRecord):
    def __init__(self, dataset, labeler, run_path, post_training_cost=1, dummy_last=0):
        super().__init__(dataset, labeler, dummy_last)
        self.ptc = post_training_cost
        self.run_path = run_path  # saving runs

    def evaluate(self, score, score2):
        # we adapt the dataframe to allow recording probability
        df = pd.DataFrame({ "gold": self.gold, "score": score, "score2": score2, "known": self.known })
        df = df.assign(adjscore = ((df.gold - 0.5)*np.inf*df.known).fillna(0) + df.score ).sort_values('adjscore', ascending=False)

        goldcum = df.gold.cumsum()
        training_cost = self.known.sum()
        pos_got = (df.gold & df.known).sum()

        try:
            dfr80 = np.where(goldcum >= 0.8*self.npos)[0][0] + 1
        except IndexError:
            dfr80 = np.ceil(self.N * 0.8)
        
        try:
            dfr85 = np.where(goldcum >= 0.85*self.npos)[0][0] + 1
        except IndexError:
            dfr85 = np.ceil(self.N * 0.85)

        try:
            dfr90 = np.where(goldcum >= 0.9*self.npos)[0][0] + 1
        except IndexError:
            dfr90 = np.ceil(self.N * 0.9)

        try:
            dfr95 = np.where(goldcum >= 0.95*self.npos)[0][0] + 1
        except IndexError:
            dfr95 = np.ceil(self.N * 0.95)

        # print(df)
        df.to_csv(self.run_path)
        # print(dfr80, pos_got)
        return {
            # "P@5": df.gold[:5].mean(),
            # "P@20": df.gold[:20].mean(),
            # "P@50": df.gold[:50].mean(),
            "R-P": df.gold[:self.npos].mean(),

            "AP": (goldcum / np.arange(1,self.N+1)).mean(),

            "DFR@0.8": dfr80 / self.N,
            "DFR@0.9": dfr90 / self.N,
            "WSS@85": 0.85 - (dfr85 / self.N),
            "WSS@95": 0.95 - (dfr95 / self.N),

            # "CostAtDepth0.8": ((dfr80 - pos_got)*self.ptc + training_cost) / (np.ceil(self.npos * 0.8) * self.ptc),
            # "CostAtDepth0.9": ((dfr90 - pos_got)*self.ptc + training_cost) / (np.ceil(self.npos * 0.9) * self.ptc)
        }

class CostModelingTests(ActiveLearningRecord):
    def __init__(self, dataset, labeler, pos_control_masks, target_recall, dummy_last=0):
        super().__init__(dataset, labeler, dummy_last)
        for m in pos_control_masks:
            assert m.shape[0] == self.N

        self.pos_control_masks = {f'control_{j}': pos_control_masks[j] for j in range(len(pos_control_masks))}
        self.sampled_control_pos = sum([ m.sum() for m in pos_control_masks ])
        self.target_recall = target_recall
    
    @property
    def num_controls(self):
        return len(self.pos_control_masks)

    def evaluate(self, score):
        df = pd.DataFrame({ **self.pos_control_masks,
                            **{'gold': self.gold, 'score': score, 'known': self.known}
                          })
        df = df.assign(adjscore=((df.gold-0.5)*np.inf*df.known).fillna(0) + df.score ).sort_values('adjscore', ascending=False)
        
        numbers = {}

        # if self.sampled_control_pos >= df.gold.sum():
        #     # store all positives instead
        #     control_scores = {-1: df[['score', 'known']][ df['gold'] ] }
        # else:
        #     control_scores = {
        #         j: df[['score', 'known']][ df[f'control_{j}'] ]
        #         for j in range(self.num_controls)
        #     }
        pos_scores = df[['score', 'known']][ df['gold'] ]

        for tr in self.target_recall:

            # oracle reviewed df
            ordf_above = df.iloc[: (df.gold.cumsum() > df.gold.sum()*tr).values.argmax() + 1]
            ordf_below = df.iloc[(df.gold.cumsum() > df.gold.sum()*tr).values.argmax() + 1:]
            # -1 for oracle
            numbers[(tr, -1, 'training', 'pos')] = (df.known & df.gold).sum()
            numbers[(tr, -1, 'training', 'neg')] = (df.known & ~df.gold).sum()
            numbers[(tr, -1, 'post-training-above', 'pos')] = (~ordf_above.known & ordf_above.gold).sum()
            numbers[(tr, -1, 'post-training-above', 'neg')] = (~ordf_above.known & ~ordf_above.gold).sum()
            numbers[(tr, -1, 'post-training-below', 'pos')] = (~ordf_below.known & ordf_below.gold).sum()
            numbers[(tr, -1, 'post-training-below', 'neg')] = (~ordf_below.known & ~ordf_below.gold).sum()
            
            for j in range(self.num_controls):
                cdf = df[ df[f'control_{j}'] ] # control set df

                implied_cutoff = cdf.iloc[(cdf.gold.cumsum() > cdf.gold.sum()*tr).values.argmax()].score
                crdf_above = df[ df.adjscore >= implied_cutoff ] # reviewed df above control set implied cutoff
                crdf_below = df[ df.adjscore < implied_cutoff ] # df below control set implied cutoff
                
                numbers[(tr, j, 'control-found', 'pos')] = (cdf.known & cdf.gold).sum()
                numbers[(tr, j, 'control-found', 'neg')] = (cdf.known & ~cdf.gold).sum() # should be 0
                
                numbers[(tr, j, 'post-training-above', 'pos')] = (~crdf_above.known & crdf_above.gold).sum()
                numbers[(tr, j, 'post-training-above', 'neg')] = (~crdf_above.known & ~crdf_above.gold).sum()
                numbers[(tr, j, 'post-training-below', 'pos')] = (~crdf_below.known & crdf_below.gold).sum()
                numbers[(tr, j, 'post-training-below', 'neg')] = (~crdf_below.known & ~crdf_below.gold).sum()

        # backward compatible
        if len(self.target_recall) == 1:
            return { k[1:]: v for k, v in numbers.items() }, pos_scores

        return numbers, pos_scores
        
class HeuristicsStoppingInfo(ActiveLearningRecord):
    def __init__(self, dataset, labeler, target_recall, dummy_last=0):
        super().__init__(dataset, labeler, dummy_last)

        self.target_recall = target_recall
        self.previous_score = None


    def evaluate(self, score, score2):
        df = pd.DataFrame({ 'gold': self.gold.astype(bool), 'score': score, 'score2': score2, 'known': self.known })
        df = df.assign(adjscore=((df.gold-0.5)*np.inf*df.known).fillna(0) + df.score ).sort_values('adjscore', ascending=False)
        
        numbers = {}
        uni_structure = [1, 1, 1, 1]
        exp_structure = [10, 10, 1, 1]
        pos_scores = df[['score', 'known']][ df['gold'] ]

        for tr in self.target_recall:
            # oracle reviewed df
            ordf_above = df.iloc[: (df.gold.cumsum() > df.gold.sum()*tr).values.argmax() + 1]
            ordf_below = df.iloc[(df.gold.cumsum() > df.gold.sum()*tr).values.argmax() + 1:]
            # -1 for oracle
            numbers[(tr, 'training', 'pos')] = (df.known & df.gold).sum()
            numbers[(tr, 'training', 'neg')] = (df.known & ~df.gold).sum()
            numbers[(tr, 'post-training-above', 'pos')] = (~ordf_above.known & ordf_above.gold).sum()
            numbers[(tr, 'post-training-above', 'neg')] = (~ordf_above.known & ~ordf_above.gold).sum()
            numbers[(tr, 'post-training-below', 'pos')] = (~ordf_below.known & ordf_below.gold).sum()
            numbers[(tr, 'post-training-below', 'neg')] = (~ordf_below.known & ~ordf_below.gold).sum()

            record = (numbers[(tr, 'training', 'pos')], numbers[(tr, 'training', 'neg')],
                      numbers[(tr, 'post-training-above', 'pos')],
                      numbers[(tr, 'post-training-above', 'neg')])

            # we add the review cost recording on-the-fly
            uni_cost = sum(num * ucost for num, ucost in zip(record, uni_structure))
            exp_train = sum(num * ucost for num, ucost in zip(record, exp_structure))
            numbers[(tr, 'uni-cost')] = uni_cost
            numbers[(tr, 'exp-train')] = exp_train

        # summary of the rank list
        est_probs = expit(score)
        unknown_est_probs = est_probs[~self.known]
        has_unknown = unknown_est_probs.size > 0 
        numbers[(-1, 'scores-info', 'estprob-sum')] = est_probs.sum()
        numbers[(-1, 'scores-info', 'estprob-std')] = est_probs.std()
        numbers[(-1, 'scores-info', 'estprob-bivar')] = (est_probs*(1-est_probs)).sum()
        numbers[(-1, 'scores-info', 'estprob-median')] = np.median( est_probs )

        numbers[(-1, 'scores-info', 'estprob-sum-unknown')] = unknown_est_probs.sum() if has_unknown else None
        numbers[(-1, 'scores-info', 'estprob-std-unknown')] = unknown_est_probs.std() if has_unknown else None
        numbers[(-1, 'scores-info', 'estprob-bivar-unknown')] = (est_probs*(1-est_probs))[ ~self.known ].sum() if has_unknown else None
        numbers[(-1, 'scores-info', 'estprob-median-unknown')] = np.median( unknown_est_probs ) if has_unknown else None

        numbers[(-1, 'scores-info', 'max-uncertain-unknown')] = np.abs( unknown_est_probs - 0.5 ).max() if has_unknown else None
        numbers[(-1, 'scores-info', 'max-score-unknown')] = score[ ~self.known ].max() if has_unknown else None
        numbers[(-1, 'scores-info', 'expected-error-unknown')] = ( unknown_est_probs * (1-unknown_est_probs) ).sum() if has_unknown else None

        # numbers[(-1, 'scores-info', 'likelihood-ratio-known')] = self.gold[self.known].sum() / est_probs[self.known].sum()

        # if self.previous_score is not None:
        #     numbers[(-1, 'scores-info', 'corr')] = np.correlate(score, self.previous_score)[0]
        #     numbers[(-1, 'scores-info', 'corr-unknown')] = np.correlate(score[~self.known], self.previous_score[~self.known])[0] if has_unknown else None
        #     numbers[(-1, 'scores-info', 'corrcoef')] = np.corrcoef(score, self.previous_score)[0,1]
        #     numbers[(-1, 'scores-info', 'corrcoef-unknown')] = np.corrcoef(score[~self.known], self.previous_score[~self.known])[0,1] if has_unknown else None

        #     # ranking similarity
        #     current_ranking = score.argsort()
        #     previous_ranking = self.previous_score.argsort()

        #     numbers[(-1, 'rank-info', 'kendaltau')] = kendalltau(current_ranking, previous_ranking)
        #     numbers[(-1, 'rank-info', 'spearmanr')] = spearmanr(current_ranking, previous_ranking)
        #     numbers[(-1, 'rank-info', 'kendaltau-unknown')] = kendalltau(current_ranking[~self.known], previous_ranking[~self.known]) if has_unknown else None
        #     numbers[(-1, 'rank-info', 'spearmanr-unknown')] = spearmanr(current_ranking[~self.known], previous_ranking[~self.known]) if has_unknown else None

            
        # backward compatible
        if len(self.target_recall) == 1:
            return { k[1:]: v for k, v in numbers.items() }, pos_scores

        self.previous_score = score
        return numbers, pos_scores

class WithControlSet(object):
    def __init__(self, dataset, rel, control_mask, post_training_cost):
        self.dataset = dataset
        self.all_gold = rel
        self.control_mask = control_mask
        self.ptc = post_training_cost

        self.npos_all = self.all_gold.sum()
        self.npos_control = self.all_gold[ self.control_mask ].sum()
        self.npos_pool = self.npos_all - self.npos_control

        self.control_size = self.control_mask.sum()

    @property 
    def N(self):
        return self.all_gold.shape[0]

    @property
    def rich(self):
        return self.npos_all / self.N
    
    @property
    def pool_known(self):
        # returns a mask where the known pooled documents are True
        # and others are False
        m = np.zeros( self.N, dtype=bool )
        m[ self.all_gold.index[ ~self.control_mask ][ self.dataset.get_labeled_mask() ] ] = True
        return m

    def evaluate(self, score):
        df = pd.DataFrame({
            "gold": self.all_gold, "control": self.control_mask,
            "pool_known": self.pool_known,
            "known": self.control_mask | self.pool_known, # real known 
            "score": score
        })
        df = df.assign(adjscore = ((df.gold - 0.5)*np.inf*df.known).fillna(0) + df.score ).sort_values(['adjscore', 'score'], ascending=False)

        # control set
        control_cumpos = df[ df.control ].gold.cumsum()
        control_depth = _depth( control_cumpos, 0.8, self.npos_control )
        control_cut_score = df[ df.control ].iloc[ control_depth ].score
        control = {
            "R-P-control": df[ df.control ].gold[:self.npos_control].mean(),
            "dfr@0.8-control": (control_depth + 1) / self.control_size, 
            "cutScore-control": control_cut_score
        }

        inferred_post_review_set = np.zeros(self.N, dtype=bool)
        inferred_post_review_set[ ~df.known & (df.score >= control_cut_score) ] = True

        pool_cumpos = df[ ~df.control ].gold.cumsum()
        pool_depth = _depth(pool_cumpos, 0.8, self.npos_pool)
        sampled_doc = df.pool_known.sum()
        pool_knownpos = ( df.pool_known & df.gold ).sum()
        pool = {
            "R-P-pool": df[ ~df.control ].gold[:self.npos_pool].mean(),
            "dfr@0.8-pool": (pool_depth + 1) / ( self.N - self.control_size ),

            # with an oracle that tells exactly where to stop review
            "CostAtDepth0.8-pool-oracle": (pool_depth - pool_knownpos)*self.ptc + sampled_doc,
            # infer the depth from control set
            "CostAtDepth0.8-pool-inferred": inferred_post_review_set.sum()*self.ptc + sampled_doc
        }

        
        entire_set = {
            # on the entire set
            "R-P-all": df.gold[:self.npos_all].mean(), 

            # important to see how the ability of control set goes down -- sequentail bias
            "recallOnAcutalReviewed-all": ((inferred_post_review_set | df.known) & df.gold).sum() / self.npos_all
        }


        return {
            **control, **pool, **entire_set
        }
    
def _depth(cumpos, recall, npos):
    try:
        return np.where( cumpos >= recall * npos )[0][0]
    except IndexError:
        return np.ceil( cumpos.size * recall )
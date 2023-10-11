# Goldilocks-Reproduce-Repository
This is the repo for the Reproducibility track Paper *A Reproducibility Study of Goldilocks: Just-Right Tuning of BERT for TAR*

# Datasets & Data Processing
How to get the datasets?
- **RCV1-v2**

  Please request the original dataset from https://trec.nist.gov/data/reuters/reuters.html, and check the necessary files provided under `./data-processing/rcv1-v2`
- **Jeb-Bush**
  
  Please e-mail the author found at https://web.archive.org/web/20160220052206/http://plg.uwaterloo.ca/~gvcormac/total-recall/, and check the necessary files provided under `./data-processing/jeb-bush`
- **CLEF-TAR**
  
  Please check the original data at the official CLEF-TAR repo https://github.com/CLEF-TAR/tar. 
We provide the clean version of title & abstract of all docs in [`all_clean.jsonl`](https://drive.google.com/file/d/1kppExc6Wo81sCPYI2hkSsxO-ekgD2Qcc/view?usp=drive_link), and the processed clef collections used in the experiments can be downloaded from [clef_for_goldilocks](https://drive.google.com/file/d/1HcrOgTjAPm0cP8kUq6wdMa1v6eaztPwD/view?usp=sharing), the data processing process can be found under `./data-processing/clef`.

Large files under `./data-processing/rcv1-v2` and `./data-processing/jeb-bush` can be downloaded from [rcv1_path.txt](https://drive.google.com/file/d/1lk5o0tezGumV6ySLUSW_ADj8X37E4Y0e/view?usp=sharing), [id.txt](https://drive.google.com/file/d/152Xfr3pTtH_RXtlEEmP6v_N5J2SRz3cS/view?usp=sharing), and [athome1.md5](https://drive.google.com/file/d/1I-035nLg5HIZaD9hDOh_D0ZRLqkkG9N8/view?usp=sharing).

We also provide the [category information table](https://docs.google.com/spreadsheets/d/1N6_CqRNxu1Cn9VHvDRwmfdMNToRsLh5LoBQo1glWBks/edit?usp=sharing) used in this reproducibility paper of these three datasets.
# Hyperparameters
  
The optimizer and loss function are the same as in the original paper. 
- For BERT, we use `ADAM` as the optimizer with `no weight decay` and `no warm-up` period, and a `learning rate` of `5 * 10^-5` for `further pre-training` with `mask-language modelling`. The language model pre-training ranges from not pre-training at all on the target collection, to performing ten iterations over the collection. For `classification fine-tuning`, we use the `ADAM` optimizer with a `linear weight decay` of `0.01` with `50 warm-up steps` and an `initial learning rate` of `0.001`.
- For logistic regression, we use `L2 regularization` on the losses and fit to `convergence with default settings` in the active learning pipeline.
- In order to better fit our GPUs, we increased the `training` and `evaluation` `batch sizes` for BERT to `100` and `1000` respectively -- examining the original author's code, we could understand they instead used batch sizes of 26 and 600. Furthermore, we found that using `mixed precision` training (fp16) could **largely reduce the training time**.

# TAR with BERT
We adapted the codes from the original authors to our own experiment environment, please check the comments inside each file if necessary. 
- **Environment**
  
  For our setting: `3 * A100`, we use `python=3.8`, `cuda=11.7`, run
  ```bash
  conda env create -f env.yaml
  ```
  and install the dev version of `libact` for active learning as below.
  
  To install the env provided by the original authors, run
  ```bash
  conda create -n huggingface python=3.7
  conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
  pip install transformers
  conda install numpy scipy pandas scikit-learn nltk tqdm Cython joblib
  LIBACT_BUILD_HINTSVM=0 LIBACT_BUILD_VARIANCE_REDUCTION=0 pip install -e ~/repositories/libact
  ```
  **N.B.** The active learning library `libact` used is the dev version from the original authors https://github.com/eugene-yang/libact/tree/dev
- **Tokenization**
  
  - For reproducibility on **RCV1-v2** and **Jeb-Bush**, run
    
    ```bash
    python3 bert.tokenize.py rcv1
    ```
  and change the option for both `rcv1` and `jb`.
  
  - For generalizability on **CLEF collections**, run
    
    ```bash
    python3 clef.bert.tokenize.py 2017/test
    ```

    - With **BioLinkBert-base**, run
      
    ```bash
    python3 clef.biolinkbert.tokenize.py 2017/test
    ```
    and the option for CLEF collections ranges from `2017/test` to `2019/intervention/train`
  
- **Further pre-training with** `mlm-finetuning`
  
  - For reproducibility on **RCV1-v2** and **Jeb-Bush**, run
  
    ```bash
    python3 mlm-finetuning.py
    ```
  and check the comments inside for both `rcv1` and `jb`.
  
  - For generalizability on **CLEF collections**, run

    ```bash
    python3 clef-mlm-finetuning.py 2017/test
    ```
    - With **BioLinkBert-base**, check the comments inside `clef-mlm-finetuning.py`.
    
    and the option for CLEF collections ranges from `2017/test` to `2019/intervention/train`.
    
- **Reproduce goldilocks-tar**
  - For reproducibility on **RCV1-v2** and **Jeb-Bush**, run

    ```bash
    python3 al_exp.py --category 434 \
        --cached_dataset ./cache_new/jb50sub_org_bert-base.512.pkl.gz \
        --dataset_path  ./jb_info/ \
        --model_path ./mlm-finetune/bert-base_jb50sub_ep10 \
        --output_path  ./results/jb/ep10/ \
    ```
    and change the options for `rcv1` and `jb` with corresponding categories, `ep` refers to the further pre-training epochs from the previous stage. 
  - For generalizability on **CLEF collections**, run
    
    ```bash
    python3 clef17_al_exp.py --topic CD012019 \
        --cached_dataset ./cache_new/clef/clef2017_test_CD012019_org_bert-base.512.pkl.gz \
        --dataset_path  ./clef_info/2017/test/CD012019 \
        --output_path  ./results/clef/ep2/clef17_test/ \
        --batch_size 25 \
        --model_path ./mlm-finetune/clef/2017/bert-base_clef_2017_test_CD012019_ep2 \
    ```
    and the options according to CLEF collections range from `clef17_al_exp.py` to `clef19_al_exp.py` with corresponding topics.
    
    - With **BioLinkBert-base**, run corresponding `biolink` version such as 

    ```bash
    python3 clef17_biolink_exp.py --topic CD011984 \
      --cached_dataset ./cache_new/clef_biolink/clef2017_train_CD011984_biolink_bert-base.512.pkl.gz \
      --dataset_path  ./clef_info/2017/train/CD011984 \
      --output_path  ./results/biolink/ep0/clef17_train/ \
      --batch_size 25 \
      --model_path  michiyasunaga/BioLinkBERT-base \
    ```
    and the options according to CLEF collections range from `clef17_biolink_exp.py` to `clef19_biolink_exp.py` with corresponding topics.

# Baseline
- **Feature engineering**
  - For reproducibility on **RCV1-v2** and **Jeb-Bush**, run
    ```bash
    python3 lr.feature.py rcv1
    ```
    and change the option for both `rcv1` and `jb`.
    
  - For generalizability on **CLEF collections**, run
    ```bash
    python3 clef.lr.feature.py 2019/intervention/train
    ```
    and the option for CLEF collections ranges from `2017/test` to `2019/intervention/train`.
    
- **Reproduce goldilocks-tar baseline with logistic regression**
  - For reproducibility on **RCV1-v2** and **Jeb-Bush**, run
    ```bash
    python3 al_baseline.py --category 434 \
        --cached_dataset ./jb_info/jb_sampled_features.pkl \
        --dataset_path  ./jb_info \
        --output_path  ./results/baseline/jb/
    ```
    and change the options for `rcv1` and `jb` with corresponding categories.
    
  - For generalizability on **CLEF collections**, run
    ```bash
    python3 clef_al_baseline.py --dataset clef2017_train \
        --topic CD011984 \
        --batch_size 25 \
        --cached_dataset ./clef_info/lr_features/clef2017_train_CD011984_features.pkl \
        --dataset_path  ./clef_info/2017/train/CD011984 \
        --output_path  ./results/baseline/clef/clef17_train/
    ```
    and change the options for CLEF collection features ranges from `clef2017_train` to `clef2019_intervention_train` with corresponding topics.

# Analysis 
- **R-Precision**
  
  Please refer to `analysis_RP.py` under `./utils`.
- **Review Cost**
  
  Please refer to `analysis_cost.py` and `analysis_cost_bin.py` under `./utils`.
- **Statistical significance test**
  
  Please refer to `stat_RP.py` and `stat_cost.py` under `./utils`.

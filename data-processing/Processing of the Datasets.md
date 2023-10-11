# Processing of the Datasets
This document provides the details of processing the datasets for the reproducibility paper.
## Reproducibility with RCV1-v2 and Jeb Bush

To reproduce the work of **_Goldilocks: Just-Right Tuning of BERT for TAR_**, access to the raw data for the `RCV1-v2` and `Jeb Bush` datasets is required.



**The original Goldilocks paper** relies on a pre-processed `RCv1-v2` and `Jeb Bush` collection.

For `RCV1-v2`, we acknowledge that a publicly available processed `RCV1-v2` collection with `TF-IDF` features can be obtained through `scikit-learn`. However, **_the Goldilocks paper_** downsampled the collection and computed the text features for the baseline using BM25 within document-saturated term frequencies. Therefore, this public encoding is not suitable for our reproducibility experiment, and we require the raw materials for BERT tokenization. For `Jeb Bush`, no publicly released features or encodings are available for either baseline or BERT and **_the Goldilocks paper_** performed similar processing as for `RCV1-v2`.



We obtained the raw material (RCV1-v1) from NIST; this version does not include correction.

To convert `RCV1-v1` to `RCV1-v2`, we used the related information provided by **_RCV1: A New Benchmark Collection_** for Text Categorization Research: IDs of the 804,414 news documents in `RCV1-v2`, rectified codes in the categories. To fit with the code for the BERT experiments from **_the Goldilocks paper_**, we concatenated the title and body for each news document, and stored this raw text in a CSV file, with columns as "news id, title, text, code", and recorded the relevance information for each category in a separate CSV file, putting the 45 categories in the column instead of the raw text. We followed **_the Goldilocks paper_** to tokenize with WordPiece and truncate with 512 tokens due to the limitation of BERT input length. As **_the Goldilocks paper_** suggested, the leading passages of the news documents cover the most important information already. For computational efficiency in the TAR task, we followed **_the Goldilocks paper_** and the code provided to downsample the datasets with randomly selected indexes as a mask. We did not perform any specific text processing for the data for BERT, as there was no clear mention in the paper. However, for logistic regression, we performed basic text processing, referring to the original authors' previous paper **_On minimizing cost in legal document review workflows_** by lowercasing, removing punctuations, and separating words by whitespace. We also followed the original paper in applying the scikit-learn `CountVectorizer` first and with our own implementation of the saturated term frequency upon it, using all the words as features, as suggested by **_On minimizing cost in legal document review workflows_**, which relied upon the same datasets.



The `Jeb Bush` email dataset is not as clean and structured as the `RCv1-v2` dataset. We first de-duplicated the files from 290,099 to 274,124 following **_the Goldilocks paper's_** former work in filtering by md5 code, which gave us the exact number of unique emails as reported in the original paper. The raw materials are emails in their original content stored in text files, and there is no clear structure for all the texts. We can only follow **_the Goldilocks paper_** by selecting the first subject line and treating the remaining part as the body. We then concatenated them as the raw text. We did the same as `RCv1-v2` for storing data, relevance information, sample mask, BERT tokenization, and logistic regression features. **_the Goldilocks paper_** also suggested that including the most recent replies and content from emails, which are typically located at the beginning of the email, is sufficient.



## Generalisability with the CLEF-TAR collections

To test the generalisability of the method by **_the Goldilocks paper_**, we chose to use the `CLEF` collections from the biomedical domain. The data is publicly available on the [CLEF-TAR GitHub repository](https://github.com/CLEF-TAR/tar). Given the unique nature of the `CLEF` task, we treated each topic from each training/test split of each year as an individual dataset, rather than integrating the entire data and building categories with topics as in the two classification datasets used by **_the Goldilocks paper_**. Since our interest is in the actual relevant documents obtained on the fly and how much cost can be saved by using the active learning pipeline, we did not follow the training/test split as the `CLEF` dataset names suggest. Instead, we used the names to distinguish between the individual datasets.



We created a `JSONL` file that contains all the topic-document information in "pid, title, abstract" format from all the collections. We removed 5 missing PIDs and detected 11 cases with missing titles and 58,809 cases with missing abstracts. None of them had any overlap.


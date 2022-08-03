# NEWTS 🦎
### NEWs Topic-based Summarization Dataset

This repository contains the NEWTS dataset, created while researching topic-focused abstractive summarization. The dataset, which is based on the CNN/DailyMail dataset, features two human-written summaries per news article, where each summary focuses on a different topic within the article. The dataset also contains four prompt types paired with each news article: words, phrases, sentences and topic IDs. 

### Reading the dataset
`read.py` provides two functions that read in the csvs with the proper settings, assuming that the files are in the same directory as the code being run. An alternative path can also be passed in as an argument to `read_test()` or `read_train()`. 

```python
>>> from read import read_test, read_train
>>> newts_test = read_test()
>>> newts_train = read_train()
>>> example_article = newts_train.iloc[2263]
>>> example_article['words1']
'family, wife, daughter, husband, couple, pictured, friends, left, brother, friend,'
>>> example_article['sentences2']
'This topic is about pop songs, music fans rejoice, a singer and songwriter, the first track on the new album, doing a sound check, and singing along with a prerecorded track.'
>>> example_article['summary2']
'Dame Esther Rantzen has released an album of pop songs from the 1940s to the 1970s. The album is called Silver Linings, Songs of a Lifetime, and is being sold by Amazon, HMV and iTunes. The first track on the new album is a sing along with a prerecorded track.'
```

To explain each of the columns, `docId` can be mapped back to the original CNN/DailyMail dataset, and `AssignmentID` is our numbering of each task completed on MTurk. Then, for each row we have the `article` and its two summaries (`summary1` and `summary2`), with different prompting methods (`tid`, `words`, `phrases`, and `sentences`) that correspond with those summaries.
```python
>>> newts_train.columns
Index(['AssignmentId', 'docId', 'article', 'tid1', 'tid2', 'words1', 'words2',
       'phrases1', 'phrases2', 'sentences1', 'sentences2', 'summary1',
       'summary2'],
      dtype='object')
```

Pandas logic can be used to filter the dataset just as any other DataFrame object. For example, to get a DataFrame with all of the articles about music (topic 227), you can query
```python
newts_train.loc[(newts_train['tid1']==227) | (newts_train['tid2']==227)]
```

### Pre-trained LDA Topic Model
If you want to calculate your own scores for different topics defined in this paper, first download our pretrained LDA topic model [here](https://drive.google.com/file/d/1KJm_3bCpFSA2A7hbqPPyrkEBTCJWs6X-/view?usp=sharing), unzip it and place the folder into your directory. Then, use the helper `read_lda()` in `read.py` to get the LDA and dictionary objects, passing in the path to the folder that you just unzipped. You will need to have `gensim` installed (4.2.0 was the version used here).  

```python
>>> from read import read_lda
>>> lda, dic = read_lda(path="../path_to_folder/")
Loaded the LDA model
Loaded dictionary
```

### Evaluation Helpers
The file `topics.py` contains helpers that make calculating topical focus of summaries easier using our pretrained LDA model. 

- `topic_score` returns the probability of a specific topic in a document
- `ab_topic_diff_score` returns a measure of the difference between two given topics within a single document
- `doc_topics` retrieves all of the topics in a document and their probabilities 

```python
>>> from read import read_test, read_lda
>>> tst = read_test()
>>> article = tst.iloc[544]['article']
>>> sum1 = tst.iloc[544]['summary1']
>>> sum2 = tst.iloc[544]['summary2']
>>> lda, dic = read_lda()
Loaded the LDA model
Loaded dictionary
>>> from topics import topic_score, ab_topic_diff_score, doc_topics
>>> topic_score(61, sum1, lda, dic)
0.48479586809953273
>>> topic_score(101, sum2, lda, dic)
0.2633118692828942
>>> topic_score(112, sum2, lda, dic)
0.0
>>> ab_topic_diff_score(61, 101, sum1, lda, dic)
0.44630929105608924
>>> ab_topic_diff_score(101, 61, sum2, lda, dic)
0.12313773855286882
>>> doc_topics(article, lda, dic)
{61: 0.30035412198675426, 101: 0.16101886092983858, 178: 0.059208078169137465, 243: 0.052341754932947966, 165: 0.04935078427257754, 68: 0.040860301695817905, 199: 0.03572639028721966, 80: 0.03391943209662231, 89: 0.03314630124061601, 78: 0.02538953593935106, 168: 0.023327834677880924, 233: 0.023244942075707924, 7: 0.022448601058724964, 142: 0.022379338418076773, 112: 0.01907594198194806, 136: 0.01593389564869909, 196: 0.013231417373290753, 96: 0.012689071430317729, 73: 0.01216818616271325, 239: 0.011652188283954154, 50: 0.01156249021143494, 87: 0.010487772505659507}
```


### Citation
We introduced NEWTS at ACL 2022. The citation for the dataset is:

```
@inproceedings{bahrainian-etal-2022-newts,
    title = "{NEWTS}: A Corpus for News Topic-Focused Summarization",
    author = "Bahrainian, Seyed Ali  and
      Feucht, Sheridan  and
      Eickhoff, Carsten",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.42",
    doi = "10.18653/v1/2022.findings-acl.42",
    pages = "493--503"
}
```

### License:
The NEWTS dataset and other content of this repository are shared under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license. 

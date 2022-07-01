# NEWTS
### NEWs Topic-based Summarization Dataset

This repository presents the NEWTS dataset created during a research project on Topic-focused Abstractive Summarization. The dataset, which is based on the CNN/Dailymail dataset, features two human-written summaries per news article each focusing on a different topic. The dataset also contains four prompt types paired with each news article, namely, topic words, topic phrases, topic sentences and topic IDs. 

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

### Reading the dataset:
The dataset can be read using the following sample code:

```
>>> newts_test = pd.read_csv('NEWTS_test_600.csv', encoding='utf-8', index_col=[0])
>>> newts_train = pd.read_csv('NEWTS_train_2400.csv', encoding='utf-8', index_col=[0])


>>> example_article = newts_train.iloc[2263]
>>> example_article['words1']
'family, wife, daughter, husband, couple, pictured, friends, left, brother, friend,'
>>> example_article['sentences2']
'This topic is about pop songs, music fans rejoice, a singer and songwriter, the first track on the new album, doing a sound check, and singing along with a prerecorded track.'
>>> example_article['summary2']
'Dame Esther Rantzen has released an album of pop songs from the 1940s to the 1970s. The album is called Silver Linings, Songs of a Lifetime, and is being sold by Amazon, HMV and iTunes. The first track on the new album is a sing along with a prerecorded track.'


# to get a df with all the articles about music, topic 227
newts_train.loc[(newts_train['tid1']==227) | (newts_train['tid2']==227)]
```

### Pre-trained LDA Topic Model
The model that we used in this project, both for computing the distribution of topics for article selection, as well as, evaluation of topic-focus scores  of the generated summaries can be downloaded [HERE](https://drive.google.com/file/d/1KJm_3bCpFSA2A7hbqPPyrkEBTCJWs6X-/view?usp=sharing). 

The model can be read using the following code:

```
   def readLDA (self, modelAddress):
        lda = gensim.models.ldamodel.LdaModel.load(modelAddress+'lda250.model', mmap = 'r')
        print "Loaded the LDA model"
        dictionary = corpora.Dictionary.load(modelAddress+'dictionary250.model', mmap = 'r')
        print "Loaded dictionary"
        return lda, dictionary
    def returnTopicLabels(self, lda, dictionary, inputDocs):
        outputs = []
        outputs_allTopics = []
        lda.minimum_phi_value = 0.01
        lda.per_word_topics = False
        for doc in inputDocs:
            vec_bow = dictionary.doc2bow(doc.split(' '))
            temp = lda[vec_bow]
            temp.sort(key=lambda x:x[1], reverse = True)
            topicsString = ''
            for item in temp:
                topicsString+= '\t'+str(item[0])+' '+str(item[1])
            outputs_allTopics.append(topicsString)
            try:
                string = str(temp[0][0])
            except:
                string = 'nan'
                print('No topics exist for this document')
            outputs.append(string)
        return outputs, outputs_allTopics
```

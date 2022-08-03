"""Defines a number of functions that can be used to evaluate summaries based on
whether they are focused on a given topic or not. 

Requires lda model to be loaded in (see README for details):
    lda, dictionary = readLDA('path/to/model')
"""

# Helper function that returns dictionary of topic ids mapped to prevalence of
# that topic within the given document. 
#
# Inputs: 
# - document: string containing a summary or article to calculate topics for 
# - lda: gensim.models.ldamodel.LdaModel loaded in using README instructions
# - dictionary: gensim.corpora.dictionary.Dictionary loaded in following README
# 
# Outputs:
# - Dictionary with keys as topic ids and values as the prevalence of that topic in document
def doc_topics(document, lda, dictionary):
    lda.minimum_phi_value = 0.01
    lda.per_word_topics = False
    vec_bow = dictionary.doc2bow(document.split(' '))
    temp = lda[vec_bow]
    temp.sort(key=lambda x:x[1], reverse = True)
    return dict(temp)

# Returns the prevalence of the given topic in a particular document. Can be applied
# to human-written summaries, machine-generated summaries, or the articles themselves.
#
# Inputs: 
# - tid: integer representing the id of the topic we want to find in `document`
# - document: string containing a summary or article to calculate score for 
# - lda: gensim.models.ldamodel.LdaModel loaded in using README instructions
# - dictionary: gensim.corpora.dictionary.Dictionary loaded in following README
#
# Returns: 
# - float representing the probability of tid for that document
def topic_score(tid, document, lda, dictionary):
    prevalences = doc_topics(document, lda, dictionary)

    if tid not in prevalences.keys():
        return 0.0
    else:
        return prevalences[tid]


# Given an article and a single summary for that article (each with separate target topic ids), 
# calculates a measure comparing the prevalence score for tid_a versus tid_b. Calculates the 
# difference between the two topic_scores, then normalizes by the sum of those scores.
# 
# This metric ranges between -1 and 1. A score of -1 means that the document is 
# completely focused on tid_b (score for tid_a is 0), whereas a score of 1 means 
# that the document is completely focused on tid_a. A score of 0 means that there 
# is no difference between the two topics. 
# 
# The original use case was having tid_a being the intended topic that the summary
# is supposed to be focused on, and tid_b being a different topic that the other 
# summary is supposed to be focused on. So in this case, higher would be better.
# 
# Inputs:
# - tid_a: integer representing the id of the first topic 
# - tid_b: integer representing the id of the second topic
# Returns:
# - number between -1 and 1, where higher means that the prevalence of tid_a is
#   higher, and 0 means that they are present with equal probabilities.
def ab_topic_diff_score(tid_a, tid_b, document, lda, dictionary):
    if tid_a == tid_b:
        return 0

    a = topic_score(tid_a, document, lda, dictionary)
    b = topic_score(tid_b, document, lda, dictionary)
    return 0 if (a == 0 and b == 0) else (a - b) / (a + b) 

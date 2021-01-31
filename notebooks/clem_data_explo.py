# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # Topic modelling from Ireland schools inspection reports

# <markdowncell>

# ## Imports

# <codecell>

import pandas as pd
import os
from nltk.corpus import stopwords 
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# <markdowncell>

# ## text preprocessing function

# <codecell>

def clean(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ') # Remove Punctuation
    lowercased = text.lower() # Lower Case
    tokenized = word_tokenize(lowercased) # Tokenize
    words_only = [word for word in tokenized if word.isalpha()] # Remove numbers
    stop = set(stopwords.words('english')) # Make stopword list
    
    newStopWords = set(['school','schools','learning','student','students','pupil','pupils','teacher','teachers','management','managements','teaching','support','suppports', 'lesson','lessons','boards', 'board'])
    stop = stop.union(newStopWords)
 
    without_stopwords = [word for word in words_only if word not in stop] # Remove Stop Words
    
    lemma=WordNetLemmatizer() # Initiate Lemmatizer
    lemmatized = [lemma.lemmatize(word) for word in without_stopwords] # Lemmatize
   
    return lemmatized

# <markdowncell>

# ## from PDF to raw text

# <codecell>



# <markdowncell>

# ## starting with a small sample of 5 texts from 5 PDFs

# <codecell>

list_sample = os.listdir('../../IIPE-data/IIPE/data_sample/plain_text_sample')
list_sample = [file for file in list_sample if file[-4:] == '.txt']
list_sample

# <markdowncell>

# ## putting the text into a DataFrame

# <codecell>

cols , dfs = [], []
for file in list_sample:
    cols.append('../../IIPE-data/IIPE/data_sample/plain_text_sample/'+file)
for file in list_sample:
    dfs.append(pd.read_csv('../../IIPE-data/IIPE/data_sample/plain_text_sample/'+file,header=None,sep='\t',))
    
sample_df = pd.concat(dfs,)
sample_df.reset_index(inplace=True)   
sample_df.drop(columns='index',inplace=True)
sample_df.rename(columns={0:'text'},inplace=True)
sample_df



# <markdowncell>

# ## preprocessing the text

# <codecell>

# Apply to all texts
sample_df['clean_text'] = sample_df['text'].apply(clean)

sample_df.drop(columns='text',inplace=True)

sample_df.head()

# <codecell>

sample_df.clean_text[0]

# <markdowncell>

# ## feature engineering?

# <codecell>



# <markdowncell>

# ## vectorizing the data: TF-IDF  


# <codecell>

# the argument passed to the TF-IDF vectorizer must be a list of strings
corpus = []
for text_ in list(sample_df.clean_text):
    corpus.append(' '.join(text_))
corpus

# <codecell>

tf_idf_vectorizer = TfidfVectorizer(max_df=0.8,min_df=0.1,ngram_range=(2,3))

X = tf_idf_vectorizer.fit_transform(corpus)

# <codecell>

pd.DataFrame(X.toarray(),columns = tf_idf_vectorizer.get_feature_names())

# <codecell>

lda_model = LatentDirichletAllocation(n_components=3).fit(X)

def print_topics(model, vectorizer):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-11:-1]])
        

print_topics(lda_model, tf_idf_vectorizer)

# <codecell>

example = ["please have a positive attitude towards assessment data and written work"]

example_vectorized = tf_idf_vectorizer.transform(example)

lda_vectors = lda_model.transform(example_vectorized)
print(lda_vectors)

#print("topic 0 :", lda_vectors[0][0])
#print("topic 1 :", lda_vectors[0][1])

# <markdowncell>

# ## vectorizing the data: n-grams

# <codecell>



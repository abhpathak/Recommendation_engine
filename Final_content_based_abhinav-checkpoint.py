
# coding: utf-8

# #### Load required libraries

# In[ ]:

get_ipython().magic('matplotlib inline')
# import necessary libraries and specify that graphs should be plotted inline. 
import numpy as np
import pandas as pd
import graphlab
from pandas import Series,DataFrame
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
import json
from pprint import pprint
import plotly.plotly as py
from plotly.tools import FigureFactory as FF
from ggplot import *
import plotly.graph_objs as go
import json
import graphlab as gl


# In[ ]:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from pandas import Series,DataFrame
import nltk
import string
import os
from nltk.stem.porter import PorterStemmer


# #### Join Restaurant and Review Data

# In[4]:

restaurant_review = pd.merge(review_LV, restaurant_LV, how = 'inner',left_on =  'business_id', right_on = 'business_id')


# #### Joining all the reviews for each business

# In[ ]:

restaurant_review = restaurant_review.groupby('business_id')['text'].apply(' '.join).reset_index()


# In[ ]:

restaurant_review = restaurant_review[['business_id','text']]


# #### Remove not useful characters from text

# In[ ]:

restaurant_review['text'] = restaurant_review.text.apply(lambda x: x.replace("\\",""))
restaurant_review['text'] = restaurant_review.text.apply(lambda x: x.replace(","," "))#.replace("\\","")
restaurant_review['text'] = restaurant_review.text.apply(lambda x: x.replace("\n\n"," "))#.replace("\\","")
restaurant_review['text'] = restaurant_review.text.apply(lambda x: x.replace("u'",""))#.replace("\\","")


# #### Functions for stemming and Tokenization

# In[ ]:

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokenizer = RegexpTokenizer(r'([a-zA-Z]+)')
    tokens = tokenizer.tokenize(text)
    stemmer = PorterStemmer()
    stems = stem_tokens(tokens, stemmer)
    return stems


# #### Get TFIDF matrix

# In[ ]:

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df = 0, stop_words = 'english',tokenizer=tokenize,max_features = 120)


# In[ ]:

corpus = []
for line in sorted(restaurant_review.itertuples(), key=lambda t: int(t[0])):
    corpus.append(line[2])


# In[ ]:

tfidf_matrix =  tf.fit_transform(corpus)
feature_names = tf.get_feature_names()


# #### Create TFIDF DataFrame

# In[ ]:

tfidf_dataframe = pd.DataFrame(tfidf_matrix.todense(), columns = feature_names)


# #### Add TFIDF data to the table with restaurant attributes

# In[ ]:

content_based_df = pd.merge(restaurant_review, tfidf_dataframe, left_on = 'business_id', right_on='business_id', how = 'left')


# #### Replace spaces in column names with underscore

# In[ ]:

new_col = []
for col in content_based_df.columns.values:
    colx = col.replace (" ", "_")
    coly = colx.replace("-","_")
    new_col.append(coly)
    
content_based_df.columns = new_col


# #### Impute Missing Values

# In[ ]:

#IMPUTING MISSING VALUES
#Some assumptions are made here hoping they will not have a major effect (low counts of missing values)

#content_based_df[content_based_df['Acc_credit_card'] == True]
content_based_df['Good_For'] = content_based_df['Good_For'].replace('N/A',0)
content_based_df['Acc_credit_card'] = content_based_df['Acc_credit_card'].replace('N/A',False)
content_based_df['Alcohol'] = content_based_df['Alcohol'].replace('N/A','unknown')
content_based_df['Ambience'] = content_based_df['Ambience'].replace('N/A',0)
content_based_df['Delivery'] = content_based_df['Delivery'].replace('N/A',False)
content_based_df['Parking'] = content_based_df['Parking'].fillna(0)
content_based_df['Price_Range'] = content_based_df['Price_Range'].replace('N/A',2.5)
content_based_df['Take_out'] = content_based_df['Take_out'].replace('N/A',False)
content_based_df['Takes_Reservations'] = content_based_df['Takes_Reservations'].replace('N/A',False)
content_based_df['Waiter_Service'] = content_based_df['Waiter_Service'].replace('N/A',False)
content_based_df['Noise_Level'] = content_based_df['Noise_Level'].replace('N/A','average')
content_based_df['Outdoor_Seating'] = content_based_df['Outdoor_Seating'].replace('N/A',False)

#review[['Take-out']]
#rest_revw.columns.values


# #### Seperate user item interaction data

# In[ ]:

observation_data = content_based_df[["business_id","user_id","stars_x"]]


# #### Get item attribute data

# In[ ]:

item_data = pd.merge(restaurant_review,restaurants_LV, left_on = 'business_id', right_on='business_id', how = 'inner')


# In[5]:

item_data = item_data[['business_id',
       'open', 'review_count', 'stars', 'Acc_credit_card', 'Alcohol',
       'Ambience', 'Delivery', 'Parking', 'Price_Range',
       'Take_out', 'Takes_Reservations', 'Waiter_Service', 'Noise_Level', 'Good_For',
       'Outdoor_Seating', 'alway', 'amaz',  'bad', 'bar',
        'beer', 'befor', 'best', 'better', 'bit', 'bread',
       'burger', 'busi', 'came', 'chees', 'chicken', 'come', 'cook', 'day',
       'definit', 'delici','dinner', 'dish',
        'drink', 'eat', 'enjoy', 'everyth', 'experi',
       'favorit', 'flavor', 'food', 'food_wa', 'fresh', 'fri', 'friend',
       'friendli', 'good', 'got', 'great', 'hot', 'hour', 'just',
       'know', 'like', 'littl', 'locat',  'lot', 'love', 'lunch',
        'make', 'meal', 'meat', 'menu', 'minut', 'nice', 'night',
       'onli', 'order', 'peopl', 'pizza', 'place', 'pretti', 'price',
       'realli', 'recommend', 'restaur', 'review', 'right', 'roll',
        'salad', 'sandwich', 'sauc', 'seat', 'serv',
       'server', 'servic', 'servic_wa', 'small', 'special',
       'staff', 'star', 'steak', 'sure', 'sushi', 'tabl', 'tast',
        'think', 'time', 'tri', 'vega',
       'visit', 'wa_good', 'wait', 'want', 'way']]


# ### Create SFrame using Graphlab

# In[ ]:

item_data_sf = gl.SFrame(item_data_x)
observation_data_sf = gl.SFrame(observation_data)


# In[ ]:

## Split the dataset into train and test


# In[ ]:

train, test = gl.recommender.util.random_split_by_user(observation_data_sf,item_id = "business_id", user_id = "user_id", max_num_users=2000)


# ### Build the model using training set

# In[ ]:

m = gl.recommender.item_content_recommender.create(item_data_sf,observation_data = train, target= "stars_x",item_id="business_id",user_id = "user_id",max_item_neighborhood_size=64)


# ### Evaluate RMSE on test

# In[ ]:

m.evaluate_rmse(test, target="stars_x")


# ### Evaluate Precision and Recall on test

# In[ ]:

m.evaluate_precision_recall(test)


#!/usr/bin/env python
# coding: utf-8

# # WeRateDogs Twitter Archive: Data Wrangling and Analysis
# 
# > #### By Reema Mohsin

# ## Gather Data

# In[2]:


#import all libraries necessary for project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import tweepy
import json

#magic function that will allow plots to be displayed in notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#create a DataFrame of the WeRateDogs twitter archive (file on hand)
df_archive = pd.read_csv('twitter-archive-enhanced.csv')


# In[4]:


#download file from link
url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'
response = requests.get(url)
with open('image_predictions.tsv',mode='wb') as file:
    file.write(response.content)


# In[5]:


#create DataFrame for downloaded data
df_image_predictions = pd.read_csv('image_predictions.tsv',delimiter = '\t')


# In[6]:


#access twitter API

####keys, secrets, and tokens have been REMOVED

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

#create API object
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


# In[7]:


#create a list from all the tweet ids in the archive
tweet_id_list = list(df_archive['tweet_id'])

#initiate a counter to track runtime
#count = 0
with open('tweet_json.txt','w') as file:
    for tweet_id in tweet_id_list:
        try:
            #extract tweet using api
            tweet = api.get_status(tweet_id, tweet_mode='extended')
            #write tweet to file
            json.dump(tweet._json,file)
            #write newline to file
            file.write('\n')
            #count=count+1
            #print(count)
        except Exception as e:
            #capture tweets that have been deleted
            print(e)


# In[6]:


#extract tweet id, retweet count, and favorite count from each json line by line
tweets = []
with open('tweet_json.txt', encoding='utf-8') as file:
    for line in file:
        tweet = json.loads(line)
        tweet_id = tweet['id']
        retweet_count = int(tweet['retweet_count'])
        favorite_count = int(tweet['favorite_count'])
        tweets.append({'tweet_id': tweet_id,
                       'retweet_count': retweet_count,
                       'favorite_count': favorite_count})


# In[7]:


#create DataFrame of tweet information
df_tweets = pd.DataFrame(tweets, columns = ['tweet_id','retweet_count','favorite_count'])
df_tweets.head()


# ## Assess Data

# In[27]:


df_archive.info()


# In[28]:


df_archive.head()


# In[90]:


df_archive.describe()


# In[80]:


#checks unique source values and counts
df_archive.source.value_counts()


# In[18]:


#returns numbers of names with all lowercase letters
df_archive.name.str.islower().sum()


# In[26]:


#returns lowercase names and their value counts
df_archive[df_archive['name'].str.islower()==True]['name'].value_counts()


# In[81]:


#finds rows with decimal values
df_archive[df_archive.text.str.contains(r"(\d+\.\d*\/\d+)")][['text', 'rating_numerator']]


# In[82]:


#returns labelling of tweets with doggo in text
df_archive[df_archive.text.str.contains("doggo")]['doggo'].value_counts()


# In[78]:


#checks to see how many tweets are labelled doggo
df_archive[df_archive['doggo']=='doggo'].shape[0]


# In[83]:


#returns labelling of tweets with floofer in text
df_archive[df_archive.text.str.contains("floofer")]['floofer'].value_counts()


# In[88]:


#checks to see how many tweets are labelled floofer
df_archive[df_archive['floofer']=='floofer'].shape[0]


# In[84]:


#returns labelling of tweets with pupper in text
df_archive[df_archive.text.str.contains("pupper")]['pupper'].value_counts()


# In[86]:


#checks to see how many tweets are labelled pupper
df_archive[df_archive['pupper']=='pupper'].shape[0]


# In[85]:


#returns labelling of tweets with puppo in text
df_archive[df_archive.text.str.contains("puppo")]['puppo'].value_counts()


# In[87]:


#checks to see how many tweets are labelled puppo
df_archive[df_archive['puppo']=='puppo'].shape[0]


# In[38]:


df_image_predictions.info()


# In[39]:


df_image_predictions.head()


# ### Quality
# 
# * 181 retweets are included
# * *tweet_id* and *rating_numerator* are int objects, *timestamp*, *retweeted_status_timestamp*, and *source* are string objects
# * *source* values contain reference to link
# * 109 values in the *name* column are invalid names
# * 6 rows are missing full decimal value of *rating_numerator* 
# * archive contains tweets that have now been deleted
# * rows not correctly labelled *doggo*, *floofer*, *pupper*, or *puppo*
# * *in_reply_to_status_id* and *in_reply_to_user_id* are unneeded values
# 
# 
# ### Tidiness
# 
# * *doggo*, *floofer*, *pupper*, and *puppo*, in 4 different columns
# * same observational unit represented on 3 different DataFrames

# ## Clean Data

# ###### Define
# `Merge` all three DataFrames on *tweet_id* to create a master table. This will also remove deleted tweets from the table.
# ###### Code

# In[164]:


df = df_archive.merge(df_tweets, on='tweet_id')
df = df.merge(df_image_predictions, how='left', on='tweet_id')
#create copy of dataset
df_master = df.copy()


# ###### Test

# In[146]:


df_master.info()


# Before fixing the second tidiness issue, I will first address the quality issue related to *doggo*, *floofer*, *pupper*, and *puppo* mislabelling.
# 
# ###### Define
# 
# For each stage, create a mask using `str.contains`. Then use `loc` to set the values of those rows with the proper stage name. Then invert the mask with `~` and use `loc` to the set the values of those rows with 'None'. Repeat for all stages.
# 
# ###### Code

# In[165]:


doggo_mask = df_master.text.str.contains("doggo")
df_master.loc[doggo_mask,'doggo'] = 'doggo'
not_doggo_mask = ~df_master.text.str.contains("doggo")
df_master.loc[not_doggo_mask,'doggo'] = 'None'

floofer_mask = df_master.text.str.contains("floofer")
df_master.loc[floofer_mask,'floofer'] = 'floofer'
not_floofer_mask = ~df_master.text.str.contains("floofer")
df_master.loc[not_floofer_mask,'floofer'] = 'None'

pupper_mask = df_master.text.str.contains("pupper")
df_master.loc[pupper_mask,'pupper'] = 'pupper'
not_pupper_mask = ~df_master.text.str.contains("pupper")
df_master.loc[not_pupper_mask,'pupper'] = 'None'

puppo_mask = df_master.text.str.contains("puppo")
df_master.loc[puppo_mask,'puppo'] = 'puppo'
not_puppo_mask = ~df_master.text.str.contains("puppo")
df_master.loc[not_puppo_mask,'puppo'] = 'None'


# ###### Test

# In[166]:


#checks if the number of rows containing the stage name in text equals the number of rows labelled with that stage name
(df_master[df_master.text.str.contains("doggo")]['doggo'].value_counts()[0] == df_master.doggo.value_counts()[1],
df_master[df_master.text.str.contains("floofer")]['floofer'].value_counts()[0] == df_master.floofer.value_counts()[1],
df_master[df_master.text.str.contains("pupper")]['pupper'].value_counts()[0] == df_master.pupper.value_counts()[1],
df_master[df_master.text.str.contains("puppo")]['puppo'].value_counts()[0] == df_master.puppo.value_counts()[1])


# ###### Define
# Create a new DataFrame containing only *tweet_id*, *doggo*, *floofer*, *pupper*, *puppo*. `Melt` stage columns. Then drop any rows that contain *'None'*. Next, combine the rows of tweets with multiple dog stages using a `lambda` function and `groupby`. Because this will return a *Series* object, use `to_frame()` to transform into a DataFrame and `merge` it with the master DataFrame of tweets. Rename new column and drop original stage columns.
# 
# ###### Code

# In[167]:


#new DataFrame with just stages
df_dogstage = df_master[['tweet_id','doggo','floofer','pupper','puppo']]
#melt stages
df_dogstage = pd.melt(df_dogstage, id_vars='tweet_id')


# In[168]:


#drop rows with no stage value
df_dogstage = df_dogstage[df_dogstage.value != 'None']
#combine rows of tweets with multiple stage values
df_dogstage = df_dogstage.groupby('tweet_id')['value'].apply(lambda x: "%s" % ', '.join(x))
#from Series to DataFrame
df_dogstage = df_dogstage.to_frame()


# In[169]:


#add newly cleaned column to df
df_master = df_master.merge(df_dogstage,how='left',on='tweet_id')
df_master = df_master.rename(columns={'value':'stage'})
df_master = df_master.drop(['doggo','floofer','pupper','puppo'], axis = 1)


# ###### Test

# In[170]:


#there should be no 'None'
df_master.stage.value_counts()


# In[153]:


#checks that columns are gone
df_master.info()


# ###### Define
# Remove *in_reply_to_status_id* and *in_reply_to_user_id* columns using `drop`.
# 
# ###### Code

# In[171]:


#dropping unneeded columns
df_master = df_master.drop(columns = ['in_reply_to_status_id','in_reply_to_user_id'], axis=1)


# ###### Test

# In[172]:


df_master.info()


# ###### Define
# Remove rows where *retweeted_status_id* is *null*. Then use `drop` to remove the columns associated with retweets.
# 
# ###### Code

# In[173]:


df_master = df_master[df_master['retweeted_status_id'].isnull()]
df_master = df_master.drop(columns = ['retweeted_status_id','retweeted_status_user_id',
                                      'retweeted_status_timestamp'], axis=0)


# ###### Test

# In[157]:


df_master.info()


# ###### Define
# Isolate source from *source* column by applying a `lambda` function that uses `str.split()`.
# 
# ###### Code

# In[174]:


#check structure of source
df_master.source.unique()


# In[175]:


#split on '>' character and remove last 3 '</a' characters
df_master['source'] = df_master['source'].apply(lambda x: x.split('>')[1][:-3])


# ###### Test

# In[160]:


df_master['source'].value_counts()


# ###### Define
# Cast *tweet_id* to a *string*, *rating_numerator* to a *float*, and *source* to a *category* using `astype`. Convert *timestamp* column to a datetime object using `to_datetime`.
# 
# ###### Code

# In[176]:


df_master['timestamp'] = pd.to_datetime(df_master['timestamp'])
df_master.tweet_id = df_master.tweet_id.astype('str')
df_master.rating_numerator = df_master.rating_numerator.astype('float')
df_master.source = df_master.source.astype('category')


# ###### Test

# In[177]:


df_master.info()


# ###### Define
# Use `str.islower()` to find incorrect *name* values, and change those to *NaN* using `loc`.
# 
# ###### Code

# In[178]:


name_mask = df_master['name'].str.islower()
df_master.loc[name_mask,'name'] = np.nan


# ###### Test

# In[184]:


df_master[df_master.name.str.islower() == True].name.count()


# ###### Define
# Fix *rating_numerator* to include decimal values using `str.extract`.
# 
# ###### Code

# In[197]:


df_master['rating_numerator'] = df_master.text.str.extract(r"(\d+\.\d*)", expand=True)


# ###### Test

# In[198]:


df_master[df_master.text.str.contains(r"(\d+\.\d*\/\d+)")][['text', 'rating_numerator']]


# ## Store Data
# 
# Save the fully cleaned table in a csv file.

# In[199]:


df_master.to_csv('twitter_archive_master.csv')


# ## Analyze Data

# In[206]:


#returns top 10 objects and non-dog animals that were predicted for p1 of the image predictions
df_master.query('p1_dog == False').p1.value_counts().head(10)


# In[202]:


fig = df_master.source.value_counts().plot.bar(figsize=(8,8), rot=0, width=0.8, title = 'WeRateDogs Tweet Source')
fig.get_figure().savefig('WeRateDogs-bar-chart.png')


# As illustrated by the bar chart, an Iphone was the most used source for this account when creating tweets. The breakdown of each source is below.

# In[203]:


df_master.source.value_counts()


# In[204]:


df_master.timestamp.max()-df_master.timestamp.min()


# In[205]:


rolling_mean_rt = df_master.retweet_count.rolling(window=30).mean()
rolling_mean_fav = df_master.favorite_count.rolling(window=30).mean()
plt.figure(figsize=(12,8));
plt.plot(df_master.timestamp,rolling_mean_rt);
plt.plot(df_master.timestamp,rolling_mean_fav,color='red');
plt.xlabel('Time');
plt.ylabel('Count');
plt.title('WeRateDogs "Favorite" and "Retweet" Count Over Time');
plt.legend(['Retweet Count 30 Day MA','Favorite Count 30 Day MA']);
plt.savefig('WeRateDogs-line-graph.png')


# The resulting plot shows the trend of the account’s retweets and favorites over the 2015 to 2017 timeframe, which appears to be increasing over time, with favorites consistently outnumbering retweets. This trend aligns with the intuition that, over this 2-year span, the account gained more popularity and therefore was seen by more Twitter users and received more engagement. This increased engagement more notably pertains to favorites.

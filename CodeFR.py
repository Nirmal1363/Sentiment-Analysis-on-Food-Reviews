#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk


# In[3]:


df = pd.read_csv('Reviews.csv')


# In[4]:


print(df.shape)
df = df.head(500)
print(df.shape)


# In[5]:


df.head()


# In[6]:


ax = df['Score'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Count of reviews by Stars',
          figsize=(10,5))
ax.set_xlabel('Review stars')
plt.show()


# In[7]:


example = df['Text'][50]
print(example)


# In[8]:


nltk.download('punkt')


# In[9]:


tokens = nltk.word_tokenize(example)


# In[10]:


nltk.download('averaged_perceptron_tagger')


# In[11]:


tagged = nltk.pos_tag(tokens)
tagged[:10]


# In[12]:


nltk.download('maxent_ne_chunker')
nltk.download('words')


# In[13]:


entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()


# In[14]:


nltk.download('vader_lexicon')


# In[15]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()


# In[16]:


sia.polarity_scores('I am so happy!')


# In[17]:


sia.polarity_scores('This is the worst thing ever.')


# In[18]:


sia.polarity_scores(example)


# In[19]:


res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']  
    myid = row['Id'] 
    res[myid] = sia.polarity_scores(text)


# In[20]:


vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')


# In[21]:


vaders.head()


# In[22]:


sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compound Score by Amazon Star Reviews')
plt.show()


# In[23]:


fig, axs = plt.subplots(1, 3, figsize=(15, 5))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


# In[24]:


from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
get_ipython().system('pip3 install torch torchvision torchaudio')
import torch


# In[25]:


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# In[26]:


print(example)
sia.polarity_scores(example)


# In[27]:


encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2],
}
print(scores_dict)


# In[28]:


def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2],
    }
    return(scores_dict)


# In[29]:


res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']  
        myid = row['Id'] 
        res[myid] = sia.polarity_scores(text)
        vader_result = sia.polarity_scores(text)

        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id{myid}')


# In[30]:


results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id','neg': 'vader_neg','neu': 'vader_neu','pos': 'vader_pos'})
results_df = results_df.merge(df, how='left')


# In[54]:


results_df.columns


# In[32]:


sns.pairplot(data=results_df,
            vars=['vader_neg', 'vader_neu', 'vader_pos', 'roberta_neg',
                  'roberta_neu', 'roberta_pos'],
            hue='Score',
            palette='tab10') 
            


# In[33]:


results_df.query('Score == 1') \
    .sort_values('roberta_pos', ascending=False)['Text'].values[0]


# In[34]:


results_df.query('Score == 1') \
    .sort_values('vader_pos', ascending=False)['Text'].values[0]


# In[35]:


results_df.query('Score == 5') \
    .sort_values('roberta_pos', ascending=False)['Text'].values[0]


# In[36]:


results_df.query('Score == 5') \
    .sort_values('vader_pos', ascending=False)['Text'].values[0]


# In[37]:


from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis")


# In[38]:


sent_pipeline('I love Football')


# In[39]:


sent_pipeline('I hate you')


# In[40]:


sent_pipeline('I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells better. My Labrador is finicky and she appreciates this product better than  most.')


# In[41]:


sent_pipeline('Product arrived labeled as Jumbo Salted Peanuts...the peanuts were actually small sized unsalted. Not sure if this was an error or if the vendor intended to represent the product as "Jumbo".')


# In[42]:


sent_pipeline("This seems a little more wholesome than some of the supermarket brands, but it is somewhat mushy and doesn't have quite as much flavor either.  It didn't pass muster with my kids, so I probably won't buy it again.")


# In[43]:


sent_pipeline("McCann's Instant Irish Oatmeal, Variety Pack of Regular, Apples & Cinnamon, and Maple & Brown Sugar, 10-Count Boxes (Pack of 6)<br /><br />I'm a fan of the McCann's steel-cut oats, so I thought I'd give the instant variety a try. I found it to be a hardy meal, not too sweet, and great for folks like me (post-bariatric surgery) who need food that is palatable, easily digestible, with fiber but won't make you bloat.")


# In[44]:


sent_pipeline("Average service. Food was average")


# In[45]:


sent_pipeline("the food was bland")


# In[46]:


sent_pipeline("food could be better")


# In[ ]:





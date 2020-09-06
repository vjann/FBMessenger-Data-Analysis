

```python
import pandas as pd
import numpy as np
import glob
import datetime
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import emoji
```

# Creating DataFrame from Facebook Messages Raw Data

After downloading your messages from Facebook, you might be a little confused about it's structure. I'll try to lay it out.
1. Provide the directory to the inbox folder of your Facebook Raw Data
2. Using glob, loop through all folders, check for .json files, and add those paths

The reason we end up with a nested list is because **a single conversation can have multiple .json files**. This happens when if it's a very long conversation (only two or three of my conversations were split into multiple files). An example of that is shown through the output of the next cell.


```python
#Find all .json files of messages
inbox_dir = "../messages/inbox/*" #put the path to your own data. Inside the inbox directory are lots of sub-directories, one for each chat 

# Inside inbox_dir, there are 2 folders for each group or individual chats, one for messages and one for files, like images shared.
# The following only takes the .json files.
messages_paths = []
for f in glob.glob(inbox_dir): #for each folder
    chats_json = glob.glob(f + '/*.json')
    if not chats_json:# The folders for shared media have no .json files-- ignore those
        continue
    messages_paths.append(glob.glob(f + '/*.json')) #if your conversation is longer than 10k messages, there will be multiple .json files
```

You likely care about the longest conversations you have. An easy, approximate way to do this (without opening them) is to sort by the file sizes. I decided I really only cared about the top 100 I talked to the most.


```python
#Take the top 100 messages (lists of message files), sorted by total storage size (not number of messages). 
#Hence helper function total_size
def total_size(message_files):
    size = 0
    for file in message_files: # to combine conversation sizes for those that were split into multiple files
        size += os.path.getsize(file)
    return size
messages_paths.sort(reverse=True, key = total_size)
messages_paths = messages_paths[:100]
```


```python
#Read .json files into python. We end up with a list of python dictionaries I think (a list per person 
#because each person's convo are split into multiple files if too long (10,000 messages per file))
#messages_paths is list of file paths for json, messages_dict are the list of lists python dictionaries

messages_dict = []
for convos in messages_paths:
    data = []
    for convo in convos:
        with open(convo) as f:
            data.append(json.load(f))
    messages_dict.append(data)
```


```python
#split messages_dict into group convos and individual convos
individuals = [convos for convos in messages_dict if len(convos[0]['participants']) == 2]
#individuals_name is important for keeping track of whose dataframe is whose
individuals_name = [x[0]['participants'][0]['name'] for x in individuals]

#Aliasing names! for privacy. Remove as needed
aliases = []
with open('randomNames.txt') as file: 
    names = file.readlines()
    aliases = [name.rstrip('\n') for name in names]
np.random.shuffle(aliases)
individuals_name = [aliases[i] for i in range(len(individuals))]

groups = [convos for convos in messages_dict if len(convos[0]['participants']) > 2]
```

Cool! Now a conversation with each person is represented by a list of dictionaries. Let's take the person I talk to the most, and put it into something a little more familiar: a dataframe!

# Preprocessing Dataframe + Feature Engineering!
Now that we have a taste of what we can do, lets look at what else we can uncover.
Here's some information I'm going to extract and make explicit in the dataframe:
1. number of words in each message
2. Change timestamp into a much more useful time object
3. Whether a message is a response or followup
4. Message Sentiment!
For convenience sake, I'll also package everything into a neat function to apply for all my conversations instead of just my_dear_friend


```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

sid_obj = SentimentIntensityAnalyzer()
def create_df(lst_of_dict):
    """function creates dataframe for each convo"""
    def get_message_sentiment(message):
        result = sid_obj.polarity_scores(message)
        return result['pos'] - result['neg']
    
    #if conversation split into diff json files, combine them
    all_messages = pd.concat(pd.DataFrame.from_dict(x['messages']) for x in lst_of_dict[::-1])
    all_messages['content'] = all_messages['content'].fillna('')
    
    #change weird time_stamp into usable/understandable numbers and datetime objects
    all_messages['time_stamp'] = all_messages['timestamp_ms'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000))
    all_messages['hour'] = all_messages['time_stamp'].dt.hour + all_messages['time_stamp'].dt.minute/60
    time_stamp_diff = all_messages['time_stamp'].diff(periods=-1)
    all_messages['response_seconds'] = time_stamp_diff.dt.microseconds/1000000 + time_stamp_diff.dt.seconds
    all_messages['response_days'] = time_stamp_diff.dt.days.fillna(0)
    
    
    all_messages['num_words'] = all_messages['content'].str.findall(r"\w+").apply(lambda x: len(x))
    all_messages['message_sentiment'] = all_messages['content'].apply(get_message_sentiment)
    
    #hashing names because subsequent rolling must take in numbers
    all_messages['name_hashed'] = all_messages['sender_name'].apply(hash)
    all_messages['reply'] = np.roll(all_messages['name_hashed'].rolling(2).apply(lambda x: int(list(x)[0] - list(x)[1])) != 0, -1)
    
    #pick out relevant columns
    final_columns = ['sender_name', 'content', 'message_sentiment', 'num_words', 'time_stamp', 
         'hour', 'reply', 'response_seconds', 'response_days', 'reactions', 'sticker', 
         'photos', 'videos', 'files', 'gifs', 'share', 'type']
    # Not columns will be the same, e.g. you won't have a gifs column for a conversation where noone sent a gif
    all_messages = all_messages[[col for col in final_columns if col in all_messages.columns]] 
    return all_messages

#create dataframe for each individual convo
individual_df = [create_df(x) for x in individuals]
```

Wow, that's a lot. I'll break it down into chunks. First, we turn the list of json objects into dataframe, just as before. (The following code snippets are pulled from the function above for understanding, and will not run properly)


```python
#if conversation split into diff json files, combine them
all_messages = pd.concat(pd.DataFrame.from_dict(x['messages']) for x in lst_of_json[::-1])
all_messages['content'] = all_messages['content'].fillna('')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-36-c467f69a39b2> in <module>
          1 #if conversation split into diff json files, combine them
    ----> 2 all_messages = pd.concat(pd.DataFrame.from_dict(x['messages']) for x in lst_of_json[::-1])
          3 all_messages['content'] = all_messages['content'].fillna('')
          4 
          5 #change weird time_stamp into usable/understandable numbers and datetime objects


    NameError: name 'lst_of_json' is not defined


Then, we'll convert timestamp_ms (milliseconds after Jan 1st, 1970) into a datetime object, which allows us to easily find differences between two times, etc. I'm also extracting the hour each message to find daily trends and the time between messages (using the handy datetime object we just created).


```python
#change weird time_stamp into usable/understandable numbers and datetime objects
all_messages['time_stamp'] = all_messages['timestamp_ms'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000))
all_messages['hour'] = all_messages['time_stamp'].dt.hour + all_messages['time_stamp'].dt.minute/60
time_stamp_diff = all_messages['time_stamp'].diff(periods=-1)
all_messages['response_seconds'] = time_stamp_diff.dt.microseconds/1000000 + time_stamp_diff.dt.seconds
all_messages['response_days'] = time_stamp_diff.dt.days.fillna(0)
```

Then we find the message sentiment. We're using the simplest method possible: every individual word has a score associated with it. For example, "happy" would be positive, "agonizing" would be negative. We create a function that provides such a score to a single message and apply it to every row (every message).


```python
def get_message_sentiment(message):
    result = sid_obj.polarity_scores(message)
    return result['pos'] - result['neg']
all_messages['num_words'] = all_messages['content'].str.findall(r"\w+").apply(lambda x: len(x))
all_messages['message_sentiment'] = all_messages['content'].apply(get_message_sentiment)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-41-d1e03bbdc0e8> in <module>
          2     result = sid_obj.polarity_scores(message)
          3     return result['pos'] - result['neg']
    ----> 4 all_messages['num_words'] = all_messages['content'].str.findall(r"\w+").apply(lambda x: len(x))
          5 all_messages['message_sentiment'] = all_messages['content'].apply(get_message_sentiment)


    NameError: name 'all_messages' is not defined


Then, we check whether a message is a a reply to the other, or just one person sending multiple messages at a time. To do this, we check whether sender of the previous message is the same. This method is a little convoluted, but we hash the sender name, and check whether back-to-back rows have the same hashed sender_name (and therefore same sender).


```python
#hashing names because subsequent rolling must take in numbers
all_messages['name_hashed'] = all_messages['sender_name'].apply(hash)
all_messages['reply'] = np.roll(all_messages['name_hashed'].rolling(2).apply(lambda x: int(list(x)[0] - list(x)[1])) != 0, -1)
```

Lastly, get rid of the columns we don't need! And there we have it, a cleaned dataframe with more or less all the information that you could wish for.


```python
#pick out relevant columns
final_columns = ['sender_name', 'content', 'message_sentiment', 'num_words', 'time_stamp', 
     'hour', 'reply', 'response_seconds', 'response_days', 'reactions', 'sticker', 
     'photos', 'videos', 'files', 'gifs', 'share', 'type']
# Not columns will be the same, e.g. you won't have a gifs column for a conversation where noone sent a gif
all_messages = all_messages[[col for col in final_columns if col in all_messages.columns]] 
```

# Exploring the Data
This is what it looks like


```python
#each person is an entry in individual_df
someone = individual_df[0]

#name change for identity protection
someone.loc[someone['sender_name'] != 'Victor Jann', 'sender_name'] = individuals_name[0]
someone.sample(5, random_state=42)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sender_name</th>
      <th>content</th>
      <th>message_sentiment</th>
      <th>num_words</th>
      <th>time_stamp</th>
      <th>hour</th>
      <th>reply</th>
      <th>response_seconds</th>
      <th>response_days</th>
      <th>reactions</th>
      <th>sticker</th>
      <th>photos</th>
      <th>videos</th>
      <th>files</th>
      <th>gifs</th>
      <th>share</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4948</th>
      <td>Murrah</td>
      <td>lame stuff</td>
      <td>-0.737</td>
      <td>2</td>
      <td>2019-12-01 01:02:47.924</td>
      <td>1.033333</td>
      <td>True</td>
      <td>0.864</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Generic</td>
    </tr>
    <tr>
      <th>4875</th>
      <td>Victor Jann</td>
      <td>i mostly think the same</td>
      <td>0.000</td>
      <td>5</td>
      <td>2020-01-16 23:28:36.061</td>
      <td>23.466667</td>
      <td>False</td>
      <td>3.704</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Generic</td>
    </tr>
    <tr>
      <th>9951</th>
      <td>Victor Jann</td>
      <td>when's ur lab?</td>
      <td>0.000</td>
      <td>4</td>
      <td>2020-02-05 02:44:44.326</td>
      <td>2.733333</td>
      <td>False</td>
      <td>7.923</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Generic</td>
    </tr>
    <tr>
      <th>1671</th>
      <td>Murrah</td>
      <td>i think my license is a scam</td>
      <td>-0.381</td>
      <td>7</td>
      <td>2020-03-25 20:03:08.610</td>
      <td>20.050000</td>
      <td>False</td>
      <td>5.186</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Generic</td>
    </tr>
    <tr>
      <th>9129</th>
      <td>Murrah</td>
      <td>hm ok</td>
      <td>0.688</td>
      <td>2</td>
      <td>2020-03-14 15:15:31.041</td>
      <td>15.250000</td>
      <td>True</td>
      <td>97.876</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Generic</td>
    </tr>
  </tbody>
</table>
</div>



Let's see who I talk to the most.


```python
num_messages = [x.shape[0] for x in individual_df] #number of rows in each dataframe == number of messages
num_people_to_plot = 5
plt.figure(figsize=[7, 7])
sns.barplot(num_messages[:num_people_to_plot],individuals_name[:num_people_to_plot])
plt.title("Messages")
plt.ylabel("Person")
plt.xlabel("Number of Messages")
plt.savefig("../plots/top_friends_msg_count")
```


![png](FBMessagesPublic_files/FBMessagesPublic_24_0.png)


Previously, the 'message_sentiment' column is sentiment of each message. To figure out how positive each person is, we have to combine them so that each word is weighted equally. That involves multiplying sentiment by the number of words, summing
it up, and finally dividing by total number of words by that person.


```python
def determine_avg_sentiments(df):
    """returns tuple, where first item is Victor's avg sentiment, second item is the other's avg sentiment"""
    victor = df[df['sender_name'] ==  "Victor Jann"]
    not_victor = df[df['sender_name'] != "Victor Jann"]
    victor_sentiment = np.sum(victor['message_sentiment']*victor['num_words'])/np.sum(victor['num_words'])
    not_victor_sentiment = np.sum(not_victor['message_sentiment']*not_victor['num_words'])/np.sum(not_victor['num_words'])
    return (round(victor_sentiment, 4), round(not_victor_sentiment, 4))
names = []
sentiment_mine = []
sentiment_other = []
#plot for many different conversations
for i in range(10):
    other_person = individuals_name[i].split(' ')[0] #take the other person's first name
    my_sentiment, their_sentiment = determine_avg_sentiments(individual_df[i])
    names.append(other_person)
    sentiment_mine.append(my_sentiment)
    sentiment_other.append(their_sentiment)
#     print("Sentiment with {}: Victor:{}, {}:{}".format(other_person, senti_results[0], other_person, senti_results[1]))
temp = pd.DataFrame(data={'name':names, 'Victor': sentiment_mine, 'Other': sentiment_other})
temp = pd.melt(temp, id_vars='name', var_name='me_or_not', value_name='sentiment')
plt.figure(figsize=[10, 5])
sns.barplot(data=temp, x='name', y='sentiment', hue='me_or_not')

```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f752ee26da0>




![png](FBMessagesPublic_files/FBMessagesPublic_26_1.png)


Ever wonder when is peak talking time? You can really see the difference in my sleep schedule between college and high school


```python
def plot_time_distribution(df, name):
    plt.figure(figsize=[10, 5])
    sns.distplot(df['hour'], kde=False, label = 'hi', norm_hist=True, bins=np.arange(0, 25, 0.5))
    plt.title("Time of Day Distribution with {}".format(name))
    plt.ylabel("Proportion of Messages")
    plt.xticks(np.arange(0, 25, 1))
[plot_time_distribution(individual_df[i], individuals_name[i]) for i in range(2)]
```




    [None, None]




![png](FBMessagesPublic_files/FBMessagesPublic_28_1.png)



![png](FBMessagesPublic_files/FBMessagesPublic_28_2.png)


If we take another step back, we get the progression of how much you talk, every day


```python
someone['date'] = someone['time_stamp'].dt.date
plt.figure(figsize=[10, 7])
groupby_date = someone.groupby('date').count()
plt.bar(x=groupby_date.index, height=groupby_date['content'])
plt.title("Messages per Day")
plt.ylabel("Number of Messages")
plt.xlabel("Date")
# plt.show()
plt.savefig('../plots/messages_over_time')
```


![png](FBMessagesPublic_files/FBMessagesPublic_30_0.png)


### How quickly do people reply? 
Luckily, we calculated most of the heavy work when creating the dataframe, so we simply use response_seconds.


```python
def plot_response_time(df, name, i):
    plt.figure(figsize=[10, 5])
    sns.distplot(np.log(df[df['reply'] & (df['sender_name'] == 'Victor Jann')]['response_seconds']).fillna(1.0), label='Victor')
    sns.distplot(np.log(df[df['reply'] & (df['sender_name'] != 'Victor Jann')]['response_seconds']).fillna(1.0), label=name)
    plt.title('Response Time with {}'.format(name))
    plt.xlabel("Response Time in log(seconds)")
    plt.legend()
#     plt.show()
    plt.savefig('../plots/response_time_distribution_' + name)
[plot_response_time(individual_df[i], individuals_name[i], i) for i in range(2)]
```




    [None, None]




![png](FBMessagesPublic_files/FBMessagesPublic_32_1.png)



![png](FBMessagesPublic_files/FBMessagesPublic_32_2.png)


# Analyzing Messages with an Individual


```python
def plot_simple_groupings(df, title):
    plt.figure()
    sns.barplot(data=df, x=df.index, y='num_words')
    plt.title("{}".format(title))
    plt.ylabel("Count".format(title))
    plt.xlabel("Person")
#     plt.show()
    plt.savefig('../plots/simple_groupings_' + title)
messages = someone.groupby('sender_name').count()

message_len = someone.groupby('sender_name').mean()

baby_face_emoji = someone[someone['content'].str.contains("\u00f0\u009f\u00a5\u00ba")].groupby('sender_name').count() #I believe it's unicode for the 'pleading face' emoji

stickers = someone[someone['sticker'].notna()].groupby('sender_name').count()

spotify_links = someone[(someone['content'].str.contains("https://open.spotify.com/track")) | 
                       someone['share'].fillna({'link': 'dummy'}).apply(lambda x: not isinstance(x, float) and "https://open.spotify.com/track" in x['link'])]
spotify_links = spotify_links.groupby('sender_name').count()

uwus = someone[someone['content'].str.findall(r'\buwu\b').apply(lambda x: len(x) != 0)].groupby('sender_name').count()

plot_simple_groupings(messages, 'Messages Sent')
plot_simple_groupings(message_len, 'Average Message Length (Words)')
# ... keep going (or turn it into a loop)
plot_simple_groupings(spotify_links, "Spotify Links Sent")

```


![png](FBMessagesPublic_files/FBMessagesPublic_34_0.png)



![png](FBMessagesPublic_files/FBMessagesPublic_34_1.png)



![png](FBMessagesPublic_files/FBMessagesPublic_34_2.png)


### Word Cloud!
I found it too time-consuming to rid of the filler words, but other than that here it is!


```python
#used this website https://www.geeksforgeeks.org/find-k-frequent-words-data-set-python/?ref=rp
from PIL import Image as PILImage
from collections import Counter 
from wordcloud import WordCloud, STOPWORDS

#optional add more words to ignore
more_words = []
for w in more_words:
    STOPWORDS.add(w)

np.savetxt(r'../messages/pure_message_content_vj.txt', someone[someone['sender_name'] == 'Victor Jann']['content'].values, fmt="%s")

dataset_vj = open("../messages/pure_message_content_vj.txt", "r").read().lower()

maskArray = np.array(PILImage.open("../bear.jpg"))
cloud = WordCloud(background_color = "white", max_words = 200, mask = maskArray, stopwords = set(STOPWORDS))
cloud.generate(dataset_vj)
cloud.to_file("./outputs/wordCloud_vj.png")

```




    <wordcloud.wordcloud.WordCloud at 0x7f752bd31d30>




```python
from IPython.display import Image as ipythonImage
ipythonImage(filename='outputs/wordCloud_vj.png') 
```




![png](FBMessagesPublic_files/FBMessagesPublic_37_0.png)



Lastly, here's some useful tools for finding chats at specific times or words.


```python
#this is useful. find all messages in convo within "within" minutes of 'reference_date'. 
#You can add hours or days
reference_date = datetime.datetime(year=2020, month=3, day=20, hour=1, minute=24)
someone[abs(someone['time_stamp'] - reference_date) < datetime.timedelta(hours=0, minutes=0, seconds=30)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sender_name</th>
      <th>content</th>
      <th>message_sentiment</th>
      <th>num_words</th>
      <th>time_stamp</th>
      <th>hour</th>
      <th>reply</th>
      <th>response_seconds</th>
      <th>response_days</th>
      <th>reactions</th>
      <th>sticker</th>
      <th>photos</th>
      <th>videos</th>
      <th>files</th>
      <th>gifs</th>
      <th>share</th>
      <th>type</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6243</th>
      <td>Murrah</td>
      <td>it is but i am lagging behind</td>
      <td>-0.306</td>
      <td>7</td>
      <td>2020-03-20 01:24:26.123</td>
      <td>1.400000</td>
      <td>True</td>
      <td>27.764</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Generic</td>
      <td>2020-03-20</td>
    </tr>
    <tr>
      <th>6244</th>
      <td>Victor Jann</td>
      <td>its a two person thing isnt it</td>
      <td>0.000</td>
      <td>7</td>
      <td>2020-03-20 01:23:58.359</td>
      <td>1.383333</td>
      <td>False</td>
      <td>6.593</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Generic</td>
      <td>2020-03-20</td>
    </tr>
    <tr>
      <th>6245</th>
      <td>Victor Jann</td>
      <td>LOL stop</td>
      <td>0.232</td>
      <td>2</td>
      <td>2020-03-20 01:23:51.766</td>
      <td>1.383333</td>
      <td>True</td>
      <td>20.413</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Generic</td>
      <td>2020-03-20</td>
    </tr>
    <tr>
      <th>6246</th>
      <td>Murrah</td>
      <td>im sorry</td>
      <td>-0.565</td>
      <td>2</td>
      <td>2020-03-20 01:23:31.353</td>
      <td>1.383333</td>
      <td>False</td>
      <td>2.660</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Generic</td>
      <td>2020-03-20</td>
    </tr>
  </tbody>
</table>
</div>




```python
#use regex to find all occurence of string.
someone[someone['content'].str.findall(r'go bears').apply(lambda x: len(x) != 0)].sort_values('time_stamp',ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sender_name</th>
      <th>content</th>
      <th>message_sentiment</th>
      <th>num_words</th>
      <th>time_stamp</th>
      <th>hour</th>
      <th>reply</th>
      <th>response_seconds</th>
      <th>response_days</th>
      <th>reactions</th>
      <th>sticker</th>
      <th>photos</th>
      <th>videos</th>
      <th>files</th>
      <th>gifs</th>
      <th>share</th>
      <th>type</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>694</th>
      <td>Victor Jann</td>
      <td>go bears</td>
      <td>0.0</td>
      <td>2</td>
      <td>2020-03-27 14:04:57.872</td>
      <td>14.066667</td>
      <td>False</td>
      <td>2.333</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Generic</td>
      <td>2020-03-27</td>
    </tr>
    <tr>
      <th>5617</th>
      <td>Victor Jann</td>
      <td>go bears</td>
      <td>0.0</td>
      <td>2</td>
      <td>2020-02-24 00:31:16.259</td>
      <td>0.516667</td>
      <td>False</td>
      <td>5.779</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Generic</td>
      <td>2020-02-24</td>
    </tr>
  </tbody>
</table>
</div>



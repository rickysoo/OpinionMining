# Databricks notebook source
# MAGIC %md
# MAGIC # Opinion Mining with Big Data

# COMMAND ----------

# MAGIC %md
# MAGIC Enter a topic. See what people say about it on Twitter, in real time.
# MAGIC 1. Set up environment
# MAGIC 2. Stream from Twitter
# MAGIC 3. Visualize tweets

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Set Up Environment

# COMMAND ----------

# Ask for topic in Databricks
dbutils.widgets.text('topic', 'Omicron', 'What people are talking about:')
topic = dbutils.widgets.get('topic')

# Ask for topic in other environments
# topic = input('Enter a topic: ')

# Or run without asking for input
# topic = 'Omicron'

topic = topic.lower().strip()
print('The topic is ' + topic)

# COMMAND ----------

# Turn off widget in Databricks if not in used
# dbutils.widgets.removeAll()

# COMMAND ----------

# Install modules
!pip install tweepy -q --disable-pip-version-check
!pip install nltk -q --disable-pip-version-check
!pip install wordcloud -q --disable-pip-version-check

# COMMAND ----------

# Install Spark NLP if Jupyter Notebook is used
# pip install spark-nlp==3.4.0

# Install Spark NLP if Google Colab is used
# !wget http://setup.johnsnowlabs.com/colab.sh -O - | bash

# Install Spark NLP as per https://nlp.johnsnowlabs.com/docs/en/install if Databricks is used

# COMMAND ----------

# Import modules
import json
import tweepy
import time
from datetime import datetime
import matplotlib.pyplot as plt

from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml import Pipeline

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline
from sparknlp import Finisher

from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# COMMAND ----------

# Set up NLTK module for natural language processing
nltk.download('stopwords')
nltk.download('punkt')

# COMMAND ----------

# Start Spark NLP and show link to Spark UI
spark = sparknlp.start()
spark

# COMMAND ----------

# Load pipeline for sentiment analysis
# https://nlp.johnsnowlabs.com/2021/01/18/analyze_sentimentdl_use_twitter_en.html
print('Loading pipeline for sentiment analysis...')

# Load pipeline from memory if available
if 'pipeline_sentiment' in globals():
    print('Pipeline for sentiment analysis loaded from memory')
else:
    # Or download pipeline from Spark NLP server
    pipeline_sentiment = PretrainedPipeline("analyze_sentimentdl_use_twitter", lang = "en") 
        
print(pipeline_sentiment)

# COMMAND ----------

# Load pipeline for emotion analysis
# https://nlp.johnsnowlabs.com/2021/11/21/distilbert_sequence_classifier_emotion_en.html
print('Loading pipeline for emotion analysis...')

# Try to load pipeline from memory
if 'pipeline_emotion' in globals():
    print('Pipeline for emotion analysis loaded from memory')
else:
    # Download pipeline from Spark NLP server
    document_assembler = DocumentAssembler() \
        .setInputCol('text') \
        .setOutputCol('document')

    tokenizer = Tokenizer() \
        .setInputCols(['document']) \
        .setOutputCol('token')

    sequenceClassifier = DistilBertForSequenceClassification \
          .pretrained('distilbert_sequence_classifier_emotion', 'en') \
          .setInputCols(['token', 'document']) \
          .setOutputCol('class') \
          .setMaxSentenceLength(512)

    pipeline_emotion = Pipeline(stages=[
        document_assembler, 
        tokenizer,
        sequenceClassifier    
    ])
    
    print('Pipeline for emotion analysis downloaded from server')

print(pipeline_emotion)

# COMMAND ----------

# Twitter developer account credentials (use your own)
consumer_key = 'mWTEf1dCKQbyc1trrd0lzvsrn'
consumer_secret = 'NJEbzBemHcoC1crn7Mioo6aut70VsvP8vhdeeRMZmSUYjvZQXX'
access_token = '9618812-MVKlOl4WDy1Tr4cGhIjGwp60wrH8hI4cf3KRUVmaDw'
access_token_secret = '77AImSbrh9PyorLlPM2xvZH0aJfcy53bfwyrz99cxfjhY'

# COMMAND ----------

# Authenticate credentials with Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# COMMAND ----------

# Location used to store tweets on DBFS
path = '/FileStore/tweets/' + datetime.now().strftime('%Y%m%d%H%M%S') + '/'
dbutils.fs.mkdirs(path)
print('Location to save tweets: ' + path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Stream From Twitter

# COMMAND ----------

# A function to convert date in Twitter format to Spark timestamp format
def ConvertDate(tweet_date):
    return datetime.strftime(datetime.strptime(tweet_date,'%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d %H:%M:%S')

# COMMAND ----------

# A function to get the sentiment of a text using pre-trained pipeline
def GetSentiment(text):
    data = spark.createDataFrame([[text]]).toDF('text')
    result = pipeline_sentiment.transform(data)
    sentiment = result.select('sentiment.result').select('result').collect()[0][0][0].title()
    return sentiment

# COMMAND ----------

# A function to get the sentiment of a text using pre-trained pipeline
def GetEmotion(text):
    data = spark.createDataFrame([[text]]).toDF('text')
    result = pipeline_emotion.fit(data).transform(data)
    emotion = result.select('class.result').collect()[0][0][0].title()
    return emotion

# COMMAND ----------

# A function to retrieve relevant parts of a tweet and save the tweets in JSON format
def SaveTweet(tweet):
    # Process only tweets that are not empty and not retweeted
    if tweet and ('retweeted_status' not in tweet) and topic in tweet['text'].lower():
        out = {}
        out['ID'] = tweet['id']
        out['Created'] = ConvertDate(tweet['created_at'])
        out['User'] = tweet['user']['screen_name']
        out['Tweet'] = tweet['text']
        out['Sentiment'] = GetSentiment(tweet['text'])
        out['Emotion'] = GetEmotion(tweet['text'])
        
        filename = str(tweet['id']) + '.json'
        with open('/dbfs' + path + filename, 'w') as outfile:
            json.dump(out, outfile)

# COMMAND ----------

# A class to specify what to do upon certain events
class TweetStream(tweepy.Stream):        
    def on_connection_error(self):
        print('Disconnecting...')
        self.disconnect()
    
    def on_data(self, data):
        current_tweet = json.loads(data)
        SaveTweet(current_tweet)
        
    def on_status(self, status):
        pass

# COMMAND ----------

# Start streaming from Twitter
tweets = TweetStream(
  consumer_key, consumer_secret,
  access_token, access_token_secret
)

# COMMAND ----------

# Real time tweets are retrieved based on selected topic. Get English tweets only. Run in separate thread.
thread_tweets = tweets.filter(track=[topic], languages=['en'], threaded=True)

# COMMAND ----------

# Confirm the stream is running
tweets.running

# COMMAND ----------

# Define the JSON schema
jsonSchema = StructType([
    StructField('ID', StringType(), True), # Tweet ID
    StructField('Created', TimestampType(), True), # Tweet creation time
    StructField('User', StringType(), True), # Twitter user screen name
    StructField('Tweet', StringType(), True), # Tweet text
    StructField('Sentiment', StringType(), True), # Sentiment of tweet
    StructField('Emotion', StringType(), True) # Emotion of tweet
])

# COMMAND ----------

# Stream tweets from DBFS directory
streamingInputDF = (
  spark
    .readStream                       
    .schema(jsonSchema)
    .option("maxFilesPerTrigger", 1)
    .json(path)
)

# Confirm the stream is running
streamingInputDF.isStreaming

# COMMAND ----------

streamingInputDF.createOrReplaceTempView("QueryTweets")

# COMMAND ----------

# Reduce the size of shuffles from the default 200
spark.conf.set('spark.sql.shuffle.partitions', '2')

# COMMAND ----------

# Confirm tweets are stored in directory
dbutils.fs.ls(path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Visualize Tweets

# COMMAND ----------

# MAGIC %md
# MAGIC Table - Show the tweets.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM QueryTweets
# MAGIC 
# MAGIC -- Equivalent code in PySpark
# MAGIC -- df_tweets = spark.sql('SELECT * FROM QueryTweets')
# MAGIC -- df_tweets.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Visualization #1 - Show the sentiments of the tweets in bar chart.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT Sentiment, count(*) AS Count 
# MAGIC FROM QueryTweets 
# MAGIC GROUP BY Sentiment 
# MAGIC ORDER BY 
# MAGIC   CASE 
# MAGIC     WHEN Sentiment = "Negative" THEN 1 
# MAGIC     WHEN Sentiment = "Neutral" THEN 2 
# MAGIC     WHEN Sentiment = "Positive" THEN 3 
# MAGIC   END ASC
# MAGIC 
# MAGIC -- Equivalent code in PySpark
# MAGIC -- df_sentiment = spark.sql('SELECT Sentiment, count(*) AS Count FROM QueryTweets GROUP BY Sentiment ORDER BY CASE WHEN Sentiment = "negative" THEN 1 WHEN Sentiment = "neutral" THEN 2 WHEN Sentiment = "positive" THEN 3 END ASC')
# MAGIC -- df_sentiment.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Visualization #2 - Show the sentiments of the tweets in pie chart.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT Sentiment, count(*) AS Count 
# MAGIC FROM QueryTweets 
# MAGIC GROUP BY Sentiment 
# MAGIC ORDER BY 
# MAGIC   CASE 
# MAGIC     WHEN Sentiment = "Negative" THEN 1 
# MAGIC     WHEN Sentiment = "Neutral" THEN 2 
# MAGIC     WHEN Sentiment = "Positive" THEN 3 
# MAGIC   END ASC

# COMMAND ----------

# MAGIC %md
# MAGIC Visualization #3 - Show the emotions of the tweets in bar chart.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT Emotion, count(*) AS Count 
# MAGIC FROM QueryTweets 
# MAGIC GROUP BY Emotion 
# MAGIC ORDER BY Emotion
# MAGIC 
# MAGIC -- Equivalent code in PySpark
# MAGIC -- df_emotion = spark.sql('SELECT Emotion, count(*) AS Count FROM QueryTweets GROUP BY Emotion ORDER BY Emotion')
# MAGIC -- df_emotion.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Visualization #4 - Show the emotions of the tweets in pie chart.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT Emotion, count(*) AS Count 
# MAGIC FROM QueryTweets 
# MAGIC GROUP BY Emotion 
# MAGIC ORDER BY Emotion

# COMMAND ----------

# MAGIC %md
# MAGIC Visualization #5 - Show a word cloud of the tweets.

# COMMAND ----------

# Wait for tweets to come in
time.sleep(15)

# COMMAND ----------

# Create list of stop words
stopwords_en = stopwords.words('english')
stopwords_en.append('https')
stopwords_en.append('n\'t')

# COMMAND ----------

streamingInputDF.writeStream.format('memory').queryName('QueryCloud').outputMode('append').start()

# COMMAND ----------

# Prepare content for word cloud
df_cloud = spark.sql('SELECT Tweet FROM QueryCloud')
rows = df_cloud.select('Tweet').rdd.flatMap(lambda x: x).collect()
words = word_tokenize(' '.join(rows).lower())
words = [word for word in words if word not in stopwords_en and len(word) > 1 and not word.startswith('//t.co')]
content = ' '.join(words)

if content:
    plt.figure(figsize=(16, 8))
    wordcloud = WordCloud(max_font_size=50, max_words=200, background_color='black').generate(content)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
else:
    print('No word to show yet')

# COMMAND ----------

# Wait 30 seconds then stop all streams to save data transfer
# time.sleep(30)

# tweets.disconnect()

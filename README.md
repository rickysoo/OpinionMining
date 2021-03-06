# Opinion Mining
Do you want to know what people are talking about a certain topic at the moment?  
Just enter a keyword, and the tweets and analysis will show up instantly!

## The Backend - Databricks Notebook

This works originally in the Databricks environment - https://databricks.com  
(It might not work in Databricks Community Edition)

1. Log into your Databricks account at https://databricks.com
2. Install Spark NLP in your Databricks cluster as per https://nlp.johnsnowlabs.com/docs/en/install#databricks-support
3. Import the [OpinionMining.dbc file](https://github.com/rickysoo/OpinionMining/raw/main/OpinionMining.dbc) into your Databricks workspace.
4. Have fun!

![Databricks notebook](https://github.com/rickysoo/OpinionMining/raw/main/OpinionMining3.png)

To run in other environments such as Google Colab, comment/uncomment certain fragments indicated in the code.

## The Frontend - Opinion Mining Dashboard

The dashboard works under the Databricks environment.

Watch the real-time tweets show up together with visualizations.

![Dashboard (page 1)](https://github.com/rickysoo/OpinionMining/raw/main/OpinionMining1.png)

Watch the sentiments and emotions change as people talk about the topic!

![Dashboard (page 2)](https://github.com/rickysoo/OpinionMining/raw/main/OpinionMining2.png)

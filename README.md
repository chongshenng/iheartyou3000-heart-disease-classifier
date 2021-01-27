# Heart Disease Classifier
A Naïve-Bayes classifier for heart disease based on the [University of California Irvine dataset](https://www.kaggle.com/ronitf/heart-disease-uci). Authored by [John Ng](www.linkedin.com/in/chongshenng). 

## Purpose
The two aims of this project are:
1. Create an app to predict likelihood of heart disease.
2. Demonstrate the potency of a _statistical_ machine learning approach in a simple classifying problem.

## The Outcome
The final data app is built using [`streamlit`](https://github.com/streamlit/streamlit), which is a fast and easy way to build interactive apps. From a DS viw, fast and easy is definitely appealing! 

<img src="https://github.com/chongshenng/blob-and-splash/blob/master/docs/iheartyou_app.gif" alt="heart app" title="heart app" width="600"/>


## Step-by-step
The data science tasks will follow the [5 steps of the Data Science Process](https://towardsdatascience.com/the-data-science-process-a19eb7ebc41b):

1. [Collection](#1_Collection)
2. [Cleaning](#2_Cleaning)
3. [Exploratory Data Analysis](#3_EDA)
4. [Model Building](#4_ModelBuilding)
5. [Model Deployment](#5_ModelDeploy)

### Collection 
<a id='1_Collection'></a>

In the Collection step, the data is imported using the `pandas` package. Since the source is in a `*.csv` format, this import step has been simplified significantly! In reality and in real-use settings, the data format will be more complex and can involve unstructured/structured SQL-type formats. 

### Cleaning
<a id='2_Cleaning'></a>

Here, before further analysis, we need to evaluate the health of the data and clean them where appropriate. What we are looking for are for example: missing values, `NaNs`, zeros or conflicting data formats in the data columns. To help with this, we will use the `pandas.dataframe.describe` API, which conveniently generates descriptive statistics of our dataset. Note that by default, the `percentiles` range are set to 25% and 75% percentiles `[0.25,0.75]`. To get a better sense of the outliers, here, we have used the 5% and 95% percentile: `[0.05,0.95]`. 

From the `df.describe` output, out dataset looks quite clean! For example, all values are present (`count`=303) and there are no `NaN`-values. It turns out that our dataset, which was obtained [via Kaggle](https://www.kaggle.com/ronitf/heart-disease-uci), has been cleaned for us. Quite convenient!

### Exploratory Data Analysis (EDA)
<a id='3_EDA'></a>

Now, we explore the data more. One efficient way of doing to is by using the `pandas-profiling` package (installed using `pip install pandas-profiling`). With the cleaned dataframe, we can simply obtain the profile report calling `ProfileReport(df, **kwargs)`. The report will be generated in a tidy embedded html output, as follows:

<img src="https://github.com/chongshenng/blob-and-splash/blob/master/docs/iheartyou_pandas_profile.gif" alt="pandas profiling gif" title="pandas profiling gif" width="600"/>

### Model Building
<a id='4_ModelBuilding'></a>

Here, we build the model with these steps in mind:
* Select the Bernoulli Naïve-Bayes classifier. We can import this directly from `scikit`. Note that other classifiers may also end up performing better than our selection (such as XGBoost, lightgbm). NB is a good place to start, plus learning from probabilities is also very informative!
* Assess the model. This step uses the confusion matrix and the metrics derived from it, that is: precision, recall, and the the $f_1$ score. `sklearn` has a useful `classification_report` tool that neatly generates this for us.
* Further testing and fine-tuning of the model using cross-validation. Here, we employ the ROC and AUC metrics. Note that we can also use the `gridsearchcv` tool to have better control of the fine-tuning process.

### Model Deployment
<a id='5_ModelDeploy'></a>
In this final step, the model parameters are stored in a `pickle` file. This file is then loaded when we excute a simple python script which uses `streamlit`, the results which you've seen at the top of this readme! Nice!

## References
* [Python Machine Learning by Example (3rd Edition), by Yuxi (Hayden) Liu](https://www.packtpub.com/product/python-machine-learning-by-example-third-edition/9781800209718)
* [TDS article by Chanin Nantasenamat (a.k.a. the Data Professor's)](https://towardsdatascience.com/the-data-science-process-a19eb7ebc41b)
* [UCI heart disease dataset from Kaggle](https://www.kaggle.com/ronitf/heart-disease-uci)

## Licenses
License

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png)](http://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [John Ng](www.chongshenng.github.io) has waived all copyright and related or neighboring rights to this work.

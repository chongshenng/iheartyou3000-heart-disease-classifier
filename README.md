# Heart Disease Classifier
A Naïve-Bayes classifier for heart disease based on the [University of California Irvine dataset](https://www.kaggle.com/ronitf/heart-disease-uci). Authored by John Ng. 

## Purpose
The aim of this project is to demonstrate the potency of a _statistical_ machine learning approach to classify and predict the likelihood that a patient exhibits heart diseases. 

## Step-by-step
The tasks will follow the [5 steps of the Data Science Process](https://towardsdatascience.com/the-data-science-process-a19eb7ebc41b):

1. [Collection](#1_Collection)
2. [Cleaning](#2_Cleaning)
3. [Exploratory Data Analysis](#3_EDA)
4. [Model Building](#4_ModelBuilding)
  * Naïve-Bayes - A quick recap
  * The academic approach
  * The applied approach using `scikit`
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

<img src="https://github.com/chongshenng/blob-and-splash/blob/master/docs/iheartyou_01.gif" alt="pandas profiling gif" title="pandas profiling gif" width="500"/>
<!-- ![](https://github.com/Voyz/voyz_public/blob/master/databay_promo_vidA_gif_A03.gif) -->

### Model Building
<a id='4_ModelBuilding'></a>

* Applying `scikit`
* Model assessment.
* ROC and AUC.

### Model Deployment
<a id='5_ModelDeploy'></a>

## References
* [Python Machine Learning by Example (3rd Edition), by Yuxi (Hayden) Liu](https://www.packtpub.com/product/python-machine-learning-by-example-third-edition/9781800209718)
* [TDS article by Chanin Nantasenamat (a.k.a. the Data Professor's)](https://towardsdatascience.com/the-data-science-process-a19eb7ebc41b)
* [UCI heart disease dataset from Kaggle](https://www.kaggle.com/ronitf/heart-disease-uci)

## Licenses
License

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png)](http://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [John Ng](www.chongshenng.github.io) has waived all copyright and related or neighboring rights to this work.

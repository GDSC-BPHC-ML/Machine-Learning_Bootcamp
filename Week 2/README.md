# Week 2

Welcome to Week 2! Having laid the foundations of Machine Learning, this week we start exploring algorithms and concepts of Machine Learning. We discuss about the fudamentals of Machine Learning - 

* **[Linear Regression](#linear-regression)**
* **[Polynomial Regression](#polynomial-regression)**
* **[Segmented Regression](#segmented-regression)**
* **[Locally Weighted Regression](#locally-weighted-regression)**
* **[Exploratory Data Analysis](#exploratory-data-analysis)**
* **[Data Pre-Processing](#data-pre-processing)**
* **[Generalised Linear Models](#generalised-linear-models)** (Advanced optional content)

First we shall go into Machine Learning models for estimation of values. This is termed as regression. A good example would be using regression to obtain the least square fit line equation which we all have used in PH117 to obtain a good estimate of a linear model of the data points given. There are other models as well however linear regression is the simplest.  
## Recap of Basic Linear Algebra, Probability and Statistics

The crux of machine learning and any subsets of it lie within the mathematics involved in them. To get a good grasp of how a machine learning algorithm truly works, we must first ensure that our basics in linear algebra, probability theory and statistics are firm (Thankfully, these topics are covered in the college curriculum). However, in case you do need to brush up here is a list of the most important topics and some resources to revise them.
### Linear Algebra topics to revise:
* Scalar, Vectors, Matrices and Tensors
* Dimensions and Transpose of a Matrix
* The various products that can be performed on Matrices (Dot Product, Matrix Multiplication, Hadarmard Product...)
* Norms of vectors and matrices
* Derivatives of a Matrix

### Probability Topics to revise:
* Basics of Probability
* Bayes Theorem
* Types of Probability distributions
#### Optionally, knowing concepts of statistics can be useful in understanding data cleaning and some correction terms used in Machine Learning and Deep Learning Algorithms.

#### Here are a few useful links:

* **[Linear Algebra Refresher - Stanford](https://stanford.edu/~shervine/teaching/cs-229/refresher-algebra-calculus)**
* **[Linear Algebra Refresher - CERN](https://cas.web.cern.ch/sites/default/files/lectures/thessaloniki-2018/linalg-lect-1.pdf)**
* **[Linear Algebra - CMU](http://mlsp.cs.cmu.edu/courses/fall2019/lectures/lecture3_linear_algebra_part2.pdf)**
* **[Probability and Statistics - NYU](https://cims.nyu.edu/~cfgranda/pages/stuff/probability_stats_for_DS.pdf)**
* **[Probability and Statistics Refresher - Baylor](http://cs.baylor.edu/~hamerly/courses/5325_17s/resources/haas_prob_stats_refresher.pdf)**
* **[Probability Refresher - Stanford](https://stanford.edu/~shervine/teaching/cs-229/refresher-probabilities-statistics)**

## Linear Regression

Linear Regression is one of the most fundamental models in Machine Learning. It assumes a linear relationship between target variable *(y)* and predictors (input variables) *(x)*. Formally stating, *(y)* can be estimated assuming a linear combination of the input variables *(x)*. 

When we have a single predictor as the input, the model is called as **Simple Linear Regression** and when we have multiple predictors, the model is called **Multiple Linear Regression**.  
The input variable *(x)* is a vector of the features of our dataset, for eg. when we are predicting housing prices from a dataset, the features of the dataset will be Area of the house, Locality, etc. and the output variable *(y)* in this case will be the housing price. The general representation of a Simple Linear Regression Model is -
                                                
                                                y = β(0) + β(1)*x

where *β(0)* is known as the **bias term**, and *β* in general is called as the **weight vector**, there are the parameters of the linear regression model. This **[video](https://www.coursera.org/learn/machine-learning/lecture/db3jS/model-representation)** gives a detailed explanation of how we arrived at this representation. For multiple linear regression, the equation modifies to - 

                                                y = transpose(β)*(X) 

where *X* is a vector containing all the features and *β* is a vector containing all the corresponding weights (bias and weights both) i.e. the parameters of the linear model. Also note that *X* here has an additonal feature - the constant term 1 which accounts for the intercept of the linear fit.

We define a function called the **Cost Function** that accounts for the prediction errors of the model. We try to minimize the cost function so that we can obtain a model that fits the data as good as possible. To reach the optima of the cost function, we employ a method that is used in almost all of machine learning called **Gradient Descent**.  

---

#### Note

The relationship established between the target and the predictors is a statistical relation and not determinsitic. A deterministic relation is possible only when the data is actually prefectly linear.

---

#### Useful Resources

* **[Overview of Gradient Descent](https://medium.com/@saishruthi.tn/math-behind-gradient-descent-4d66eb96d68d)**

* This **[video](https://www.coursera.org/learn/machine-learning/lecture/Z9DKX/gradient-descent-for-multiple-variables)** here explains an optimisation technique for Machine learning models, (Linear Regression in our case), known as **Gradient Descent** to reach at the minima of the **[Cost Function](https://www.coursera.org/lecture/machine-learning/cost-function-rkTp3)** so as to arrive at the optimum parameters/weights, i.e., find a value of *Θ* (the weight vector) such that the prediction of our model is the most accurate. And this is what we mean by **Model Training**, provide training data to the model, then the model is optimized according to the training data, and the predictions become more accurate. This process is repeated every time we provide new training data to the model.  
 
* (Optional) This **[article's](http://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/03/lecture-03.pdf)** sections 2 and 3 explain the concept of Linear Regression as a Statistical Model, where you can calculate certain statistical quantities to determine and improve the accuracy of your model. This is known as **Optimisation**.


## Polynomial Regression

In Linear Regression, we only try to classify the data based on the assumption that the relationship between the variables is linear. As such we draw a line of best fit and try to minimize the distance of each point from that line. However, in reality this is not always the case. The relationship between variables is often more complex than a simple linear relationship and there is a need to introduce some element of non-linearity into the models to help them achieve better results for such data.

This is where Polynomial Regression comes into play. We introduce non-linear terms into the mix as follows:

                                    y = β(0) + β(1)*x + β(2)*x^2 + β(3)*x^3 + ...

Now if we had multiple input features the number of terms would grow similar based on their interaction with each other. In this case we take a case where we have 2 input features and they both have an effect on each other:

![Multiple Polynomial Regression](https://slideplayer.com/slide/16553073/96/images/11/Models+with+Interaction+and+Quadratic+Predictors.jpg)

## Exploratory Data Analysis

Last week, we learned about Data Visualisation and Exploration. To get hands on experience on Data Analysis on Regression and Classification, refer to the links below - 
* **[Regression](https://github.com/MicrosoftLearning/Principles-of-Machine-Learning-Python/blob/master/Module2/VisualizingDataForRegression.ipynb)**
* **[Classification](https://github.com/MicrosoftLearning/Principles-of-Machine-Learning-Python/blob/master/Module2/VisualizingDataForClassification.ipynb)**

Now finally let us look into one important aspect of data analysis that is important for machine learning, data cleaning.

## Data Pre-Processing

Let us consider a simple classification problem using logistic regression. Suppose you have 10 columns in your data which would make up the raw features given to you. A naive model would involve training your classifier using all these columns as features. However are all the features equally relevant? This may not be the case. As a worst case example suppose all the data entries in your training set have the same value. Then it does not make sense to consider this as a feature since any tweaking to the parameter corresponding to this feature that you do can be done by changing the bias term as well. This is a redundant input feature that you should remove. Similarly if you have a column that has very low variance it may make sense to remove this feature from your dataset as well. When we work with high dimensional data sometimes it makes sense to work with fewer dimensions. Thus it makes sense to remove the lower variance dimensions. Note that sometimes this reduction of dimension may not be as straightforward and next week we will see how to do this using PCA.

* This **[article](https://machinelearningmastery.com/basic-data-cleaning-for-machine-learning/)** provides a proper introduction to data cleaning for machine learning
* This **[article](https://www.dataquest.io/blog/machine-learning-preparing-data/)** is also useful for data cleaning.

Another way we can clean and improve our data is by performing appropriate transformations on the data. Consider the task of Sentiment Classification using Logistic Regression. You are given a tweet and you have to state whether it expresses happy or sad sentiment. You could just take the tweet in and feed it into the classifier (using a particular representation, the details aren't important). But do all the words really matter? 

Consider a sample tweet
```
#FollowFriday @France_Inte @PKuchly57 @Milipol_Paris for being top engaged members in my community this week :)
```
Clearly any tags in this tweet are irrelevant. Similarly symbols like '#' are also not needed. Thus we need to clean the input data to remove all this unnecesary information. Further in Natural Language Processing words like 'for', 'in' do not contribute to the sentiment and a proper classification would require us to remove this as well. All of this comes under data cleaning and preprocessing.

The preprocessed version of the above tweet would be:
```
['followfriday', 'top', 'engag', 'member', 'commun', 'week', ':)']
```

* This **[article](https://towardsdatascience.com/data-preprocessing-concepts-fa946d11c825)** explains data preprocessing.
* This **[tutorial](https://towardsdatascience.com/twitter-sentiment-analysis-classification-using-nltk-python-fa912578614c)** uses Naive Bayes to solve the above tweet classification problem. Please go through this to acquaint yourself with the various data processing methods needed for natural language processing (Note that this tutorial uses sklearn to implement naive bayes instead of doing it from scratch. Feel free to skip the implementational details if you are facing issues with it as we will not be covering sklearn in this course)

The concepts of feature aggregration, data cleaning, dimensionality reduction are important for a data scientist and it is essential to have a proper understanding of them before continuing.  


## How to choose the right model? Can a model be too good?
Now that we have learnt all about exploratory analysis and data pre-processing, we can get a better idea of how the input features are related to each other using these. We can also test out multiple models and see how they perform for our *testing data*.

The most important part is understanding that our model could be perfect on our training data but still perform poorly on testing data. Let us take an example where the data has a complex non-linear relationship. Obviously, the more non-linear the relationship, the more it makes sense for us to use a higher degree model. Right? Unfortunately, this does not always result in the desired outcome. 

Why does this happen?

This is due to a phenomenon called overfitting where the model learns the training data too well and accounts even for outliars that may otherwise have been ignored by less complex models. In doing so, the general trend is given up for exactly matching the pattern of the data it is trained on which may lead to poor testing performance and lower accuracy as compared to a lower degree model
![Overfitting](https://builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/national/underfitting%2520and%2520overfitting%2520example.png)


---
## Here are some topics for additional reading

## Segmented Regression

It is also referred to as **Piecewise Regression** or **Broken Stick Regression**. Unlike standard Linear Regression, we fit separate line segments to intervals of independent variables. Formally stating, independent linear models are assigned to individual partitions of feature spaces. This is pretty useful in the cases when the data assumes different linear relationships in separate intervals of the feature values. Generally, applying linear regression on such tasks can incur large statistical losses.  

It is important to note that despite it's better accuracy, it can be computationally inefficient because many independent linear regressions have to be performed.     

The points where the data is split into intervals are called **Break Points**. There are various statistical tests that are to be performed in order to determine the break points but the details of those tests are beyond the scope of this course.  

#### Useful Resources

* To get an overview of the concepts, go through **[this](https://storybydata.com/datacated-challenge/piecewise-linear-regression/)** article and **[this](https://en.wikipedia.org/wiki/Segmented_regression)** page.  

* Refer to the **[repo](https://github.com/srivatsan88/piecewise-regression/blob/master/piecewise_linear_regression.ipynb)** for a small demo on Piecewise Linear Regression.  

* (Optional) To delve deeper into the mathematical aspects and get hands on experience, go through **[this](https://www.fs.fed.us/rm/pubs/rmrs_gtr189.pdf)**. This is pretty complicated to understand and it is upto the interest of the reader.  

## Locally Weighted Regression

In Locally Weighted Regression, we try to fit a line locally to the point of prediction. What it means is that we give more emphasis to the training data points that are close to the point where prediction is to be made. A weight function (generally a bell shaped function) is used to determine the amount of emphasis given to a data point for prediction. This kind of model comes under **Non Parametric Learning** unlike the **Parametric Learning** models that we have seen so far. 

Such models are useful when the training data is small and has small number of features. It is computationally inefficient because a linear regression is performed everytime we need to predict some value. But at the same time, this allows us to fit even complicated non-linear curves with ease.  

#### Useful Resources

* Follow this article to get an [overview](https://medium.com/100-days-of-algorithms/day-97-locally-weighted-regression-c9cfaff087fb) of this model.

For classification tasks we need a different architecture. In the next section, we discuss about Logistic Regression - a regression tool employed for classification. 
 

## Generalised Linear Models

On a very high level, Generalised Linear Models (GLM) are a superclass of all linear models including Linear and Logistic Regression that we talked about earlier. 

* Linear Regression is based on the assumption that the data assumes **Gaussian/Normal Distribution**.
* Binary Logistic Regression is based on the assumption that data assumes **Bernoulli Distribution**.

GLM's are based on Exponential Family Functions. **[Exponential Family](https://en.wikipedia.org/wiki/Exponential_family)** includes these two and a lot of other distributions of data that we talked about in week 1. So therefore GLM's serve as a generalisation of all linear models. To get a firm hold on generalised linear models, head **[here](https://towardsdatascience.com/generalized-linear-models-9cbf848bb8ab)**.

---

Next week we will be looking at more supervised machine learning techniques. Till then stay safe and have fun.

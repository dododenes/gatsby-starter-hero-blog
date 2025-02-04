---
title: Machine Learning by Andrew Ng - Week 1 (Summary)
category: "machine learning" 
cover: large-icon.png
author: Dorottya Denes
---

## 1. What is Machine Learning?

 There isn't a universally accepted definition of machine learning by its practitioners, hence multiple definitions are provided.

In the 1950s, [Arthur Samuel](https://en.wikipedia.org/wiki/Arthur_Samuel) built a checkers program that learned by playing against 
itself for thousands of times. He coined the term 'machine learning' as 

>the field of study that gives computers the ability to learn without being explicitly programmed.

Later this definition became outdated and more formality was introduced by [Tom Mitchell](http://www.cs.cmu.edu/~tom/). In his definition, 

>a computer program is said to learn from **experience E** with respect to some class of **tasks T** and **performance measure P**, if its performance at tasks in T, as measured by P, improves with experience E.

In the checkers example, the program gains Experience ($E$) by playing with itself. The Task ($T$) for it is to play checkers (well) and its Performance ($P$) could be measured by the probability of winning the next game.

## 2. Supervised Learning vs Unsupervised Learning

There are several types of [learning algorithms](https://en.wikipedia.org/wiki/Machine_learning#Types_of_learning_algorithms). They can differ in their 
approach, the input or output data, or the task they try to solve. The two most important categories are supervised and unsupervised learning. Others not
covered in this post is Semi-supervised and Reinforcement Learning.

### 2.1 Supervised Learning

Imagine we have a dataset about adults in Glasgow. It describes their age, profession, education level, and other 
social-economic factors, **$X$**. It also contains their annual income, **$Y$**. 

You are under the assumption, that these social-economic variables can explain a Glaswegian's yearly income. 
Based on this dataset, you could build a model **$f(X)$** to estimate how much an adult in Glasgow with the given parameters earn
in a year. You have historical data to supervise you. 

>In supervised learning, your model is being trained on an already labelled dataset.

There are two more subcategories inside supervised learning based on the type of the output data, **$Y$**.

When **$Y$** is a continuous/real value, we call it a __*regression problem*__.
Predicting annual income is such a task.

If **$Y$** is discrete, it is a __*classification problem*__.
Predicting if someone's annual income is larger than 30K is a classification, since we have two discrete
categories (yes/no) as outputs.

### 2.2 Unsupervised Learning

In unsupervised learning, you don't have labelled data. The goal is to infer patterns without knowing or referring to an outcome,
relying on an assumed underlying structure in the variable space. 

__*Clustering*__ is when you try to split your data based on similarities between the variables. For instance, grouping records
in the Glasgow adult dataset together based on their social-economic background.

An example __*non-clustering*__ unsupervised learning is the [Cocktail Party Algorithm](https://en.wikipedia.org/wiki/Cocktail_party_effect). which separates voices
from different sources.

## 3. Model and Cost Function

### 3.1 Model Representation

Let's continue with estimating the annual income of adults in Glasgow. How does our model look like?

We have some input variables **$X$** (social-economic background) that we think explain our output variable
**$Y$** (annual income). Using functional notation, under this hypothesis **$Y = h(X)$**, where **$h$** is our hypothesis
function.

How do we form this function **$h$**? By considering our historical data. We use each observation pairs -
$(\bm{x^{(i)}},\bm{y^{(i)}})$ - to teach our model something about the dependency of **$Y$** on **$X$**.
In this case, $\bm{x^{(i)}}$ is the vector of features for observation i and $\bm{y^{(i)}}$ is the annual income 
for the same observation.

I'll talk about different functions to model this dependency at a later stage.

### 3.2 Cost Function

Assume you set up your hypothesis function $h$. Now on your training dataset you want to check is the predicted
annual income values are actually close to the observed ones. That is, you want to measure the distance 
of $h(\bm{x^{(i)}})$ and $\bm{y^{(i)}}$.

An obvious choice is to take the squared difference $(h(\bm{x^{(i)}})-\bm{y^{(i)}})^2$. However 
you want to regard all $(\bm{x^{(i)}},\bm{y^{(i)}})$ pairs when calculating the accuracy of $h$.

Hence, you have to sum up the distance between $h(\bm{x^{(i)}})$ and $\bm{y^{(i)}}$ over all observations
(let's say you have $N$ of them) and take the mean of this sum. This is called the 
__*squared error function*__ or __*mean squared error*__.

The value of the cost function gets big when our predictions are far from the observed output and decreases
when these get closer. Hence our goal is to minimize the cost function.


**Cost Function**: $J = \frac{1}{2N} \sum_{i=1}^N (h(\bm{x^{(i)}})-\bm{y^{(i)}})^2$
 

You might have noticed there is an additional division by 2 that I haven't covered yet. Later when we
calculate the gradient descent, we take the derivative of this expression. Differentiation of the squared expression cancels
out the added division by 2,
making the rest of the calculations easier.

## 4. Parameter Learning

Now we set up a simple model to predict annual income. We assume that the older you are, the more experience 
you have - and the more you earn! (it is a really naive model...)

Thus using our notation, $h_{\bm{\theta}}(\bm{x^{(i)}}) = \theta_0 + \theta_1 x^{(i)}_1$. Here, $\bm{\theta}=(\theta_0, \theta_1) $ 
is our parameter vector. By changing these values, we can increase or decrease 
the output of the cost function above.

The cost function $\bm{J_{\theta}}$ is a function of the pair $\bm{\theta}=(\theta_0, \theta_1) $ .
We can plot the values for each pair resulting in a picture similar to the one below.

![plot](plot.png)

The task of finding the most accurate hypothesis function with the given functional form translates
to finding the global minimum of this surface.






 

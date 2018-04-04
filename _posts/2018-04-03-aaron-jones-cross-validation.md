---
layout: post
title: "Cross Validation"
date: 2018-04-03
---

Cross validation is a phrase that is heard often in conversations about building models, but, based on my experiences, it is a concept that many individuals struggle to fully comprehend. One of the issues is understanding what cross validation (CV) actually does since it is frequently used in concert with other techniques. For example, CV is often combined with grid search to select the optimal hyperparameters of a model (think the penalization term in regularized regressions), but its purpose is not finding hyperparameters, that's the job of the grid search. CV is the process of splitting the data set being modeled into k number of folds, using all but one of the folds to train the model, and saving the remaining fold (data unseen by the model) to test the predictive viability of the model. By fitting and evaluating the predictive ability of the model k times, a better quantification of the performance and stability of the model is gleaned. The different data sets part is key because any one data set could include outliers, or an oddly low or high amount of some variable value, which could obscure the performance of the model on that one data set. The one-liner on CV is that it's a process for determining how well a given model will generalize to new data sets. I've employed CV in at least one of my previous posts using a module in sklearn. Here, I write my own cv in an effort to elucidate the concept.

```python

import math
import random

import numpy
import pandas
from sklearn import (
    linear_model,
    metrics
)

```

As is always the case, to start, the data is loaded into the working environment, cleaned, and formatted, so that it is suitable for modeling. The data used below contains characteristics of adults, including education level, income, native country, age, among others. I chose to use income as the response variable since it is a binary, categorical variable (>50K, <=50K), which is perfect for Logistic Regression. This data was pulled from the University of California Irvine machine learning database.

```python

def load_and_format():
    """
    Loads and formats the data for modeling.
    :return: Loaded and cleaned data.
    """
    url = (
        'https://archive.ics.uci.edu/ml/machine-learning-'
        'databases/adult/adult.data'
    )
    df = pandas.read_csv(
        url,
        header=None,
        names=[
            'age', 'workclass', 'fnlwgt', 'education',
            'education_num', 'marital_status',
            'occupation', 'relationship', 'race',
            'sex', 'capital_gain', 'capital_loss',
            'hours_per_week', 'native_country', 'income'
        ],
        skipinitialspace=True,
        na_values=['?']
    )

    df.dropna(axis=0, how='any', inplace=True)
    df.income.replace(['>50K', '<=50K'], [1, 0], inplace=True)

    categorical_variables = (
        df.dtypes[df.dtypes.values == 'object']
        .index
        .tolist()
    )
    df_with_dummies = pandas.get_dummies(
        data=df,
        columns=categorical_variables,
        dummy_na=False
    )

    return df_with_dummies

```

The function below first splits the data two ways: into training and testing, and into explanatory and response. Then, it fits a Logistic Regression model, and predicts on both the training and testing data.

```python

def tune_fit_predict(
    full_data, train_indices, test_indices
):
    """
    Fit the model and predict on both sets of data.
    :param full_data: Raw data.
    :param train_indices: Indices for training data.
    :param test_indices: Indices for test data.
    :return: Predicted values for the
    train and test data.
    """
    train = full_data.iloc[train_indices]
    x_train = train.drop('income', axis=1)
    y_train = train.income

    test = full_data.iloc[test_indices]
    x_test = test.drop('income', axis=1)
    y_test = test.income

    model = linear_model.LogisticRegression(
        penalty='l2',
        class_weight='balanced',
        random_state=0,
        max_iter=1e+06
    )
    model.fit(x_train, y_train)

    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)

    return (
        y_train, y_test,
        train_prediction, test_prediction
    )

```

This final function simply calculates the accuracy score, also known as classification rate, using the actual response value and the predicted response value.

```python

def accuracy_evaluation(actual, predicted):
    """
    Computes the classification rate.
    :param actual: True values.
    :param predicted: Predicted values.
    :return: Classification rate.
    """
    classification_rate = metrics.accuracy_score(
        actual, predicted
    )
    return classification_rate

```

```python

income_data = load_and_format()
print(income_data.head(3))
print(income_data.shape)
print(round(numpy.mean(income_data.income), 4))

```

Printing a few rows of the data reveals that the variables are mostly categorical and have been encoded as dummy variables. Note: sklearn requires categorical variables be encoded as dummies.

       age  fnlwgt  education_num  capital_gain  capital_loss  hours_per_week  \
    0   39   77516             13          2174             0              40   
    1   50   83311             13             0             0              13   
    2   38  215646              9             0             0              40   
    
       income  workclass_Federal-gov  workclass_Local-gov  workclass_Private  \
    0       0                      0                    0                  0   
    1       0                      0                    0                  0   
    2       0                      0                    0                  1   
    
                 ...              native_country_Portugal  \
    0            ...                                    0   
    1            ...                                    0   
    2            ...                                    0   
    
       native_country_Puerto-Rico  native_country_Scotland  native_country_South  \
    0                           0                        0                     0   
    1                           0                        0                     0   
    2                           0                        0                     0   
    
       native_country_Taiwan  native_country_Thailand  \
    0                      0                        0   
    1                      0                        0   
    2                      0                        0   
    
       native_country_Trinadad&Tobago  native_country_United-States  \
    0                               0                             1   
    1                               0                             1   
    2                               0                             1   
    
       native_country_Vietnam  native_country_Yugoslavia  
    0                       0                          0  
    1                       0                          0  
    2                       0                          0  

The data has over 30,000 rows and over 100 features (previously called variables). Only a quarter of the data is descriptive of adults making more than 50,000 dollars annually.

    (30162, 105)
    0.2489

Now to run the CV...

This chunk of code takes the existing index of the data, shuffles it, and divides it up into 10 folds of roughly equal size. Thus, I am here performing 10-fold CV. What is returned from this code is a list of lists of indices (indices idenifying which rows are assigned to which folds) from which the training and testing sets can be built.

```python

index = list(range(income_data.shape[0]))
index_shuffled = random.sample(index, len(index))
fold_size = int(math.ceil(len(index_shuffled) / 10))
list_of_folds = [
    index_shuffled[i:i + fold_size]
    for i in range(0, len(index_shuffled), fold_size)
]

```

Here, we loop through the list of lists giving each list in the list the opportunity to be the test, sometimes called holdout, data set. As mentioned before, for each selected fold we are able to use the remaining, in this case nine, folds to build the training data set. Next, the model is fitted on said training data and then used to predict the holdout data. In this case, I also predict on the data used to build the model. Lastly, the accuracy score is calculated and added to the appropriate results dictionary.

```python

train_classification_rate_dict = {}
test_classification_rate_dict = {}

for fold in range(len(list_of_folds)):
    list_of_folds_copy = list_of_folds.copy()
    test_index = list_of_folds_copy.pop(fold)
    train_index_nested = list_of_folds_copy
    train_index = [
        i for sub_list in train_index_nested for i in sub_list
    ]

    y_train, y_test, train_prediction, test_prediction = (
        tune_fit_predict(income_data, train_index, test_index)
    )
    print(
        round(numpy.mean(y_train), 4),
        round(numpy.mean(y_test), 4)
    )

    train_classification_rate = (
        accuracy_evaluation(y_train, train_prediction)
    )
    train_classification_rate_dict['fold{0}'.format(fold)] = (
        round(train_classification_rate, 4)
    )

    test_classification_rate = (
        accuracy_evaluation(y_test, test_prediction)
    )
    test_classification_rate_dict['fold{0}'.format(fold)] = (
        round(test_classification_rate, 4)
    )

```

I also print out the proportion of income >50K to make sure that all data sets contain proportions similar to the whole data. All of the proportions below are in line with the proportion for the whole data set listed previously. If this were not the case, then a slightly more complex CV, called stratified CV, should be employed, but that was not necessary here.

    0.2497 0.2423
    0.2498 0.2406
    0.2491 0.2476
    0.2486 0.2519
    0.2468 0.2678
    0.248 0.2575
    0.2505 0.2347
    0.2483 0.2542
    0.249 0.2486
    0.2495 0.2439

Finally, I compiled the results into a table for easy reading. Each row contains a label for which data set the results belong to, the scores on each iteration of the CV, and the average of all the individual fold scores. In this case, the model is quite stable (not much deviation in performance across folds) and generalizable (the training and testing scores are similar at each fold). If multiple models were being considered, CV would be employed to select the best one. For this example, I used only one model since the purpose was exploring CV and not model selection.

```python

cv_results_df = pandas.DataFrame()
cv_results_df = cv_results_df.append(
    train_classification_rate_dict, ignore_index=True
)
cv_results_df = cv_results_df.append(
    test_classification_rate_dict, ignore_index=True
)
cv_results_df['which_data'] = ['train', 'test']
cv_results_df['mean'] = cv_results_df.mean(axis=1).values
cv_results_df.set_index('which_data', inplace=True)
cv_results_df.style.format("{:.2%}")

```

<style  type="text/css" >
</style>  
<table id="T_b538f53a_37c8_11e8_abda_3035adb3d27c" > 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >fold0</th> 
        <th class="col_heading level0 col1" >fold1</th> 
        <th class="col_heading level0 col2" >fold2</th> 
        <th class="col_heading level0 col3" >fold3</th> 
        <th class="col_heading level0 col4" >fold4</th> 
        <th class="col_heading level0 col5" >fold5</th> 
        <th class="col_heading level0 col6" >fold6</th> 
        <th class="col_heading level0 col7" >fold7</th> 
        <th class="col_heading level0 col8" >fold8</th> 
        <th class="col_heading level0 col9" >fold9</th> 
        <th class="col_heading level0 col10" >mean</th> 
    </tr>    <tr> 
        <th class="index_name level0" >which_data</th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
        <th class="blank" ></th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_b538f53a_37c8_11e8_abda_3035adb3d27clevel0_row0" class="row_heading level0 row0" >train</th> 
        <td id="T_b538f53a_37c8_11e8_abda_3035adb3d27crow0_col0" class="data row0 col0" >80.93%</td> 
        <td id="T_b538f53a_37c8_11e8_abda_3035adb3d27crow0_col1" class="data row0 col1" >80.76%</td> 
        <td id="T_b538f53a_37c8_11e8_abda_3035adb3d27crow0_col2" class="data row0 col2" >80.75%</td> 
        <td id="T_b538f53a_37c8_11e8_abda_3035adb3d27crow0_col3" class="data row0 col3" >80.90%</td> 
        <td id="T_b538f53a_37c8_11e8_abda_3035adb3d27crow0_col4" class="data row0 col4" >80.91%</td> 
        <td id="T_b538f53a_37c8_11e8_abda_3035adb3d27crow0_col5" class="data row0 col5" >80.74%</td> 
        <td id="T_b538f53a_37c8_11e8_abda_3035adb3d27crow0_col6" class="data row0 col6" >80.92%</td> 
        <td id="T_b538f53a_37c8_11e8_abda_3035adb3d27crow0_col7" class="data row0 col7" >80.78%</td> 
        <td id="T_b538f53a_37c8_11e8_abda_3035adb3d27crow0_col8" class="data row0 col8" >80.58%</td> 
        <td id="T_b538f53a_37c8_11e8_abda_3035adb3d27crow0_col9" class="data row0 col9" >79.89%</td> 
        <td id="T_b538f53a_37c8_11e8_abda_3035adb3d27crow0_col10" class="data row0 col10" >80.72%</td> 
    </tr>    <tr> 
        <th id="T_b538f53a_37c8_11e8_abda_3035adb3d27clevel0_row1" class="row_heading level0 row1" >test</th> 
        <td id="T_b538f53a_37c8_11e8_abda_3035adb3d27crow1_col0" class="data row1 col0" >81.44%</td> 
        <td id="T_b538f53a_37c8_11e8_abda_3035adb3d27crow1_col1" class="data row1 col1" >81.21%</td> 
        <td id="T_b538f53a_37c8_11e8_abda_3035adb3d27crow1_col2" class="data row1 col2" >80.91%</td> 
        <td id="T_b538f53a_37c8_11e8_abda_3035adb3d27crow1_col3" class="data row1 col3" >80.61%</td> 
        <td id="T_b538f53a_37c8_11e8_abda_3035adb3d27crow1_col4" class="data row1 col4" >80.51%</td> 
        <td id="T_b538f53a_37c8_11e8_abda_3035adb3d27crow1_col5" class="data row1 col5" >80.48%</td> 
        <td id="T_b538f53a_37c8_11e8_abda_3035adb3d27crow1_col6" class="data row1 col6" >80.78%</td> 
        <td id="T_b538f53a_37c8_11e8_abda_3035adb3d27crow1_col7" class="data row1 col7" >80.41%</td> 
        <td id="T_b538f53a_37c8_11e8_abda_3035adb3d27crow1_col8" class="data row1 col8" >80.21%</td> 
        <td id="T_b538f53a_37c8_11e8_abda_3035adb3d27crow1_col9" class="data row1 col9" >78.96%</td> 
        <td id="T_b538f53a_37c8_11e8_abda_3035adb3d27crow1_col10" class="data row1 col10" >80.55%</td> 
    </tr></tbody> 
</table> 

Thanks for reading!

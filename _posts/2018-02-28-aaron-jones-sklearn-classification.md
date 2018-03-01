---
layout: post
title: "Sklearn Classification"
date: 2018-02-28
---

When undertaking a classification problem, it is commonplace to test multiple algorithms to insure that 1. the best possible performance is being achieved, 2. to validate the results, and 3. to identify potential modeling pitfalls. In classification, the goal is to identify some series of measurements as belonging to one of k classes. The Python library sklearn (or scikit-learn) includes numerous classification algorithms making it fairly easy to test multiple classifiers. In this example, I had some fun trying to classify tumors as benign or malignant using several features quantifying the key characteristics of said tumor. Specifically, the tumors in this dataset are breast cancer.

This project considers 9 algoirthms (Dummy, Gaussian Naive Bayes, Logistic Regression, Linear Discriminant Analysis, Quadratic Discriminant Analysis, k-Nearest Neighbors, Support Vector Machine, Random Forest, Gradient Boosting) and 7 performance metrics (Accuracy, Sensitivity, Specificity, Positive Predictive Value, Negative Predictive Value, F1, ROC AUC). Despite looking at numerous metrics, the one that I base the tuning (parameter optimization) process on is accuracy. Inside the first 5 performance metric functions, I give the formula for the metric, so I won't go into much detail except to explain TP, TN, FN, and FP.

TP = True Positive. The number of positive class values that were predicted to be positive.

TN = True Negative. The number of negative class values that were predicted to be negative.

FN = False Negative. The number of positive class values that were wrongly predicted to be negative.

FP = False Positive. The number of negative class values that were wrongly predicted to be positive.

The last two metrics (F1 and ROC AUC) are

F1 = A weighted average of precision (Positive Predictive Value) and recall (Sensitivity).

ROC AUC = The area under the ROC curve. The ROC curve is a plot of true positive rate (y-axis) against the false positive rate (x-axis). The closer the ROC AUC is to one, the larger the true positive rate and the smaller the false positive rate, which is the ideal scenario.

If this were an actual modeling project - instead of a fun after hours project - the process of picking a final algorithm would be much more involved, including additional visualizations (not the least of which being the ROC curves) and additional ensembling (will save for another post).

The algorithms will be briefly explained as we move through the example.

I start by loading in the needed packages, and defining some functions to help ease the process of tuning and scoring the models.

```python

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy
import pandas
import seaborn
from sklearn import *

%matplotlib notebook


def accuracy_score(matrix):
    # Accuracy = (TP + TN) / (TP + TN + FN + FP)
    accuracy = (
        (matrix[0, 0] + matrix[1, 1])
        / float(matrix[0, 0] + matrix[0, 1] + matrix[1, 0] + matrix[1, 1])
    )
    return accuracy


def sensitivity_score(matrix):
    # Sensitivity = TP / (FN + TP)
    return matrix[1, 1] / float(matrix[1, 0] + matrix[1, 1])


def specificity_score(matrix):
    # Specificity = TN / (TN + FP)
    return matrix[0, 0] / float(matrix[0, 0] + matrix[0, 1])


def positive_predictive_value_score(matrix):
    # Positive Predictive Value = TP / (FP + TP)
    return matrix[1, 1] / float(matrix[0, 1] + matrix[1, 1])


def negative_predictive_value_score(matrix):
    # Negative Predictive Value = TN / (TN + FN)
    return matrix[0, 0] / float(matrix[0, 0] + matrix[1, 0])


def f1_score(actual, predicted):
    return metrics.f1_score(actual, predicted)


def roc_auc_score(actual, predicted):
    return metrics.roc_auc_score(actual, predicted)


def data_split(data, target):
    x = data.drop(target, axis=1)
    y = data[target]
    x_train, x_test, y_train, y_test = (
        model_selection.train_test_split(
            x, y, test_size=0.20, shuffle=True, random_state=0
        )
    )
    return x, y, x_train, x_test, y_train, y_test


def data_transform(data, target, full=False):
    x, y, x_train, x_test, y_train, y_test = (
        data_split(data, target)
    )
    std = preprocessing.StandardScaler()
    if full == True:
        x_transform = std.fit_transform(x)
        output = (x_transform, y)
    else:
        x_train_transform = std.fit_transform(x_train)
        x_test_transform = std.transform(x_test)
        output = (
            x_train_transform, x_test_transform,
            y_train, y_test
        )
    return output


def model_scoring(which_data, actual, predicted):
    confusion = metrics.confusion_matrix(actual, predicted)

    acc = round(accuracy_score(confusion), 2)
    sen = round(sensitivity_score(confusion), 2)
    spe = round(specificity_score(confusion), 2)
    ppv = round(positive_predictive_value_score(confusion), 2)
    npv = round(negative_predictive_value_score(confusion), 2)
    f1s = round(f1_score(actual, predicted), 2)
    auc = round(roc_auc_score(actual, predicted), 2)

    print('Subset: {0}'.format(which_data))
    print('Accuracy: {0}'.format(acc))
    print('Sensitivity: {0}'.format(sen))
    print('Specificity: {0}'.format(spe))
    print('Positive Predictive Value: {0}'.format(ppv))
    print('Negative Predictive Value: {0}'.format(npv))
    print('F1 (Weighted Average of Precision and Recall): {0}'.format(f1s))
    print('ROC AUC: {0}'.format(auc))

    scores = (acc, sen, spe, ppv, npv, f1s, auc)
    return scores


def model_fit(algorithm, parameters, data, target):
    x_train_transform, x_test_transform, y_train, y_test = (
        data_transform(data, target, full=False)
    )

    model = model_selection.GridSearchCV(
        algorithm,
        param_grid=parameters,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        verbose=0
    )
    model.fit(x_train_transform, y_train)

    y_train_predicted = model.predict(x_train_transform)
    y_test_predicted = model.predict(x_test_transform)

    data_dictionary = {
        'x_train_transform': x_train_transform,
        'x_test_transform': x_test_transform,
        'y_train_actual': y_train,
        'y_test_actual': y_test,
        'y_train_predicted': y_train_predicted,
        'y_test_predicted': y_test_predicted
    }

    scoring_list = [
        ('Training', y_train, y_train_predicted),
        ('Holdout', y_test, y_test_predicted)
    ]

    for which_data, actual, predicted in scoring_list:
        scores = model_scoring(which_data, actual, predicted)

    outputs = (model, data_dictionary, scores)
    return outputs

```

Next, we load in the data. I've decided to define the malignant observations 1 and the benign observations 0. Note that the feature names have been abbreviated for space by removing the word mean from each variable. So, 'area' should actually be 'area mean.' Also, instead of going through a cumbersome imputation process, I opt to just remove all rows containing missing data.

```python

cancer = (
    pandas.read_csv(
        '/Users/aaronjones/Classification-example-sklearn-python/BreastCancer.csv',
        header=0,
        names=[
            'id', 'diag', 'radius', 'texture', 'perimeter', 'area',
            'smoothness', 'compactness', 'concavity', 'concave_points',
            'symmetry', 'fractal_dimension'
        ]
    )
    .sample(frac=1)
    .dropna(axis=0)
)

cancer.diag.replace(['M', 'B'], [1, 0], inplace=True)
cancer.drop(['id'], axis=1, inplace=True)
cancer.head()

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
      <th>diag</th>
      <th>radius</th>
      <th>texture</th>
      <th>perimeter</th>
      <th>area</th>
      <th>smoothness</th>
      <th>compactness</th>
      <th>concavity</th>
      <th>concave_points</th>
      <th>symmetry</th>
      <th>fractal_dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>210</th>
      <td>1</td>
      <td>20.58</td>
      <td>22.14</td>
      <td>134.70</td>
      <td>1290.0</td>
      <td>0.09090</td>
      <td>0.13480</td>
      <td>0.1640</td>
      <td>0.09561</td>
      <td>0.1765</td>
      <td>0.05024</td>
    </tr>
    <tr>
      <th>479</th>
      <td>1</td>
      <td>16.25</td>
      <td>19.51</td>
      <td>109.80</td>
      <td>815.8</td>
      <td>0.10260</td>
      <td>0.18930</td>
      <td>0.2236</td>
      <td>0.09194</td>
      <td>0.2151</td>
      <td>0.06578</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1</td>
      <td>13.17</td>
      <td>18.66</td>
      <td>85.98</td>
      <td>534.6</td>
      <td>0.11580</td>
      <td>0.12310</td>
      <td>0.1226</td>
      <td>0.07340</td>
      <td>0.2128</td>
      <td>0.06777</td>
    </tr>
    <tr>
      <th>343</th>
      <td>1</td>
      <td>19.68</td>
      <td>21.68</td>
      <td>129.90</td>
      <td>1194.0</td>
      <td>0.09797</td>
      <td>0.13390</td>
      <td>0.1863</td>
      <td>0.11030</td>
      <td>0.2082</td>
      <td>0.05715</td>
    </tr>
    <tr>
      <th>456</th>
      <td>0</td>
      <td>11.63</td>
      <td>29.29</td>
      <td>74.87</td>
      <td>415.1</td>
      <td>0.09357</td>
      <td>0.08574</td>
      <td>0.0716</td>
      <td>0.02017</td>
      <td>0.1799</td>
      <td>0.06166</td>
    </tr>
  </tbody>
</table>
</div>




It's always good practice to visualize the data and compute some basic statistics prior to beginning the modeling. Here, I am interested in the correlation between features in order to identify possible multicollinearity issues.

```python

plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features')
seaborn.heatmap(
  data=cancer.drop('diag', axis=1).astype(float).corr(),
  linewidths=0.1,
  vmax=1.0,
  square=True,
  cmap=plt.cm.RdBu,
  linecolor='white',
  annot=True
)

```


![](/images/2018-02-24-aaron-jones-sklearn-classification_files/figure-markdown_github/output_2_1.png)



I now give one example of a two dimensional scatterplot, which is of 'radius mean' and 'fractal dimension mean.' The two features do separate nicely into malignant and benign groups, albeit some overlap exists.

```python

scat = seaborn.FacetGrid(
    data=cancer[['radius', 'fractal_dimension', 'diag']],
    hue='diag',
    aspect=1.5
).map(
    plt.scatter, 'radius', 'fractal_dimension'
).add_legend()

```


![](/images/2018-02-24-aaron-jones-sklearn-classification_files/figure-markdown_github/output_3_0.png)



One of the biggest struggles in modeling high dimensional data is visualizing all the data simultaneously. One nice way to do this is to use dimensionality reduction algorithms. The first one I employ is Principal Component Analysis. The general idea behind PCA is to use an orthogonal transform to create a series of uncorrelated, linear combinations of the original variables (i.e. principal components). In general, the number of components used in further analysis depends on how many components it takes to explain the majority of the variance in the data. I typically use 80% as my threshold. In this case, I get to just about 80% with the first two components. Note that PCA forms the components so that the first one explains the most variance, the second the second most variance, and so on.

```python

x_full, y_full = (
    data_transform(cancer, 'diag', full=True)
)

pca = decomposition.PCA(n_components=2)
x_pca = pca.fit(x_full).transform(x_full)

print('Individual % Variance Explained by First Two Components: {0}'
      .format(pca.explained_variance_ratio_))

print('Total % Variance Explained by First Two Components: {0}'
      .format(sum(pca.explained_variance_ratio_)))

```

    Individual % Variance Explained by First Two Components: [0.54851522 0.25080488]
    Total % Variance Explained by First Two Components: 0.7993201049734291


Now, I plot the first two principal components and get another look at the data.

```python

colors = ['blue', 'red']
plot_info = zip(colors, [0, 1], ['Benign', 'Malignant'])
plt.figure()
for col, i, name in plot_info:
    plt.scatter(
        x_pca[y_full == i, 0], x_pca[y_full == i, 1],
        color=col, alpha=0.8, lw=2, label=name
    )
plt.legend(loc='best', shadow=True, scatterpoints=1)
plt.title('PCA Cancer Data')

```


![](/images/2018-02-24-aaron-jones-sklearn-classification_files/figure-markdown_github/output_5_1.png)



Another dimensionality reduction technique useful in high dimensional data plotting is Multidimensional Scaling (MDS). Unlike PCA, which creates linear combinations of the features, MDS computes the distances between all the observations and puts those observations in a smaller dimensional space while simultaneously maintaining the aforementioned distances. That way it is easier to assess the similarity and or dissimilarity between observations. In this case, I reduced the dimensionality down to both 2 and 3 dimensions. In the 3 dimensional plot, it seems like there may be some separation between classes, despite the bad orientation of the plot, but in 2 dimensions there doesn't seem to be any separation. An interesting phenomenon given the clear separation in the other two plots. 

```python

mds2 = manifold.MDS(
    n_components=2, metric=False, max_iter=10000, eps=1e-8,
    n_jobs=1, random_state=0, dissimilarity='euclidean'
)
em2d = mds2.fit(x_full).embedding_

mds3 = manifold.MDS(
    n_components=3, metric=False, max_iter=10000, eps=1e-8,
    n_jobs=1, random_state=0, dissimilarity='euclidean'
)
em3d = mds3.fit(x_full).embedding_

fig = plt.figure(figsize=(5 * 2, 5))
plt2 = fig.add_subplot(121)  # plot 2d
plt2.scatter(
    em2d[y_full == 0, 0], em2d[y_full == 0, 1],
    s=20, color='blue', label='Benign'
)
plt2.scatter(
    em2d[y_full == 1, 0], em2d[y_full == 1, 1],
    s=20, color='red', label='Malignant'
)
plt.axis('tight')
plt3 = fig.add_subplot(122, projection='3d')  # plot 3d
plt3.scatter(
    em3d[y_full == 0, 0], em3d[y_full == 0, 1], em3d[y_full == 0, 2],
    s=20, color='blue', label='Benign'
)
plt3.scatter(
    em3d[y_full == 1, 0], em3d[y_full == 1, 1], em3d[y_full == 1, 2],
    s=20, color='red', label='Malignant'
)
plt3.view_init(42, 101)
plt3.view_init(-130, -33)
plt.suptitle('2D and 3D Multidimensional Scaling Plots')
plt.axis('tight')
plt.legend(loc='best', shadow=True, scatterpoints=1)
plt.show()

```


![](/images/2018-02-24-aaron-jones-sklearn-classification_files/figure-markdown_github/output_6_0.png)



To start the modeling, I run a dummy classifier in order to establish an accuracy baseline to which I can compare the more sophisticated algorithms. Notice that while the accuracy is not great, it definitely isn't 50%. This has to do with the number of each class included in the dataset.

You'll notice that the results come from two subsets of the data, titled Training and Holdout. The training data is what the model used to tune, while the holdout data is a subset of the data that was held back in order to test the algorithms ability to predict on unseen data. The difference between the numbers says a lot about the quality of the model, including whether or not there is any overfitting.

```python

dummy_grid = {
    'strategy': ['most_frequent', 'stratified', 'uniform'],
    'random_state': [0]
}

dummy_outputs = model_fit(
    algorithm=dummy.DummyClassifier(),
    parameters=dummy_grid, data=cancer, target='diag'
)
dummy_model, dummy_data_dict, dummy_scores = dummy_outputs

```

    Subset: Training
    Accuracy: 0.64
    Sensitivity: 0.0
    Specificity: 1.0
    Positive Predictive Value: nan
    Negative Predictive Value: 0.64
    F1 (Weighted Average of Precision and Recall): 0.0
    ROC AUC: 0.5
    Subset: Holdout
    Accuracy: 0.6
    Sensitivity: 0.0
    Specificity: 1.0
    Positive Predictive Value: nan
    Negative Predictive Value: 0.6
    F1 (Weighted Average of Precision and Recall): 0.0
    ROC AUC: 0.5


Next up, Gaussian Naive Bayes. This is a fairly simple probabilistic classifier, which is based on Bayes' Theorem. The naive component is that all the features are assumed by the classifier to be independent. Even with this simple classifier, the achieved accuracy is far better than that of the dummy classifier.

```python

gnb_grid = {
    'priors': [None]
}

gnb_outputs = model_fit(
    algorithm=naive_bayes.GaussianNB(),
    parameters=gnb_grid, data=cancer, target='diag'
)
gnb_model, gnb_data_dict, gnb_scores = gnb_outputs

```

    Subset: Training
    Accuracy: 0.92
    Sensitivity: 0.87
    Specificity: 0.95
    Positive Predictive Value: 0.91
    Negative Predictive Value: 0.93
    F1 (Weighted Average of Precision and Recall): 0.89
    ROC AUC: 0.91
    Subset: Holdout
    Accuracy: 0.89
    Sensitivity: 0.8
    Specificity: 0.94
    Positive Predictive Value: 0.9
    Negative Predictive Value: 0.88
    F1 (Weighted Average of Precision and Recall): 0.85
    ROC AUC: 0.87


Logistic regression is a robust binary classifier, which belongs to the family of generalized linear models. That is, the discrete response is transformed, in this case using the logit function, to a continuous variable, which can then be modeled linearly. Note that due to the transformation the predictions are no longer particular values, but instead the odds of those particular values.

```python

lr_grid = {
    'penalty': ['l1', 'l2'],
    'C': numpy.linspace(0.01, 10000, 20),
    'random_state': [0]
}

logistic_outputs = model_fit(
    algorithm=linear_model.LogisticRegression(),
    parameters=lr_grid, data=cancer, target='diag'
)
logistic_model, logistic_data_dict, logistic_scores = logistic_outputs

```

    Subset: Training
    Accuracy: 0.95
    Sensitivity: 0.92
    Specificity: 0.97
    Positive Predictive Value: 0.94
    Negative Predictive Value: 0.96
    F1 (Weighted Average of Precision and Recall): 0.93
    ROC AUC: 0.94
    Subset: Holdout
    Accuracy: 0.91
    Sensitivity: 0.85
    Specificity: 0.96
    Positive Predictive Value: 0.93
    Negative Predictive Value: 0.9
    F1 (Weighted Average of Precision and Recall): 0.89
    ROC AUC: 0.9


Linear Discriminant Analysis tries to identify a linear combination of the data that can serve as a linear decision boundary between 2 plus classes. The search for a linear combination makes it similar to PCA. LDA assumes that the conditional probability distributions (i.e. the data given class 1, the data given class 2, etc.) are normally distributed and that all the covariance matrices are identical.

```python

lda_grid = {
    'solver': ['svd'],
    'store_covariance': [True]
}

lda_outputs = model_fit(
    algorithm=discriminant_analysis.LinearDiscriminantAnalysis(),
    parameters=lda_grid, data=cancer, target='diag'
)
lda_model, lda_data_dict, lda_scores = lda_outputs

```

    Subset: Training
    Accuracy: 0.94
    Sensitivity: 0.87
    Specificity: 0.98
    Positive Predictive Value: 0.96
    Negative Predictive Value: 0.93
    F1 (Weighted Average of Precision and Recall): 0.91
    ROC AUC: 0.93
    Subset: Holdout
    Accuracy: 0.93
    Sensitivity: 0.83
    Specificity: 1.0
    Positive Predictive Value: 1.0
    Negative Predictive Value: 0.89
    F1 (Weighted Average of Precision and Recall): 0.9
    ROC AUC: 0.91


Similar to LDA is Quadratic Discriminant Analysis. The only difference ia that in QDA the covariance matrices are not assumed to be identical. This allows the decision boundary to become more complex.

```python

qda_grid = {
    'store_covariance': [True]
}

qda_outputs = model_fit(
    algorithm=discriminant_analysis.QuadraticDiscriminantAnalysis(),
    parameters=qda_grid, data=cancer, target='diag'
)
qda_model, qda_data_dict, qda_scores = qda_outputs

```

    Subset: Training
    Accuracy: 0.95
    Sensitivity: 0.88
    Specificity: 0.98
    Positive Predictive Value: 0.97
    Negative Predictive Value: 0.94
    F1 (Weighted Average of Precision and Recall): 0.92
    ROC AUC: 0.93
    Subset: Holdout
    Accuracy: 0.9
    Sensitivity: 0.83
    Specificity: 0.96
    Positive Predictive Value: 0.93
    Negative Predictive Value: 0.89
    F1 (Weighted Average of Precision and Recall): 0.87
    ROC AUC: 0.89


K-Nearest Neighbors is a very simple, but often times very powerful classifier. The general idea is that the k points nearest the point that is being predicted are pooled to determine said prediction. In general, a simple majority wins. For example, if I set k = 10, I am going to consider the 10 observations closest to my new observation. If 6 of the 10 belong to class 1 and the other 4 to class 0, then the new point would be classified as 1.

```python

knn_grid = {
    'n_neighbors': numpy.linspace(1, 10, 10).astype(int),
    'weights': ['uniform', 'distance'],
    'p': numpy.linspace(1, 6, 6).astype(int),
    'algorithm': ['auto']
}

knn_outputs = model_fit(
    algorithm=neighbors.KNeighborsClassifier(),
    parameters=knn_grid, data=cancer, target='diag'
)
knn_model, knn_data_dict, knn_scores = knn_outputs

```

    Subset: Training
    Accuracy: 0.96
    Sensitivity: 0.9
    Specificity: 0.99
    Positive Predictive Value: 0.99
    Negative Predictive Value: 0.94
    F1 (Weighted Average of Precision and Recall): 0.94
    ROC AUC: 0.94
    Subset: Holdout
    Accuracy: 0.92
    Sensitivity: 0.85
    Specificity: 0.97
    Positive Predictive Value: 0.95
    Negative Predictive Value: 0.9
    F1 (Weighted Average of Precision and Recall): 0.9
    ROC AUC: 0.91


Support Vector Machines are non-probabilistic classifiers, which means that the algorithm decides between the two classes without assigning any probabilities. Here, the data are mapped, using what is called a kernel, into a new space in which the data are linearly separated by as wide a margin as possible. The new observations are classified based on their mappings in the new space. The kernels faciliate the algorithms ability to do non-linear classification as well as linear classification.

```python

svc_grid = {
    'C': numpy.linspace(0.01, 10000, 20),
    'kernel': ['linear', 'rbf'],
    'class_weight': ['balanced', None],
    'random_state': [0]
}

svc_outputs = model_fit(
    algorithm=svm.SVC(),
    parameters=svc_grid, data=cancer, target='diag'
)
svc_model, svc_data_dict, svc_scores = svc_outputs

```

    Subset: Training
    Accuracy: 0.94
    Sensitivity: 0.9
    Specificity: 0.97
    Positive Predictive Value: 0.94
    Negative Predictive Value: 0.95
    F1 (Weighted Average of Precision and Recall): 0.92
    ROC AUC: 0.93
    Subset: Holdout
    Accuracy: 0.92
    Sensitivity: 0.85
    Specificity: 0.97
    Positive Predictive Value: 0.95
    Negative Predictive Value: 0.9
    F1 (Weighted Average of Precision and Recall): 0.9
    ROC AUC: 0.91


Random forests are an ensembled algorithm built on decision trees. Here, we create B bootstrap (another topic for another day) samples of the original data, which are used to create B decision trees. What separates random forests from the bagging algorithms is that each bootstrapped decision tree is built using a subset of the features. The goal here is to build uncorrelated trees. Note that the process determining the subset of features to be used in each tree is random. While single decision trees tend to overfit, the construction of a whole forest of trees typically corrects for that problem. In the end, the predictions from the individual trees are aggregated to produce the final prediction.

```python

rf_grid = {
    'n_estimators': [1000],
    'max_features': numpy.linspace(2, cancer.shape[1]-1, 8).astype(int),
    'max_depth': numpy.linspace(2, 8, 7).astype(int),
    'class_weight': ['balanced', None],
    'bootstrap': [True],
    'oob_score': [True],
    'random_state': [0],
    'n_jobs': [-1]
}

rf_outputs = model_fit(
    algorithm=ensemble.RandomForestClassifier(),
    parameters=rf_grid, data=cancer, target='diag'
)
rf_model, rf_data_dict, rf_scores = rf_outputs

```

    Subset: Training
    Accuracy: 1.0
    Sensitivity: 1.0
    Specificity: 1.0
    Positive Predictive Value: 1.0
    Negative Predictive Value: 1.0
    F1 (Weighted Average of Precision and Recall): 1.0
    ROC AUC: 1.0
    Subset: Holdout
    Accuracy: 0.92
    Sensitivity: 0.89
    Specificity: 0.94
    Positive Predictive Value: 0.91
    Negative Predictive Value: 0.93
    F1 (Weighted Average of Precision and Recall): 0.9
    ROC AUC: 0.92


And lastly, gradient boosting. Like random forests, grandient boosters are ensemble learners; however, unlike random forests, the decision trees in gradient boosters are built iteratively instead of simultaneously. Each subsequent tree learns to correct the previous tree. In essence, each new tree is focusing on correctly predicting the observations that were incorrectly predicted by the previous tree. The idea being to build a collection of weak learners, which combine to form one strong learner. Gradient boosting algorithms have to be tuned carefully as they can easily overfit the data.

```python

gb_grid = {
    'learning_rate': [0.01],
    'n_estimators': [500],
    'max_depth': numpy.linspace(2, 8, 7).astype(int),
    'max_features': numpy.linspace(2, cancer.shape[1]-1, 8).astype(int),
    'random_state': [0]
}

gb_outputs = model_fit(
    algorithm=ensemble.GradientBoostingClassifier(),
    parameters=gb_grid, data=cancer, target='diag'
)
gb_model, gb_data_dict, gb_scores = gb_outputs

```

    Subset: Training
    Accuracy: 1.0
    Sensitivity: 1.0
    Specificity: 1.0
    Positive Predictive Value: 1.0
    Negative Predictive Value: 1.0
    F1 (Weighted Average of Precision and Recall): 1.0
    ROC AUC: 1.0
    Subset: Holdout
    Accuracy: 0.9
    Sensitivity: 0.85
    Specificity: 0.94
    Positive Predictive Value: 0.91
    Negative Predictive Value: 0.9
    F1 (Weighted Average of Precision and Recall): 0.88
    ROC AUC: 0.89


```python

scores_list = [
    dummy_scores, gnb_scores, logistic_scores, lda_scores,
    qda_scores, knn_scores, svc_scores, rf_scores, gb_scores
]

holdout_stats = pandas.DataFrame(
    scores_list,
    columns=[
        'Accuracy', 'Sensitivity', 'Specificity',
        'PPV', 'NPV', 'F1 Score', 'ROC AUC Score'
    ]
)

holdout_stats['Algorithms'] = [
    'Dummy', 'Gaussian Naive Bayes', 'Logistic Regression',
    'Linear Discriminant Analysis', 'Quadratic Discriminant Analysis',
    'K Nearest Neighbors', 'Support Vector Classifier',
    'Random Forest', 'Gradient Boosting'
]

print(holdout_stats)

```

The table below shows the holdout metrics for each of the 9 algorithms. Using accuracy, the winning algorithm for this dataset is Linear Discriminant Analysis. It turns out that this algorithm got 94% accuracy on the training data, which, since the holdout accuracy is 93%, means that the algorithm is stable and not overfitting. In reality, all of the algorithms, with the exception of the dummy classifier, perform very well on this dataset. It turns out this dataset didn't, on this basic level, provide any real challenges, but certainly not all datasets will be this way!

       Accuracy  Sensitivity  Specificity   PPV   NPV  F1 Score  ROC AUC Score                       Algorithms
    0      0.60         0.00         1.00   NaN  0.60      0.00           0.50                            Dummy
    1      0.89         0.80         0.94  0.90  0.88      0.85           0.87             Gaussian Naive Bayes
    2      0.91         0.85         0.96  0.93  0.90      0.89           0.90              Logistic Regression
    3      0.93         0.83         1.00  1.00  0.89      0.90           0.91     Linear Discriminant Analysis
    4      0.90         0.83         0.96  0.93  0.89      0.87           0.89  Quadratic Discriminant Analysis
    5      0.92         0.85         0.97  0.95  0.90      0.90           0.91              K Nearest Neighbors
    6      0.92         0.85         0.97  0.95  0.90      0.90           0.91        Support Vector Classifier
    7      0.92         0.89         0.94  0.91  0.93      0.90           0.92                    Random Forest
    8      0.90         0.85         0.94  0.91  0.90      0.88           0.89                Gradient Boosting

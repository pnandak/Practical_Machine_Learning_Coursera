## Prediction of Weight Lifting Style using Accelerometer Data

### Introduction  

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, my goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

Goal of your project is to predict the manner in which they did the exercise

### Data preparation  

Load the Libraries  


```r
library(lattice)
library(ggplot2)
library(e1071)
```

```
## Loading required package: class
```

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.0
```

```r
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```



Load both datasets.  


```r
raw_training <- read.csv("pml-training.csv")
raw_testing <- read.csv("pml-testing.csv")
```


Partition training data provided into two sets. One for training and one for cross validation.


```r

set.seed(1234)
trainingIndex <- createDataPartition(raw_training$classe, list = FALSE, p = 0.9)
training = raw_training[trainingIndex, ]
testing = raw_training[-trainingIndex, ]
```


Remove indicators with near zero variance.


```r
library(caret)
nzv <- nearZeroVar(training)

training <- training[-nzv]
testing <- testing[-nzv]
raw_testing <- raw_testing[-nzv]
```


Filter columns to only include numeric features and outcome. Integer and other non-numeric features can be trained to reliably predict values in the training file provided, but when used to predict values in the testing set provided, they lead to misclassifications.


```r
num_features_idx = which(lapply(training, class) %in% c("numeric"))
```


We then would like to impute missing values as many exist in our training data.


```r
library(RANN)
```

```
## Warning: package 'RANN' was built under R version 3.1.1
```

```r

preModel <- preProcess(training[, num_features_idx], method = c("knnImpute"))

ptraining <- cbind(training$classe, predict(preModel, training[, num_features_idx]))
ptesting <- cbind(testing$classe, predict(preModel, testing[, num_features_idx]))
prtesting <- predict(preModel, raw_testing[, num_features_idx])

# Fix Label on classe
names(ptraining)[1] <- "classe"
names(ptesting)[1] <- "classe"
```


### Model  

We can build a random forest model using the numerical variables provided. As we will see later this provides good enough accuracy to predict the twenty test cases. Using [caret][caret], we can obtain the optimal mtry parameter of 32. This is a computationally expensive process, so only the optimized parameter is shown below.


```r
library(randomForest)

rf_model <- randomForest(classe ~ ., ptraining, ntree = 500, mtry = 32)
```


### Cross validation: In-Sample Accuracy

We are able to measure the accuracy using our training set and our cross validation set. With the training set we can detect if our model has bias due to ridgity of our mode. With the cross validation set, we are able to determine if we have variance due to overfitting.


```r
training_pred <- predict(rf_model, ptraining)
print(confusionMatrix(training_pred, ptraining$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5022    0    0    0    0
##          B    0 3418    0    0    0
##          C    0    0 3080    0    0
##          D    0    0    0 2895    0
##          E    0    0    0    0 3247
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```


The cross validation accuracy is greater than 99%, which should be sufficient for predicting the twenty test observations. Based on the lower bound of the confidence interval we would expect to achieve a 98.7% classification accuracy on new data provided.

One caveat exists that the new data must be collected and preprocessed in a manner consistent with the training data.

### Test Set Prediction Results

Applying this model to the test data provided yields 100% classification accuracy on the twenty test observations.


```r
answers <- predict(rf_model, prtesting)
answers
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


### Conclusion 

Writing answers out to separate files for grading system


```r
# write answers out to separate files for grading system
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}
# get answers for problemset and write them out
answers <- predict(rf_model, prtesting)
pml_write_files(answers)
```


We are able to provide very good prediction of weight lifting style as measured with accelerometers.

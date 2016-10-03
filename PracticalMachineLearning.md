# PracticalMachineLearning
R. Debbe  
October 2, 2016  





This document reports the work done as project for the Coursera Practical Machine Learning where we were given data collected on six subjects that performed Weight Lifting Exercise "Bicep Curl-up" while wearing accelerometers mounted in one arm, one forearm and their  belt near the body's center of gravity. Results from that study were published and their data was made public. Details about that research can be found at: 
 <http://groupware.les.inf.puc-rio.br/har>

The following libraries were loaded to the session during this project:


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(mlbench)
```

# The original training and testing data sets


```r
training <- read.csv(file="../pml-training.csv", header=TRUE, sep=",")
testing  <- read.csv(file="../pml-testing.csv", header=TRUE, sep=",")
```
The training data has 19622 samples with 160 features. The test data set has only 20 samples with the same number of features but a considerable fraction of the test features have different data types as compared to the corresponding variables in the training set; they have been stored as logical vectors while the training set has them as factors. The followin snippet of the data summary shows such mismatch.


```
## 'data.frame':	19622 obs. of  6 variables:
##  $ skewness_roll_belt  : Factor w/ 395 levels "","-0.003095",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_belt.1: Factor w/ 338 levels "","-0.005928",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_yaw_belt   : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ max_roll_belt       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_belt      : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_belt        : Factor w/ 68 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
```

```
## 'data.frame':	20 obs. of  6 variables:
##  $ skewness_roll_belt  : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_belt.1: logi  NA NA NA NA NA NA ...
##  $ skewness_yaw_belt   : logi  NA NA NA NA NA NA ...
##  $ max_roll_belt       : logi  NA NA NA NA NA NA ...
##  $ max_picth_belt      : logi  NA NA NA NA NA NA ...
##  $ max_yaw_belt        : logi  NA NA NA NA NA NA ...
```

These features found to have mismatched data types were then removed from both data sets:


```r
txt<- c(names(training))
reducedTraining<- training[, -grep("^kurtosis|^skewness|^max|^min|^avg|^var|^stddev|^amplitude", 
                                   txt, perl=TRUE, value=FALSE)]
txt<- c(names(testing))                                   
reducedTesting<- testing[, -grep("^kurtosis|^skewness|^max|^min|^avg|^var|^stddev|^amplitude", 
                                   txt, perl=TRUE, value=FALSE)]
```

The first 7 variables on both data sets have no prediction power and appear to have purely book-keeping puposes; they were also removed with:


```r
reducedTraining <- reducedTraining[, -c(1, 2, 3, 4, 5, 6, 7)]
reducedTesting  <- reducedTesting[, -c(1, 2, 3, 4, 5, 6, 7)]
```

The remaining 53 features appear as good candidates for predictor as their correlations are visible the following summary plot:

![](PracticalMachineLearning_files/figure-html/summary plot-1.png)<!-- -->
We then divide the training data set in two sets to be able to do cross-validation as we fit several models.


```r
inTrain <- createDataPartition(y=reducedTraining$classe, p=0.5, list=FALSE)
subtraining <- reducedTraining[inTrain, ]
subtesting <- reducedTraining[-inTrain, ]

dim(subtraining)
```

```
## [1] 9812   53
```

```r
dim(subtesting)
```

```
## [1] 9810   53
```

We used the R package mlbench to compare three models: Random Forest (rf) with max number of trees set to 300, Boosted Random Forest (gbm) with default parameters, and a simpler classifier like Support Vector Machine (svm) also with default parameters. The comparison is done with 10 K-Fold cross validation and tree repeats:


```r
control <- trainControl(method="repeatedcv", number=10, repeats=3)
```

The actual training run overnight on OSX 'El Capitan' :


```r
set.seed(42)
svmModel<- train(classe ~., data=subtraining, method="svmRadial",  trControl=control)
```

```
## Loading required package: kernlab
```

```
## 
## Attaching package: 'kernlab'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     alpha
```

```r
confusionMatrix(subtesting$classe, predict(svmModel, subtesting[,-53]))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2762    9   14    5    0
##          B  184 1611   89    3   11
##          C   20   99 1547   37    8
##          D   13   17  195 1377    6
##          E    7   20   72   52 1652
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9122          
##                  95% CI : (0.9065, 0.9178)
##     No Information Rate : 0.3044          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8887          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9250   0.9174   0.8070   0.9342   0.9851
## Specificity            0.9959   0.9644   0.9792   0.9723   0.9814
## Pos Pred Value         0.9900   0.8488   0.9041   0.8563   0.9163
## Neg Pred Value         0.9681   0.9817   0.9543   0.9882   0.9969
## Prevalence             0.3044   0.1790   0.1954   0.1503   0.1709
## Detection Rate         0.2815   0.1642   0.1577   0.1404   0.1684
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9604   0.9409   0.8931   0.9532   0.9833
```

```r
set.seed(42)
gbmModel<- train(classe ~., data=subtraining, method="gbm", verbose=FALSE, trControl=control)
```

```
## Loading required package: gbm
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.1
```

```
## Loading required package: plyr
```

```r
confusionMatrix(subtesting$classe, predict(gbmModel, subtesting[,-53]))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2742   30   10    3    5
##          B   64 1768   61    1    4
##          C    0   50 1638   21    2
##          D    1    6   72 1522    7
##          E    3   30   17   26 1727
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9579          
##                  95% CI : (0.9537, 0.9618)
##     No Information Rate : 0.2864          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9467          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9758   0.9384   0.9110   0.9676   0.9897
## Specificity            0.9931   0.9836   0.9909   0.9896   0.9906
## Pos Pred Value         0.9828   0.9315   0.9573   0.9465   0.9578
## Neg Pred Value         0.9903   0.9853   0.9802   0.9938   0.9978
## Prevalence             0.2864   0.1920   0.1833   0.1603   0.1779
## Detection Rate         0.2795   0.1802   0.1670   0.1551   0.1760
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9845   0.9610   0.9510   0.9786   0.9901
```

```r
set.seed(42)
rfModel_0p5_tree300<- train(classe ~., data=subtraining, method="rf", ntree = 300, trControl=control)
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
confusionMatrix(subtesting$classe, predict(rfModel_0p5_tree300, subtesting[,-53]))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2783    6    1    0    0
##          B   16 1875    7    0    0
##          C    0   18 1684    9    0
##          D    1    2   41 1563    1
##          E    0    3    6    7 1787
## 
## Overall Statistics
##                                         
##                Accuracy : 0.988         
##                  95% CI : (0.9856, 0.99)
##     No Information Rate : 0.2854        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.9848        
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9939   0.9848   0.9684   0.9899   0.9994
## Specificity            0.9990   0.9971   0.9967   0.9945   0.9980
## Pos Pred Value         0.9975   0.9879   0.9842   0.9720   0.9911
## Neg Pred Value         0.9976   0.9963   0.9932   0.9980   0.9999
## Prevalence             0.2854   0.1941   0.1773   0.1610   0.1823
## Detection Rate         0.2837   0.1911   0.1717   0.1593   0.1822
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9965   0.9909   0.9825   0.9922   0.9987
```

As was already mentioned by the instructor, Random Forest is the best classifier as it achieved the highest accuracy as shown in the plot below:


```r
comparisons <- resamples(list(RF=rfModel_0p5_tree300, GBM=gbmModel, SVM=svmModel))
summary(comparisons)
```

```
## 
## Call:
## summary.resamples(object = comparisons)
## 
## Models: RF, GBM, SVM 
## Number of resamples: 30 
## 
## Accuracy 
##       Min. 1st Qu. Median   Mean 3rd Qu.   Max. NA's
## RF  0.9806  0.9850 0.9888 0.9881  0.9908 0.9939    0
## GBM 0.9480  0.9544 0.9593 0.9588  0.9638 0.9715    0
## SVM 0.8930  0.9001 0.9052 0.9077  0.9144 0.9327    0
## 
## Kappa 
##       Min. 1st Qu. Median   Mean 3rd Qu.   Max. NA's
## RF  0.9755  0.9810 0.9858 0.9849  0.9884 0.9923    0
## GBM 0.9341  0.9423 0.9485 0.9479  0.9542 0.9639    0
## SVM 0.8644  0.8733 0.8798 0.8830  0.8915 0.9147    0
```

```r
bwplot(comparisons)
```

![](PracticalMachineLearning_files/figure-html/compare-1.png)<!-- -->

Some time was invested in exploring the parameter space to the 'rf' model but was not pursued beyon the maximum number of trees set at 300 because the resulting accurarcy was not improving much beyond that value.

# Predict the outcome of the test data set

As mentioned above, the test data set was prepared in the same manner as the training set. Using our best model (Random Forest) we get the following outcome:


```r
results <- predict(rfModel_0p5_tree300, reducedTesting[,-53])
```
# Acknowledgement:
We used the data collected by the Human Activity Research project:
Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6.

#####################
## Project Summary ##
#####################

## Sentiment analysis for mobile phones - iphone

###########################
## Set working directory ##
###########################


setwd("C:\\Users\\Maja\\Documents\\Ubiqum\\Module 4\\Part 3\\")


###############
# Load packages
###############


library(dplyr)
library(tidyr)
library(caret)
library(plotly)
library(lubridate)
library(rgdal)
library(parallel)
library(doParallel)
library(ggplot2)
library(ggmap)
library(corrplot)
library(e1071)
library(kknn)
library(C50)
#library(dummies)




#########################
## Parallel processing ##
#########################

##create cluster with desired number of cores 
cl <- makeCluster(detectCores() - 1)

##register cluster
registerDoParallel(cl)

##confirm how many cores are assigned
#getDoParWorkers() 



#################
## Import data ##
#################

iphone_m <- read.csv("iphone_smallmatrix_labeled_8d.csv",
                     header = TRUE,
                     stringsAsFactors = FALSE)

large_iphone_m <- read.csv("iphoneLargeMatrix.csv",
                           header = TRUE,
                           stringsAsFactors = FALSE) 


######################
## Investigate data ##
######################

head(iphone_m, n = 10)
tail(iphone_m, n = 10)
summary(iphone_m)
str(iphone_m)
attributes(iphone_m)
sapply(iphone_m, class)
summary(iphone_m[,521:529])

plot_ly(iphone_m, x= ~iphone_m$iphonesentiment, type='histogram')


#########################
## Preprocess the data ##
#########################

## check for NAs
length(which(is.na(iphone_m))) ## no NAs

## convert from int to factor
iphone_m$iphonesentiment <- as.factor(iphone_m$iphonesentiment)
large_iphone_m$iphonesentiment <- as.factor(large_iphone_m$iphonesentiment)

## correlation
options(max.print=1000000) #to enable printing all values for corr

iphone_corr <- cor(iphone_m)
corrplot(iphone_corr, method = "number", number.cex = .6, order = "FPC")

p95 <- cor.mtest(iphone_m ,conf.level = .95)
corrplot(iphone_corr, p.mat = p95$p, sig.level = 0.4, insig = "blank")

## drop id column in large matrix
large_iphone_m$id <- NULL

#######################
## Feature selection ##
#######################

## ZERO VARIANCE
## near zero variance with metrics
nzvMetrics <- nearZeroVar(iphone_m, saveMetrics = TRUE)
nzvMetrics

## near zero variance without metrics
nzv <- nearZeroVar(iphone_m, saveMetrics = FALSE)
nzv

## create a new data set and remove near zero variance features
iphone_NZV <- iphone_m[,-nzv]
str(iphone_NZV)



## RFE - automated feature selection
## Let's sample the data before using RFE
set.seed(1234)
iphoneSample <- iphone_recode[sample(1:nrow(iphone_recode), 1000, replace=FALSE),]

## Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

## Use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfeResults <- rfe(iphoneSample[,1:58], 
                  iphoneSample$iphonesentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)

## Get results
rfeResults

## Plot results
plot(rfeResults, type=c("g", "o"))

# create new data set with rfe recommended features
iphoneRFE <- iphone_recode[,predictors(rfeResults)]
large_iphone_RFE <- large_iphone_m[,predictors(rfeResults)]

# add the dependent variable to iphoneRFE
iphoneRFE$iphonesentiment <- iphone_recode$iphonesentiment
large_iphone_RFE$iphonesentiment <- large_iphone_m$iphonesentiment

# review outcome
str(iphoneRFE)
str(large_iphone_RFE)



################
### Modeling ###
################

#######################################
### Train control and preprocessing ###
#######################################

#######################
## 1. OUT OF THE BOX ##
#######################

#preprocessing - normalization
preP_train <- preProcess(iphone_m[,1:(ncol(iphone_m) - 1)], method=c("center", "scale"))

# cross validation
fitControl <- trainControl(method = "repeatedcv",
              number = 10,
              repeats = 3,
              preP_train)

# set seed for reproducibility
set.seed(1234)

#split train test
inTraining_iphone <- createDataPartition(iphone_recode$iphonesentiment, p = .7, list = FALSE)
trainSet_iphone <- iphone_recode[inTraining_iphone,]
testSet_iphone <- iphone_recode[-inTraining_iphone,]



#########################
## SVM                 ##
## Accuracy     Kappa  ##
## 0.7136247 0.4174391 ##
#########################

#SVM model training
SVMFit_iphone <- train(iphonesentiment~., 
                       data = trainSet_iphone, 
                       method = "svmLinear2",
                       trControl=fitControl)

#predicting
predictionSVM_iphone <- predict(SVMFit_iphone, testSet_iphone)

#performace measurment
postResample(predictionSVM_iphone, testSet_iphone$iphonesentiment)

#plot predicted vs actual
plot(predictionSVM_iphone, testSet_iphone$iphonesentiment)


#########################
## KNN                 ##
## Accuracy     Kappa  ##
## 0.3269923 0.1595332 ##
#########################


#KNN model training
KNNFit_iphone <- train(iphonesentiment~., 
                       data = trainSet_iphone, 
                       method = "kknn",
                       trControl=fitControl)

#predicting
predictionKNN_iphone <- predict(KNNFit_iphone, testSet_iphone)

#performace measurment
postResample(predictionKNN_iphone, testSet_iphone$iphonesentiment)

#plot predicted vs actual
plot(predictionKNN_iphone, testSet_iphone$iphonesentiment)

#########################
## RANDOM FOREST       ##
## Accuracy     Kappa  ##
## 0.7691517 0.5541610 ##
#########################

RFFit_iphone <- train(iphonesentiment~., 
                      data = trainSet_iphone, 
                      method = "rf",
                      trControl=fitControl)

#predicting
predictionRF_iphone <- predict(RFFit_iphone, testSet_iphone)

#performace measurment
postResample(predictionRF_iphone, testSet_iphone$iphonesentiment)

#create a confusion matrix from random forest predictions 
cmRF <- confusionMatrix(predictionRF_iphone, testSet_iphone$phonesentiment) 
cmRF

#plot predicted vs actual
plot(predictionRF_iphone, testSet_iphone$iphonesentiment)


#########################
## C5.0                ##
## Accuracy     Kappa  ##
## 0.7637532 0.5411485 ##
#########################

C5Fit_iphone <- train(iphonesentiment~., 
                      data = trainSet_iphone, 
                      method = "C5.0",
                      trControl=fitControl)

#predicting
predictionC5_iphone <- predict(C5Fit_iphone, testSet_iphone)

#performace measurment
postResample(predictionC5_iphone, testSet_iphone$iphonesentiment)

# Create a confusion matrix from random forest predictions 
cmC5 <- confusionMatrix(predictionC5_iphone, testSet_iphone$iphonesentiment) 
cmC5

#plot predicted vs actual
plot(predictionC5_iphone, testSet_iphone$iphonesentiment)

#########################
## Feature engineering ##
#########################

##RECODING
# create a new dataset that will be used for recoding sentiment
iphone_recode <- iphone_m
# recode sentiment to combine factor levels 0 & 1 and 4 & 5
iphone_recode$iphonesentiment <- recode(iphone_recode$iphonesentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 
# inspect results
summary(iphone_recode)
str(iphone_recode)
# make iphonesentiment a factor
iphone_recode$iphonesentiment <- as.factor(iphone_recode$iphonesentiment)

##PCA
# data = training and testing from iphone_m (no feature selection) 
# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to .95
preprocessParams <- preProcess(trainSet_iphone[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams)

# use predict to apply pca parameters, create training, exclude dependant
training_iphone.pca <- predict(preprocessParams, trainSet_iphone[,-59])

# add the dependent to training
training_iphone.pca$iphonesentiment <- trainSet_iphone$iphonesentiment

# use predict to apply pca parameters, create testing, exclude dependant
testing_iphone.pca <- predict(preprocessParams, testSet_iphone[,-59])

# add the dependent to training
testing_iphone.pca$iphonesentiment <- testSet_iphone$iphonesentiment

# inspect results
str(training_iphone.pca)
str(testing_iphone.pca)


#########################
## 2. NZV RF           ##
## Accuracy     Kappa  ##
## 0.7547558 0.5158788 ##
#########################

#preprocessing - normalization
preP_train_NZV <- preProcess(iphone_NZV[,1:(ncol(iphone_NZV) - 1)], method=c("center", "scale"))

# cross validation
fitControl_NZV <- trainControl(method = "repeatedcv",
                               number = 10,
                               repeats = 3,
                               preP_train_NZV)

# set seed for reproducibility
set.seed(1234)

#split train test
inTraining_iphone_NZV <- createDataPartition(iphone_NZV$iphonesentiment, p = .7, list = FALSE)
trainSet_iphone_NZV <- iphone_NZV[inTraining_iphone_NZV,]
testSet_iphone_NZV <- iphone_NZV[-inTraining_iphone_NZV,]

RFFit_iphone_NZV <- train(iphonesentiment~., 
                          data = trainSet_iphone_NZV, 
                          method = "rf",
                          trControl=fitControl_NZV)

#predicting
predictionRF_iphone_NZV <- predict(RFFit_iphone_NZV, testSet_iphone_NZV)

#performace measurment
postResample(predictionRF_iphone_NZV, testSet_iphone_NZV$iphonesentiment)

# Create a confusion matrix from random forest predictions 
cmNZV <- confusionMatrix(predictionRF_iphone_NZV, testSet_iphone_NZV$iphonesentiment) 
cmNZV

#plot predicted vs actual
plot(predictionRF_iphone_NZV, testSet_iphone_NZV$iphonesentiment)



#########################
## 3.                  ##
## RFE                 ##
## Accuracy     Kappa  ##
## 0.7699229 0.5564168 ##
## RFE with recoding   ##
## Accuracy     Kappa  ##
## 0.8462725 0.6191514 ##
#########################

#preprocessing - normalization
preP_train_RFE <- preProcess(iphoneRFE[,1:(ncol(iphoneRFE) - 1)], method=c("center", "scale"))

# cross validation
fitControl_RFE <- trainControl(method = "repeatedcv",
                               number = 10,
                               repeats = 3,
                               preP_train_RFE)


#split train test
inTraining_iphone_RFE <- createDataPartition(iphoneRFE$iphonesentiment, p = .7, list = FALSE)
trainSet_iphone_RFE <- iphoneRFE[inTraining_iphone_RFE,]
testSet_iphone_RFE <- iphoneRFE[-inTraining_iphone_RFE,]

RFFit_iphone_RFE <- train(iphonesentiment~., 
                          data = trainSet_iphone_RFE, 
                          method = "rf",
                          trControl=fitControl_RFE)

#predicting
predictionRF_iphone_RFE <- predict(RFFit_iphone_RFE, testSet_iphone_RFE)

#performace measurment
postResample(predictionRF_iphone_RFE, testSet_iphone_RFE$iphonesentiment)

# Create a confusion matrix from random forest predictions 
cmRFE <- confusionMatrix(predictionRF_iphone_RFE, testSet_iphone_RFE$iphonesentiment) 
cmRFE

#plot predicted vs actual
plot(predictionRF_iphone_RFE, testSet_iphone_RFE$iphonesentiment)

##predict on large matrix
predictionRF_iphone_RFE_large_m <- predict(RFFit_iphone_RFE, large_iphone_RFE)
large_iphone_RFE$iphonesentiment <- predictionRF_iphone_RFE_large_m

##write to csv
write.csv(large_iphone_RFE, "iphoneLargeMatrix_RFE_recoding.csv")

plot_ly(large_iphone_RFE, x= ~large_iphone_RFE$iphonesentiment, type='histogram')


#########################
## 4. Recoding         ##
## Accuracy     Kappa  ##
## 0.8434447 0.6080863 ##
#########################


#preprocessing - normalization
preP_train_recode <- preProcess(iphone_recode[,1:(ncol(iphone_recode) - 1)], method=c("center", "scale"))

# cross validation
fitControl_recode <- trainControl(method = "repeatedcv",
                                  number = 10,
                                  repeats = 3,
                                  preP_train_recode)

#split train test
inTraining_iphone_recode <- createDataPartition(iphone_recode$iphonesentiment, p = .7, list = FALSE)
trainSet_iphone_recode <- iphone_recode[inTraining_iphone_recode,]
testSet_iphone_recode <- iphone_recode[-inTraining_iphone_recode,]

RFFit_iphone_recode <- train(iphonesentiment~., 
                             data = trainSet_iphone_recode, 
                             method = "rf",
                             trControl=fitControl_recode)

#predicting
predictionRF_iphone_recode <- predict(RFFit_iphone_recode, testSet_iphone_recode)

#performace measurment
postResample(predictionRF_iphone_recode, testSet_iphone_recode$iphonesentiment)

# Create a confusion matrix from random forest predictions 
cmRecode <- confusionMatrix(predictionRF_iphone_recode, testSet_iphone_recode$iphonesentiment) 
cmRecode

#plot predicted vs actual
plot(predictionRF_iphone_recode, testSet_iphone_recode$iphonesentiment)



#########################
## 5.                  ##   
## PCA                 ##
## Accuracy     Kappa  ##
## 0.7568123 0.5322864 ##
## Accuracy     Kappa  ##
## 0.8398458 0.6029579 ##
#########################


# cross validation
fitControl_PCA <- trainControl(method = "repeatedcv",
                               number = 10,
                               repeats = 3)

RFFit_iphone_PCA <- train(iphonesentiment~., 
                          data = training_iphone.pca, 
                          method = "rf",
                          trControl=fitControl_PCA)

#predicting
predictionRF_iphone_PCA <- predict(RFFit_iphone_PCA, testing_iphone.pca)

#performace measurment
postResample(predictionRF_iphone_PCA, testing_iphone.pca$iphonesentiment)

# Create a confusion matrix from random forest predictions 
cmPCA <- confusionMatrix(predictionRF_iphone_PCA, testing_iphone.pca$iphonesentiment) 
cmPCA

#plot predicted vs actual
plot(predictionRF_iphone_PCA, testing_iphone.pca$iphonesentiment)

#after finishing tasks stop your cluster. 
stopCluster(cl)



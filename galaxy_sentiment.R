#####################
## Project Summary ##
#####################

## Sentiment analysis for mobile phones - samsung

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

galaxy_m <- read.csv("galaxy_smallmatrix_labeled_9d.csv",
                     header = TRUE,
                     stringsAsFactors = FALSE)

large_galaxy_m <- read.csv("galaxyLargeMatrix.csv",
                           header = TRUE,
                           stringsAsFactors = FALSE) 


######################
## Investigate data ##
######################

head(galaxy_m, n = 10)
tail(galaxy_m, n = 10)
summary(galaxy_m)
str(galaxy_m)
attributes(galaxy_m)
sapply(galaxy_m, class)
summary(galaxy_m[,521:529])

plot_ly(galaxy_m, x= ~galaxy_m$galaxysentiment, type='histogram')


#########################
## Preprocess the data ##
#########################

## check for NAs
length(which(is.na(galaxy_m))) ## no NAs

## correlation
options(max.print=1000000) #to enable printing all values for corr

galaxy_corr <- cor(galaxy_m)
corrplot(galaxy_corr, method = "number", number.cex = .6, order = "FPC")

p95 <- cor.mtest(galaxy_m ,conf.level = .95)
corrplot(galaxy_corr, p.mat = p95$p, sig.level = 0.4, insig = "blank")

## convert from int to factor
galaxy_m$galaxysentiment <- as.factor(galaxy_m$galaxysentiment)

large_galaxy_m$id <- NULL

#######################
## Feature selection ##
#######################

## ZERO VARIANCE
## near zero variance with metrics
nzvMetrics <- nearZeroVar(galaxy_recode, saveMetrics = TRUE)
nzvMetrics

## near zero variance without metrics
nzv <- nearZeroVar(galaxy_recode, saveMetrics = FALSE)
nzv

## create a new data set and remove near zero variance features
galaxy_NZV <- galaxy_recode[,-nzv]
str(galaxy_NZV)



## RFE - automated feature selection
## Let's sample the data before using RFE
set.seed(1234)
galaxySample <- galaxy_recode[sample(1:nrow(galaxy_recode), 1000, replace=FALSE),]

## Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

## Use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfeResults <- rfe(galaxySample[,1:58], 
                  galaxySample$galaxysentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)

## Get results
rfeResults

## Plot results
plot(rfeResults, type=c("g", "o"))

# create new data set with rfe recommended features
galaxy_RFE <- galaxy_recode[,predictors(rfeResults)]

# add the dependent variable to iphoneRFE
galaxy_RFE$galaxysentiment <- galaxy_recode$galaxysentiment

# review outcome
str(galaxy_RFE)

# from num to factor
galaxy_RFE$galaxysentiment <- as.factor(galaxy_RFE$galaxysentiment)


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
preP_train <- preProcess(galaxy_recode[,1:(ncol(galaxy_recode) - 1)], method=c("center", "scale"))

# cross validation
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 3,
                           preP_train)

# set seed for reproducibility
set.seed(1234)

#split train test
inTraining_galaxy <- createDataPartition(galaxy_recode$galaxysentiment, p = .7, list = FALSE)
trainSet_galaxy <- galaxy_recode[inTraining_galaxy,]
testSet_galaxy <- galaxy_recode[-inTraining_galaxy,]



#########################
## SVM                 ##
## Accuracy     Kappa  ##
## 0.7202273 0.4079727 ##
## Accuracy     Kappa  ##
## 0.7820248 0.4016532 ##
#########################

#SVM model training
SVMFit_galaxy <- train(galaxysentiment~., 
                       data = trainSet_galaxy, 
                       method = "svmLinear2",
                       trControl=fitControl)

#predicting
predictionSVM_galaxy <- predict(SVMFit_galaxy, testSet_galaxy)

#performace measurment
postResample(predictionSVM_galaxy, testSet_galaxy$galaxysentiment)

#plot predicted vs actual
plot(predictionSVM_galaxy, testSet_galaxy$galaxysentiment)


#########################
## KNN                 ##
## Accuracy     Kappa  ##
## 0.7520021 0.5106467 ##
##  Accuracy     Kappa ##
## 0.8303202 0.5583613 ##
#########################


#KNN model training
KNNFit_galaxy <- train(galaxysentiment~., 
                       data = trainSet_galaxy, 
                       method = "kknn",
                       trControl=fitControl)

#predicting
predictionKNN_galaxy <- predict(KNNFit_galaxy, testSet_galaxy)

#performace measurment
postResample(predictionKNN_galaxy, testSet_galaxy$galaxysentiment)

#plot predicted vs actual
plot(predictionKNN_galaxy, testSet_galaxy$galaxysentiment)

#########################
## RANDOM FOREST       ##
## Accuracy     Kappa  ##
## 0.7638853 0.5304795 ##
## Accuracy     Kappa  ##
## 0.8365186 0.5730535 ##
#########################

RFFit_galaxy <- train(galaxysentiment~., 
                      data = trainSet_galaxy, 
                      method = "rf",
                      trControl=fitControl)

#predicting
predictionRF_galaxy <- predict(RFFit_galaxy, testSet_galaxy)

#performace measurment
postResample(predictionRF_galaxy, testSet_galaxy$galaxysentiment)

#create a confusion matrix from random forest predictions 
cmRF <- confusionMatrix(predictionRF_galaxy, testSet_galaxy$galaxysentiment) 
cmRF

#plot predicted vs actual
plot(predictionRF_galaxy, testSet_galaxy$galaxysentiment)


#########################
## C5.0                ##
## Accuracy     Kappa  ##
## 0.7687936 0.5371038 ##
## Accuracy     Kappa  ##
## 0.8370351 0.5729237 ##
#########################

C5Fit_galaxy <- train(galaxysentiment~., 
                      data = trainSet_galaxy, 
                      method = "C5.0",
                      trControl=fitControl)

#predicting
predictionC5_galaxy <- predict(C5Fit_galaxy, testSet_galaxy)

#performace measurment
postResample(predictionC5_galaxy, testSet_galaxy$galaxysentiment)

# Create a confusion matrix from random forest predictions 
cmC5 <- confusionMatrix(predictionC5_galaxy, testSet_galaxy$galaxysentiment) 
cmC5

#plot predicted vs actual
plot(predictionC5_galaxy, testSet_galaxy$galaxysentiment)

#########################
## Feature engineering ##
#########################

##RECODING
# create a new dataset that will be used for recoding sentiment
galaxy_recode <- galaxy_m
# recode sentiment to combine factor levels 0 & 1 and 4 & 5
galaxy_recode$galaxysentiment <- recode(galaxy_recode$galaxysentiment, '0' = 1, '1' = 1, '2' = 2, '3' = 3, '4' = 4, '5' = 4) 
# inspect results
summary(galaxy_recode)
str(galaxy_recode)
# make galaxysentiment a factor
galaxy_recode$galaxysentiment <- as.factor(galaxy_recode$galaxysentiment)

##PCA
# data = training and testing from iphone_m (no feature selection) 
# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to .95
preprocessParams <- preProcess(trainSet_galaxy[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams)

# use predict to apply pca parameters, create training, exclude dependant
training_galaxy.pca <- predict(preprocessParams, trainSet_galaxy[,-59])

# add the dependent to training
training_galaxy.pca$galaxysentiment <- trainSet_galaxy$galaxysentiment

# use predict to apply pca parameters, create testing, exclude dependant
testing_galaxy.pca <- predict(preprocessParams, testSet_galaxy[,-59])

# add the dependent to training
testing_galaxy.pca$galaxysentiment <- testSet_galaxy$galaxysentiment

# inspect results
str(training_galaxy.pca)
str(testing_galaxy.pca)


#########################
## 2. NZV RF           ##
## Accuracy     Kappa  ##
## 0.7556187 0.5030115 ##
##  NZV with recoding  ##
##  Accuracy     Kappa ##
## 0.8236054 0.5249330 ##
#########################

#preprocessing - normalization
preP_train_NZV_galaxy <- preProcess(galaxy_NZV[,1:(ncol(galaxy_NZV) - 1)], method=c("center", "scale"))

#cross validation
fitControl_NZV_galaxy <- trainControl(method = "repeatedcv",
                                      number = 10,
                                      repeats = 3,
                                      preP_train_NZV_galaxy)

# set seed for reproducibility
set.seed(1234)

#split train test
inTraining_galaxy_NZV <- createDataPartition(galaxy_NZV$galaxysentiment, p = .7, list = FALSE)
trainSet_galaxy_NZV <- galaxy_NZV[inTraining_galaxy_NZV,]
testSet_galaxy_NZV <- galaxy_NZV[-inTraining_galaxy_NZV,]

RFFit_galaxy_NZV <- train(galaxysentiment~., 
                          data = trainSet_galaxy_NZV, 
                          method = "rf",
                          trControl=fitControl_NZV_galaxy)

#predicting
predictionRF_galaxy_NZV <- predict(RFFit_galaxy_NZV, testSet_galaxy_NZV)

#performace measurment
postResample(predictionRF_galaxy_NZV, testSet_galaxy_NZV$galaxysentiment)

#create a confusion matrix from random forest predictions 
cmNZV <- confusionMatrix(predictionRF_galaxy_NZV, testSet_galaxy_NZV$galaxysentiment) 
cmNZV

#plot predicted vs actual
plot(predictionRF_galaxy_NZV, testSet_galaxy_NZV$galaxysentiment)



#########################
## 3.                  ##
## RFE                 ##
## Accuracy     Kappa  ##
## 0.7912684 0.5859996 ## 
## RFE with recoding   ##
## Accuracy     Kappa  ##
## 0.8403926 0.5927913 ##
## RFE recoding C5     ##
## Accuracy     Kappa  ##
## 0.8388430 0.5828736 ##
#########################

#preprocessing - normalization
preP_train_RFE_galaxy <- preProcess(galaxy_RFE[,1:(ncol(galaxy_RFE) - 1)], method=c("center", "scale"))

# cross validation
fitControl_RFE_galaxy <- trainControl(method = "repeatedcv",
                                      number = 10,
                                      repeats = 3,
                                      preP_train_RFE_galaxy)


#split train test
inTraining_galaxy_RFE <- createDataPartition(galaxy_RFE$galaxysentiment, p = .7, list = FALSE)
trainSet_galaxy_RFE <- galaxy_RFE[inTraining_galaxy_RFE,]
testSet_galaxy_RFE <- galaxy_RFE[-inTraining_galaxy_RFE,]

RFFit_galaxy_RFE <- train(galaxysentiment~., 
                          data = trainSet_galaxy_RFE, 
                          method = "C5.0",
                          trControl=fitControl_RFE_galaxy)

#predicting
predictionRF_galaxy_RFE <- predict(RFFit_galaxy_RFE, testSet_galaxy_RFE)

#performace measurment
postResample(predictionRF_galaxy_RFE, testSet_galaxy_RFE$galaxysentiment)

#create a confusion matrix from random forest predictions 
cmRFE <- confusionMatrix(predictionRF_galaxy_RFE, testSet_galaxy_RFE$galaxysentiment) 
cmRFE

#plot predicted vs actual
plot(predictionRF_galaxy_RFE, testSet_galaxy_RFE$galaxysentiment)


#########################
## 4. Recoding         ##
## Accuracy     Kappa  ##
## 0.8354855 0.5698788 ##
#########################


#preprocessing - normalization
preP_train_recode_galaxy <- preProcess(galaxy_recode[,1:(ncol(galaxy_recode) - 1)], method=c("center", "scale"))

# cross validation
fitControl_recode_galaxy <- trainControl(method = "repeatedcv",
                                         number = 10,
                                         repeats = 3,
                                         preP_train_recode_galaxy)

#split train test
inTraining_galaxy_recode <- createDataPartition(galaxy_recode$galaxysentiment, p = .7, list = FALSE)
trainSet_galaxy_recode <- galaxy_recode[inTraining_galaxy_recode,]
testSet_galaxy_recode <- galaxy_recode[-inTraining_galaxy_recode,]

RFFit_galaxy_recode <- train(galaxysentiment~., 
                             data = trainSet_galaxy_recode, 
                             method = "rf",
                             trControl=fitControl_recode_galaxy)

#predicting
predictionRF_galaxy_recode <- predict(RFFit_galaxy_recode, testSet_galaxy_recode)

#performace measurment
postResample(predictionRF_galaxy_recode, testSet_galaxy_recode$galaxysentiment)

#create a confusion matrix from random forest predictions 
cmRecode <- confusionMatrix(predictionRF_galaxy_recode, testSet_galaxy_recode$galaxysentiment) 
cmRecode

#plot predicted vs actual
plot(predictionRF_galaxy_recode, testSet_galaxy_recode$galaxysentiment)



#########################
## 5. PCA              ##
## Accuracy     Kappa  ##
## 0.7699229 0.5564168 ##
## Accuracy     Kappa  ##
## 0.8305785 0.5565175 ##
#########################


# cross validation
fitControl_PCA <- trainControl(method = "repeatedcv",
                               number = 10,
                               repeats = 3)

RFFit_galaxy_PCA <- train(galaxysentiment~., 
                          data = training_galaxy.pca, 
                          method = "rf",
                          trControl=fitControl_PCA)

#predicting
predictionRF_galaxy_PCA <- predict(RFFit_galaxy_PCA, testing_galaxy.pca)

#performace measurment
postResample(predictionRF_galaxy_PCA, testing_galaxy.pca$galaxysentiment)

#create a confusion matrix from random forest predictions 
cmPCA <- confusionMatrix(predictionRF_galaxy_PCA, testing_galaxy.pca$galaxysentiment) 
cmPCA

#plot predicted vs actual
plot(predictionRF_galaxy_PCA, testing_galaxy.pca$galaxysentiment)

#after finishing tasks stop your cluster. 
stopCluster(cl)

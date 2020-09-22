# The aim of this script is to build a Random Forest Model for the purpose 
# of prediction of purchasing behaviour based on the Online Shoppers Purchasing Intention Dataset

# This script is built under the Professional Certificate in Data Science by HarvardX program 
# as an additional to the Capstone Project on a dataset of student's choice. 

# The Script Takes around 6 minutes to run

# Install all needed for the project libraries if they are not found on the computer
if(!require(rmarkdown)) install.packages("rmarkdown")
if(!require(plyr)) install.packages("plyr")
if(!require(dplyr)) install.packages("dplyr")
if(!require(caret)) install.packages("caret")
if(!require(tinytex)) install.packages("tinytex")
if(!require(funModeling)) install.packages("funModeling")
if(!require(stringr)) install.packages("stringr")
if(!require(lubridate)) install.packages("lubridate")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(ggthemes)) install.packages("ggthemes")
if(!require(mltools)) install.packages("mltools")
if(!require(Hmisc)) install.packages("Hmisc")
if(!require(data.table)) install.packages("data.table")
if(!require(InformationValue)) install.packages("InformationValue")
if(!require(glmnet)) install.packages("glmnet")
if(!require(elasticnet)) install.packages("elasticnet")
if(!require(rpart)) install.packages("rpart")
if(!require(randomForest)) install.packages("randomForest")
if(!require(broom)) install.packages("broom")

# Import libraries
library(rmarkdown)
library(plyr)
library(dplyr)
library(caret)
library(tinytex)
library(funModeling)
library(stringr)
library(lubridate)
library(ggplot2)
library(ggthemes)
library(mltools)
library(Hmisc)
library(data.table)
library(InformationValue)
library(glmnet)
library(elasticnet)
library(rpart)
library(randomForest)
library(broom)

### --- Download Data --- ###

#The dataset used is downloaded from UCI Machine Learning Repository and was created and 
# uploaded by Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018).

#The raw dataset is available on the following link: https://archive.ics.uci.edu/ml/machine-learning-databases/00468/

# Load automatically the dataset for analysis from UCI Machine Learning Repository
Online_Purchasing_Dataset = fread("https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv")

# Transform data types and rename columns
Online_Purchasing_Dataset_Transformed = Online_Purchasing_Dataset %>%
  # transform categorical variables to factors - this is going to be helpful for the visualization and modelling
  mutate(SpecialDay = as.factor(SpecialDay)
         , OperatingSystems = as.factor(OperatingSystems)
         , Browser = as.factor(Browser)
         , Region = as.factor(Region)
         , TrafficType = as.factor(TrafficType)
         , Month = as.factor(Month)
         , VisitorType = as.factor(VisitorType)
         , Weekend = as.factor(as.character(Weekend))
         , Revenue = as.character(Revenue)
         , Revenue = ifelse(Revenue == "TRUE",1,0)
         , Revenue = as.factor(Revenue)
  ) %>%
  # rename the Revenue column to Purchase to make the name more intuitive
  plyr::rename(c("Revenue" = "Purchase")) %>%
  # Create one character column with the Purchase Yes/No - it is more suitable for some of the visualizations and modelling
  mutate(Purchase_Yes_No = ifelse(as.character(Purchase) == "0","No","Yes"))

### --- Split Data in Train/Test in ration 70%/30% --- ###
set.seed(1) # set seed for reproducability
# createDataPartition function is used - it makes the split stratified
train.index <- createDataPartition(Online_Purchasing_Dataset_Transformed$Purchase, p = .7, list = FALSE) # 70% of the data set for train ser
Train_Set <- Online_Purchasing_Dataset_Transformed[ train.index,] # Define train set based on indices
Test_Set  <- Online_Purchasing_Dataset_Transformed[-train.index,] # Define test set based on remaining indices

### --- Build Random Forest Model

# Tune random forest model with different mtry values
Random_Forest_Tune = train(
  Purchase_Yes_No ~ ., 
  data = Train_Set %>%
    select(-Purchase) %>%
    # use the make.names funtion on factor predictor vars to avoid errors in model build
    mutate(SpecialDay = make.names(SpecialDay)
           , Month = make.names(Month)
           , OperatingSystems = make.names(OperatingSystems)
           , Browser = make.names(Browser)
           , Region = make.names(Region)
           , TrafficType = make.names(TrafficType)
           , VisitorType = make.names(VisitorType)
           , Weekend = make.names(Weekend)
    )
  , method = "rf",
  trControl = trainControl(method = "cv"
                           , number = 3
                           , classProbs=TRUE
                           , summaryFunction = twoClassSummary),
  # set different values for mtry parameter to test
  tuneGrid = expand.grid(mtry = c(7,9,11,13,15)),
  metric = "ROC"
)

#extract best mtry param
Random_Forest_Best_Tune = Random_Forest_Tune$bestTune

set.seed(1) # set seed for reproducibility

# create random foret model with the best mtry parameter.
Random_Forest_Model = randomForest(Purchase ~ .
                                   , data = Train_Set %>% 
                                     select(-Purchase_Yes_No)
                                   , mtry = Random_Forest_Best_Tune %>% pull(mtry)
)

# Make Predictions on Test Set
Random_Forest_Predictions_Test_Set = predict(Random_Forest_Model, newdata = Test_Set %>%
                                               select(-Purchase_Yes_No) 
                                             , type = "prob"
) %>%
  as.data.frame() %>%
  pull(`1`)

# Calculate AUC on Test Set
Random_Forest_Test_Set_AUC = round(auc_roc(Random_Forest_Predictions_Test_Set, ordered(Test_Set$Purchase)),3)

# Estimate Sensitivity Specificity on Test Set
confmat <- confusionMatrix(Test_Set$Purchase, Random_Forest_Predictions_Test_Set)

# Calculate Sensitivity, Specificity and Precision on Test set
Random_Forest_Sensitivity_Test_Set = round(confmat[2,]$`1`/sum(confmat[2,]),2)*100
Random_Forest_Specificity_Test_Set = round(confmat[1,]$`0`/sum(confmat[1,]),2)*100
Random_Forest_Precision_Test_Set = round(confmat[2,]$`1`/sum(confmat$`1`),2)*100

# Print results to the console
print(paste0("The AUC of the Random Forest on test set is: ",Random_Forest_Test_Set_AUC))
print(paste0("The Sensitivity of the Random Forest on test set is: ",Random_Forest_Sensitivity_Test_Set,"%"))
print(paste0("The Specificity of the Random Forest on test set is: ",Random_Forest_Specificity_Test_Set,"%"))
print(paste0("The Precision of the Random Forest on test set is: ",Random_Forest_Precision_Test_Set,"%"))













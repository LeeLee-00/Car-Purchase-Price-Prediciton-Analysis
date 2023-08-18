######################################
# title: Project - Predicting Car Purchase Price
#author: Lee Noel
#Original Data Source:" https://www.kaggle.com/datasets/yashpaloswal/ann-car-sales-price-prediction 

###############################################

# Loading libraries.
library(readr) #For reading CSVs from URL
library(leaps) #for subset selection
library(gains) #for gains and lift chart
library(forecast) #for accuracy measures
library(rpart) #for fitting the classification tree
library(rpart.plot) #for plotting the fitted classification tree
library(randomForest) #for random forest plot
library(caret) #for confusion matrix
library(class) #for knn function
library(ggplot2) #for decision tree visualization

#######################################################################################################

setwd("E:/PreAnalytics/Car Sales Prediction Analysis")

#Loading/Preprocessing/and cleaning dataset.

# In order to be able to see both the original dataset and the changes we are making, we are creating two datasets with the same data and using one to edit.

# "Obtained from where you download the Kaggle Dataset link above"

 file_path <- "E:/PreAnalytics/Car Sales Prediction Analysis/car_purchasing.csv"

#car_purchasing_original <- read.csv("E:/PreAnalytics/Car Sales Prediction Analysis/car_purchasing.csv") 
car_purchasing_original <- read.csv(file_path) # reading in the dataset.
car_purchasing <- car_purchasing_original
# Cleaning the dataset
car_purchasing <- car_purchasing[, -c(1:3)] # removing name, email, country.
colnames(car_purchasing) <- c("Gender", "Age", "Salary", "Debt", "Net_Worth", "Car_Price") # rename column for clarity.
car_purchasing$Age <- round(car_purchasing$Age) # remove decimals from age for simplicity.
car_purchasing$Gender <- factor(car_purchasing$Gender, levels = c(0, 1), 
                                labels = c("Female", "Male")) # change gender to categorical variable & labeled for better understanding.

# Dividing training and testing dataset.
set.seed(2) # set the seed so we product the same results every time.
train.index <- sample(c(1:dim(car_purchasing)[1]), dim(car_purchasing)[1]*0.60) # Split the data by 60%.
train.df <- car_purchasing[train.index, ] # create a dataset with 60% of the data.
valid.df <- car_purchasing[-train.index, ] # create a dataset with 40% of the data.


#######################################################################################################
 ## MODEL 1  - Linear Regression  ##

# Create a linear regression model.
car_purchasing.lm <- lm(Car_Price ~., data = train.df) # create a linear regression model.
summary(car_purchasing.lm) # summary of linear regression model.
search <- regsubsets(Car_Price ~., data = train.df, nbest = 1, nvmax = dim(train.df)[2], method = "exhaustive") # stepwise regression with exhaustive search.
sum <- summary(search) # summary of stepwise regression. 


#sum$which # displays true/false table. Comment 
models <-  order(sum$adjr2, decreasing = T)[1:3] # returns the top 3 models.
models # prints the top 3 models.
car_purchasing.lm.step.back <- step(car_purchasing.lm, direction = "backward") # stepwise backwards regression.
#Produces 2 models
#  Stepback model # 1 is  AIC = 3312.26 (Car_Price ~ Gender + Age + Salary + Debt + Net_Worth)
#  Stepback model #2 is  AIC = 3310.45 (Car_Price ~ Age + Salary + Debt + Net_Worth)

summary(car_purchasing.lm.step.back) # summary of stepwise backwards regression model.
# RES. St. Error = 246.9   MultiRSq = 0.9995    Adj.R.Sq= 0.9995
# Predicting
car_purchasing.lm.step.pred <- predict(car_purchasing.lm.step.back, valid.df) # prediction of the validation dataset vs the actual value.
accuracy(car_purchasing.lm.step.pred, valid.df$Car_Price) # calculating the accuracy of the predictive value. 
# ME = 17.751  RMSE = 235.28 1   MAE = 199.569      MPE - 0.051   MAPE = .4667

#Creating Prediction Frame
prediction_frame <- data.frame(cbind(actual_values = valid.df$Car_Price, predicted_values = car_purchasing.lm.step.pred ))
# prediction_frame (COMMENTED OUT FOR COMPLINING)


# Fit lm model using 10-fold CV: model
modelTrain <- train(
  Car_Price ~ .,
  train.df,
  method = "lm",
  trControl = trainControl(
    method = "cv",
    number = 10,
    verboseIter = FALSE
  )
)

modelTest <- train(
  Car_Price ~ .,
  valid.df,
  method = "lm",
  trControl = trainControl(
    method = "cv",
    number = 10,
    verboseIter = FALSE
  )
)

# Print model to console
modelTrain
# Train Results: RMSE = 249.01  Rsq = 0.9995  MAE = 216.597
modelTest
# Test Results: RMSE = 236.56  Rsq = 0.9995  MAE = 200.45

#End of Linear Model 

####################################################################
## MODEL 2  - KNN Model ##

# Set Seed to 1 
set.seed(1)

#Setting the  Median Value to Zero in order to normalize the data
car_purchasing$MEDV <- 0
#Adding med value for car price greater than 44,210
car_purchasing$MEDV <- ifelse(car_purchasing$Car_Price >= mean(car_purchasing_original$car.purchase.amount), car_purchasing$MEDV <- 1, 0)

car_purchasing <- car_purchasing[,-6]
car_purchasing$Gender <- as.numeric(car_purchasing$Gender)

train.df <- car_purchasing[train.index, ] # create a dataset with 60% of the data.
valid.df <- car_purchasing[-train.index, ] # create a dataset with 40% of the data.

#Creating the Train and Vaild Index
train.index <- sample(row.names(car_purchasing), 0.6*dim(car_purchasing)[1])
valid.index <- setdiff(row.names(car_purchasing), train.index)

#Creating Levleing factors for MEDV
valid.df$MEDV <-as.factor(valid.df$MEDV) 
# Converting Gender to Numeric 
train.df$Gender <- as.numeric(train.df$Gender)
valid.df$Gender <- as.numeric(valid.df$Gender)

#Creating a new data frame 
new.df <- data.frame(
  Gender = 1,
  Age = 46,
  Salary = 52482,
  Debt = 8569,
  Net_Worth = 543085
)

#Normalizing the Data to prediction for KNN
train.norm.df <- train.df
valid.norm.df <- valid.df
norm.df <- car_purchasing
norm.values <- preProcess(train.df[, 1:5], method=c("center", "scale"))
#Creating Factors for new columns of Median Value 
train.norm.df$MEDV <- as.factor(train.norm.df$MEDV)
valid.norm.df$MEDV <- as.factor(valid.norm.df$MEDV)
# Reconverting Gender back to Numeric
norm.df$Gender <- as.numeric(norm.df$Gender)  

train.norm.df[, 1:5] <- predict(norm.values, train.df[, 1:5])
valid.norm.df[, 1:5] <- predict(norm.values, valid.df[, 1:5])
norm.df[, 1:5] <- predict(norm.values, car_purchasing[, 1:5])
new.norm.df <- predict(norm.values, new.df)

#Predicting Knn with K of 5 
pred_nn <- knn(train = train.norm.df[, 1:5], test = new.norm.df,
               cl = train.norm.df[, 6], k = 5)
pred_nn

accuracy.df <- data.frame(k = seq(1, 5, 1), accuracy = rep(0, 5))

knn.pred <- knn(train.norm.df[, 1:5], valid.norm.df[, 1:5],
                cl = train.norm.df[, 6], k = 5)
# knn.pred  (COMMENTED OUT FOR COMPLINING)

# Creating Confusion Matrix 
accuracy.df[3, 2] <- confusionMatrix(data = knn.pred, reference =  valid.norm.df$MEDV)$overall[1]
accuracy.df

# Running Loop to test for K value with highest Accuracy Rating 
for(i in 1:300) {
  knn.pred <- knn(train.norm.df[, 1:5], valid.norm.df[, 1:5],
                  cl = train.norm.df[, 6], k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn.pred, valid.norm.df[, 6])$overall[1]
}

accuracy.df[which.max(accuracy.df$accuracy), ]
#Prediction Accuracy Highest Model is K = 29 with accuracy at 92.5%

# 10 fold Validation for Knn Model
model2Train <- train(
  MEDV~ .,
  train.norm.df,
  method = "knn",
  trControl = trainControl(
    method = "cv",
    number = 10,
    verboseIter = FALSE
  )
)

model2Test <- train(
  MEDV~ .,
  valid.norm.df,
  method = "knn",
  trControl = trainControl(
    method = "cv",
    number = 10,
    verboseIter = FALSE
  )
)

model2Train
# KNN 10 fold Train Results - Best Model K = 9 with accuracy at 89.99%
model2Test
#KNN 10 fold Test Results - Best Model K = 9 with accuracy at 89.0%

# End of KNN Model 

###############################################################################################################
#Model 3 - Decision Tree ##

# Resetting MEDV from Numeric back to factor
train.df$MEDV <- as.factor(train.df$MEDV)
valid.df$MEDV <- as.factor(valid.df$MEDV)
# Create a decision tree model.
tr <- rpart(MEDV ~., data = train.df, method = "class", cp=0.001, minbucket = 10, maxdepth = 7) # fit the classification tree using all variables as predictors for car price, our outcome variable. We tried several variations of minbucket and maxdepth and found this to be the best level of complexity.
options(scipen = 2) # decimal places proceeded by 0.
printcp(tr) # print cp of the newly fitted model.
pfit<- prune(tr, cp = tr$cptable[which.min(tr$cptable[,"xerror"]),"CP"]) # converts a decision node of minimal significance to a terminal node to reduce overfitting. Chooses cp by selecting the lowest level of the tree with the minimum "xerror".
prp(tr, box.palette = "PuOr", legend.x = NA) # prints the decision tree.
t(t(names(car_purchasing))) # prints the name of the columns of the dataset.
t(t(tr$variable.importance)) # variables displayed by importance.
tr # set of rules printed.

#Calculating Accuracy 
t.pred <- predict(tr, valid.df,type='class')
accuracy <- confusionMatrix(t.pred,as.factor(valid.df$MEDV))
accuracy
#Prediction Accuracy is 83.5%

# 10 fold Validation for Tree Model
model3Train <- train(
  MEDV ~ .,
  train.df,
  method = "rpart",
  trControl = trainControl(
    method = "cv",
    number = 10,
    verboseIter = FALSE
  )
)

model3Test <- train(
  MEDV ~ .,
  valid.df,
  method = "rpart",
  trControl = trainControl(
    method = "cv",
    number = 10,
    verboseIter = FALSE
  )
)

# Print model to console
model3Train
#Tree 10 fold Train Results -  Best Model CP = 0.0402  with accuracy at 80.96% 
model3Test
#Tree 10 fold Test Results - Best Model CP = 0.0306 with accuracy at 81.40%

########################################################################################

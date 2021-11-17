#loading Dataset
train <- read.csv("train.csv", stringsAsFactors = F, na.strings = c("NA", ""))
test <- read.csv("test.csv", stringsAsFactors = F, na.strings = c("NA", ""))

#data structure
str(train)
str(test)

#summary
summary(train[, c(3,6,10)])
summary(as.factor(train$Sex))
summary(as.factor(train$Embarked))

#missing data analysis
train_test <- train
test_test <- test
train_test$Set <- "train"
test_test$Set <- "test"
test_test$Survived <- NA
all_data <- rbind(train_test, test_test)
sapply(all_data, function (x) {sum(is.na(x))})

#exploratory data analysis
counts_Sex <- table(train$Survived, train$Sex)
barplot(counts_Sex, main="Number of Survivors by Sex", xlab="Sex", ylab="Number of Passengers", beside=TRUE, col=c("blue", "green"), names.arg=c("Female", "Male"), cex.names=0.9, cex.axis=0.9, ylim=c(0, 500), axis.lty=1)
legend("topleft", legend=c("Died", "Survived"), fill=c("blue", "green"), cex=0.8, box.lty=0)

#analysis of survival by class
prop.table(table(all_data$Pclass))
prop.table(table(train$Pclass, train$Survived), 1)


counts_Class <- table(train$Survived, train$Pclass)
barplot(counts_Class, main="Number of Survivors by Class", xlab="Class", ylab="Number of Passengers", beside=TRUE, col=c("blue", "green"), names.arg=c("First", "Second", "Third"), cex.names=0.9, cex.axis=0.9, ylim=c(0, 400), axis.lty=1)
legend("topleft", legend=c("Died", "Survived"), fill=c("blue", "green"), cex=0.8, box.lty=0)



#Decision Tree

library(rpart)
library(rpart.plot)
library(caret)
library(PRROC)

tree <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, 
              data=train, method = "class", control=rpart.control(minsplit = 10))

rpart.plot(tree)

#use this tree to predict the outcome of the test data
my_prediction <- predict(tree, test, type="class")

actual_values <- read.csv("gender_submission.csv")
test$Survived <- actual_values$Survived
confusionMatrix(table(test$Survived, my_prediction))

#roc curve
PRROC_obj <- roc.curve(scores.class0 = my_prediction, weights.class0=test$Survived ,
                       curve=TRUE)
plot(PRROC_obj)


#Random Forest
library(randomForest)

set.seed(123)

train_rand <- train[-c(1,4,9,11)]
train_rand <- na.omit(train_rand)
rand_forest <- randomForest(factor(Survived)~Age + Fare + Sex+ Pclass+Parch+SibSp+Embarked, data=train_rand)

rand_forest

#getting important features
#Get importance
varImpPlot(rand_forest)


pred_rf_valid <- predict(rand_forest,newdata = test)
pred_rf_valid[is.na(pred_rf_valid)] <- 0

confusionMatrix(table(test$Survived, pred_rf_valid))

#roc curve
PRROC_obj <- roc.curve(scores.class0 = pred_rf_valid, weights.class0=test$Survived ,
                       curve=TRUE)
plot(PRROC_obj)


#ANN

#Using neural networks 
library(nnet)
equation<-"Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked"
survival<-as.formula(equation)
nnet.fit <- nnet(formula=survival, data=train_rand, size=2) 

Survived <- predict(nnet.fit, newdata=test)

#Model Summary
summary(nnet.fit)
library(gamlss.add)
plot(nnet.fit, struct = nnet.fit$n)

#predictions 
Survived <- ifelse(Survived>0.5, 1,0)
Survived[is.na(Survived)] <- 0

confusionMatrix(table(test$Survived, Survived))

#roc curve
PRROC_obj <- roc.curve(scores.class0 = Survived, weights.class0=test$Survived ,
                       curve=TRUE)
plot(PRROC_obj)


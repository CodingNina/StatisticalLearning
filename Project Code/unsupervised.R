install.packages('Rcpp')
library(Rcpp)

#Loading data
train_set <- read.csv("train.csv",na.strings=c('NA',''), stringsAsFactors=F)
test  <- read.csv("test.csv",na.strings=c('NA',''), stringsAsFactors=F)

head(train_set)

mytrain_set <- train_set

#Extract Cabin Num from Cabin.
mytrain_set$CabinNum <- sapply(mytrain_set$Cabin,function(x) strsplit(x,'[A-Z]')[[1]][2])
mytrain_set$CabinNum <- as.numeric(mytrain_set$CabinNum)

#Transforming data class
mytrain_set$Sex <- as.factor(mytrain_set$Sex)
mytrain_set$Sex <- as.integer(mytrain_set$Sex)
mytrain_set$Embarked <- as.factor(mytrain_set$Embarked)
mytrain_set$Embarked <- as.integer(mytrain_set$Embarked)

mytrain_set <- mytrain_set[, c("Survived", "Pclass", "Sex", "Age", "SibSp",
                       "Parch", "Ticket", "Fare", "Embarked", "CabinNum")]
head(mytrain_set)

sapply(mytrain_set, function(x) {sum(is.na(x))})


#Getting correlation coefficient matrix
library(psych)
pca.train_set <- cor(mytrain_set[complete.cases(mytrain_set), 
                         c("Pclass", "Sex", "Age", "SibSp", "Parch",
                           "Fare", "Embarked", "CabinNum")])
                           
#Drawing screeplot and parallel analysis
fa.parallel(pca.train_set, fa = "pc", n.iter = 100, n.obs = sum(complete.cases(mytrain_set)),
            show.legend = FALSE, main = "Scree plot with parallel analysis")


#PCA
rc <- principal(pca.train_set, nfactors = 2, rotate = "varimax", scores = TRUE)
fa.diagram(rc)
rc

#imputing missing data
mean(!complete.cases(mytrain_set))

library(VIM)
library(Rcpp)
ms <- aggr(mytrain_set, prop = FALSE, numbers = TRUE)
summary(ms)

x <- as.data.frame(abs(is.na(mytrain_set)))
y <- x[which(sapply(x, sd) > 0)]
cor(y)

ms.train_set <- mytrain_set[, c("Survived", "Pclass", "Sex", 
                        "Age", "SibSp", "Parch", "Fare", "Embarked", "CabinNum")]

x <- as.data.frame(abs(is.na(ms.train_set)))
y <- x[which(sapply(x, sd) > 0)]

cor(ms.train_set, y, use = "pairwise.complete.obs")

#Imputing by Multiple Imputation

library(mice)
install.packages('Rcpp')
library(Rcpp)
imp <- mice(mytrain_set, seed = 1234)
completeData <- complete(imp,2)

#recheck to see if any missing values left
sum(is.na(completeData))

#Clustering
library("factoextra")
library(NbClust)

#scaling the data
completeData$Ticket <- as.numeric(as.factor(completeData$Ticket))
df <- scale(completeData)
head(df)

#distance meaures
distance <- get_dist(df)

#Finding the best number of clusters



nbc <- NbClust(data = df, min.nc = 2, max.nc = 8, method = "kmeans")

#applying k-means model
k2 <- kmeans(df, centers = 2, nstart = 25)
k2

#visualizing clusters
fviz_cluster(k2, data = df)

#model performance
#confusion matrix of our predictions

library(caret)
completeData$Survived <- ifelse(completeData$Survived == 1, 1,2)
completeData$Survived <- as.factor(as.numeric(completeData$Survived))
confusionMatrix(as.factor(k2$cluster), completeData$Survived)

#roc curve
library(PRROC)
PRROC_obj <- roc.curve(scores.class0 = as.numeric(k2$cluster), weights.class0=as.numeric(completeData$Survived) ,
                       curve=TRUE)
plot(PRROC_obj)

#DECISION TREES
# This is a deciison tree w/ CART Algorithm. I use cross validation to get a better estimate of the error. 
# I do not impute the missing values - I do not conduct listwise deletion either.
# I do not standardize or normalize the dataset.
# I keep the dataset as is. (No preprocessing)

install.packages("caret") #Download te package for splitting the dataset into training and testing
install.packages("ggplot2") #Download this package in case I want to plot some features
install.packages("rattle") #Download this to plot the decision tree
install.packages("RColorBrewer") #Download this to make the tree look fancy
install.packages("lattice") #Care package requires it
install.packages("mice") # data imputation
install.packages("dummies") # binary transformation of categorical variables
install.packages("class") # KNN package
install.packages("e1071")
install.packages("rpart")
install.packages("clusterGenomics")
install.packages("mclust")
install.packages("EMCluster")
install.packages("fastICA")

#Activate the packages
library(caret)
library(ggplot2)
library(rattle)
library(RColorBrewer)
library(lattice)
library(mice)
library(dummies)
library(class)
library(e1071)
library(rpart)
library(clusterGenomics)
library(mclust)
library(EMCluster)
library(fastICA)

set.seed(4000) #Let's set the seed for reproducibility


df=read.csv("C:/Users/Aric.johnson/Desktop/CKD.csv", header = TRUE, dec=",", na.strings = c(""," ","NA")) #read the csv file and save it to the dataframe object df


#for some reason, R reads the variales below as factors. We have to convert them to numericals
numericVectors<-c("bu", "sc", "sod", "pot", "hemo", "rbcc", "pcv")
numericVectorSize<-length(numericVectors)
for (n in 1:numericVectorSize){
  x=numericVectors[n]
  df[,x]<-as.numeric(levels(df[,x]))[df[,x]] #set some factor variables as numeric variables 
}

#for some reason, R reads the variales below as numbers. We have to convert them to factors.
factorVectors<-c("al", "su")
factorVectorSize<-length(factorVectors)
for (n in 1:factorVectorSize){
  x=factorVectors[n]
  df[, x]<-as.factor(df[,x])
}

summary(df) # let's check the dataset
str(df) #let's see if all the variables are orrectly identified as factors and numbers.

# It is time to partition the data set into Training and Test sets
trainSet<-createDataPartition(y=df$class, p=0.7, list=FALSE)
Train<-df[trainSet,] #subset the training set
Test<-df[-trainSet,] # subset the test set

#let's check if the class labels are imbalanced. 
#According to the literature, if +/- ratio is above 0.5, the class os balanced.
balanceIndicatorTrain<-length(which(Train[, dim(Train)[2]]=="notckd"))/length(which(Train[, dim(Train)[2]]=="ckd"))
balanceIndicatorTest<-length(which(Test[, dim(Test)[2]]=="notckd"))/length(which(Test[, dim(Test)[2]]=="ckd"))
balanceIndicatorTrain<-round(balanceIndicatorTrain, digits = 3)
balanceIndicatorTest<-round(balanceIndicatorTest, digits = 3)


#Let's do cross validation to get a better estimate of model accuracy
n<-nrow(Train)
K<-10 #number of folds - we do 10-fold cross validation
foldSize<-n%/%K
al<-runif(n) #we label eah row in the training set with random numebrs so we can assign them in one of the folds
ra<-rank(al) # we want to figure out which row in the training set the first random number is associated with
blocks<-(ra-1)%/%foldSize+1 # create the folds
blocks<-as.factor(blocks) #make them factors
print(summary(blocks))
errorCV <- vector(mode="numeric", length=K)
cp<-seq(0, 0.1, 0.01) # Learning rate values for tuning
minSplit<-seq(0,30,2) ## of hidden layers for tuning
meanError<-data.frame()
for(i in 1:length(cp)) {
  for(j in 1:length(minSplit)) {
    for(k in 1:K) {
      
      cvModel<-rpart(class ~. , method = "class", data=Train[blocks!=k,], 
               control = rpart.control(minsplit=minSplit[j], cp = cp[i]))
      
      pred<-predict(cvModel, newdata=Train[blocks==k,], type="class")
      cmCV<-table(pred, Train[blocks==k,"class"])
      errorK<- 1- (cmCV[1,1] + cmCV[2,2])/sum(cmCV)
      errorCV[k]<-errorK
    }
    averageError<-mean(errorCV)
    meanError[i,j]<-averageError
  }
}
which(meanError==min(meanError)) #cp=0.03 and minsplit 6 gives the bes result

TreeModel<-rpart(class ~. , method = "class", data=Train, 
                  control = rpart.control(minsplit=0, cp = 0))

plot(TreeModel) #plots the tree
text(TreeModel, pretty=0) # brings teh text onto the plot
fancyRpartPlot(TreeModel, uniform=TRUE, sub="Classification Tree") # makes the plot look nicer

pTreeModel<-rpart(class ~. , method = "class", data=Train, 
               control = rpart.control(minsplit=6, cp = 0.03))

pred<-predict(pTreeModel, newdata=Test, type="class")
cmCV<-table(pred, Test[,"class"])
Finalerror<- 1- (cmCV[1,1] + cmCV[2,2])/sum(cmCV)
plot(pTreeModel) #plots the tree
text(pTreeModel, pretty=0) # brings teh text onto the plot
fancyRpartPlot(pTreeModel, uniform=TRUE, sub="Pruned Classification Tree") # makes the plot look nicer

data1=data.frame(df$hemo,df$al, df$sc,df$class)

classColumnID1<-dim(data1)[2]


#Let's do ppm imputation and binary transformation for categorical data
imputedDF<- mice(data1,m=1,maxit=50,meth='pmm',seed=500) #imputation with pmm
data1<-complete(imputedDF) #replacing the NA's
preObj <- preProcess(data1[, -classColumnID1], method=c("center", "scale")) #standardization
data2 <- predict(preObj, data1[, -classColumnID1]) #standardization
data2<-dummy.data.frame(data2, names = NULL, omit.constants=TRUE, dummy.classes = "factor") #categorical to binary variables
data2$class=data1$df.class #add the class to the df1
data1=data2 #save df1 to df
summary(data1) #get the summary of the dataframe to make sure things are ok!

#let's get rid of some enironmental variables
a=ls()
a=a[a != "data1"]
remove(list = a)

pTreeModel<-rpart(class ~. , method = "class", data=data1, 
                  control = rpart.control(minsplit=30, cp = 0.1))
pred<-predict(pTreeModel, newdata=data1, type="class")
cmCV<-table(pred, data1[,"class"])
Finalerror<- 1- (cmCV[1,1] + cmCV[2,2])/sum(cmCV)
plot(pTreeModel) #plots the tree
text(pTreeModel, pretty=0) # brings teh text onto the plot
fancyRpartPlot(pTreeModel, uniform=TRUE, sub="Pruned Classification Tree")
data1=data1[, c("df.hemo", "df.sc", "class")]
df=data1
a=ls()
a=a[a != "df"]
remove(list = a)

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


set.seed(4000) #Let's set the seed for reproducibility
df=read.csv("C:/Users/Aric.johnson/Desktop/CKD.csv", header = TRUE, dec=",", na.strings = c(""," ","NA")) #read the csv file and save it to the dataframe object df

#for some reason, R reads the variales below as factors. have to convert them to numericals
numericVectors<-c("bu", "sc", "sod", "pot", "hemo", "rbcc", "pcv") 
for (n in 1:7){
  x=numericVectors[n]
  df[,x]<-as.numeric(levels(df[,x]))[df[,x]] #set some factor variables as numeric variables 
}

#for some reason, R reads the variales below as number so have to convert them to factors.
factorVectors<-c("al", "su")
for (n in 1:2){
  x=factorVectors[n]
  df[, x]<-as.factor(df[,x])
}
remove(factorVectors)
remove(numericVectors)
remove(n)
remove(x)

summary(df) # let's check the dataset
str(df) #let's see if all the variables are orrectly identified as factors and numbers.
classColumnID1<-dim(df)[2]


#Let's do ppm imputation and binary transformation for categorical data
imputedDF<- mice(df,m=1,maxit=50,meth='pmm',seed=500) #imputation with pmm
df<-complete(imputedDF) #replacing the NA's
preObj <- preProcess(df[, -classColumnID1], method=c("center", "scale")) #standardization
df1 <- predict(preObj, df[, -classColumnID1]) #standardization
df1<-dummy.data.frame(df1, names = NULL, omit.constants=TRUE, dummy.classes = "factor") #categorical to binary variables
df1["class"]=df$class #add the class to the df1
df=df1 #save df1 to df
summary(df) #get the summary of the dataframe to make sure things are ok!

#let's get rid of some enironmental variables
remove(imputedDF)
remove(preObj)
remove(classColumnID1)
remove(df1)
df1=df[,-ncol(df)]
pca=princomp(df1)
varianceAccounted=round(pca$sdev^2/sum(pca$sdev^2), 3)*100
CumVar=cumsum(varianceAccounted)
biplot(pca)
a=with(pca, unclass(loadings))
aload <- abs(a)
impVar=sweep(aload, 2, colSums(aload), "/")
plot(pca$sdev, pch=20, xlab="# of principal components", ylab="Eigen Values", col="blue")
lines(pca$sdev, col="blue")
principle=with(pca, unclass(scores))
principle=as.data.frame(principle)
prins=principle[, 1:10]
prins$class=df[, ncol(df)]
df=prins

a=ls()
a=a[a != "df"]
remove(list = a)

trainSet<-createDataPartition(y=df$class, p=0.7, list=FALSE) #create the testing training partitions
Train<-df[trainSet,] #subset the training set
Test<-df[-trainSet,] #subset the testing set
classColumnID2<-dim(df)[2]
remove(trainSet) #let's get rid of an enironmental variable

labeledTrain=Train
Train=Train[,-classColumnID2]

labeledTest=Test




model1 = Mclust(Train, G=2, control = emControl())
BIC=model1$BIC
prob=round(model1$z, 3)
resultTrain=model1$classification

actualTrain=as.character(labeledTrain[, classColumnID2])
for (i in 1:length(actualTrain)) {
  if (actualTrain[i]=="ckd") {
    actualTrain[i]=1
  } else {
    actualTrain[i]=2
  }
}
actualTrain=as.numeric(actualTrain)

ii=0
for (i in 1:length(actualTrain)) {
  if (actualTrain[i]==resultTrain[i]){
    ii=ii+1
  }
}
accuracyTrain=ii/length(actualTrain)

model2 = Mclust(Test, G=2, control = emControl())
BIC=model2$BIC
probTest=round(model2$z, 3)
resultTest=model2$classification

actualTest=as.character(labeledTest[, classColumnID2])
for (i in 1:length(actualTest)) {
  if (actualTest[i]=="ckd") {
    actualTest[i]=1
  } else {
    actualTest[i]=2
  }
}
actualTest=as.numeric(actualTest)

aa=0
for (i in 1:length(actualTest)) {
  if (actualTest[i]==resultTest[i]){
    aa=aa+1
  }
}
accuracyTest=aa/length(actualTest)

Test=Test[,-classColumnID2]
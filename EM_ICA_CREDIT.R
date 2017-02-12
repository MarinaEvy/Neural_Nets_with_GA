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
df=read.csv("C:/Users/mjohnson6/Desktop/Evy_Stuff/EVY_R_CODE/Credit.csv", 
            header = TRUE, dec=",", 
            na.strings = c(""," ","NA"))

numericVectors<-c("A2", "A3", "A8") 
for (n in 1:3){
  x=numericVectors[n]
  df[,x]<-as.numeric(levels(df[,x]))[df[,x]] #set some factor variables as numeric variables 
}

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

df1=df[,-ncol(df)]

ICA=fastICA(df1, 2, alg.typ = c("parallel","deflation"),
            fun = c("logcosh","exp"), alpha = 1.0, method = c("R","C"),
            row.norm = FALSE, maxit = 200, tol = 1e-04, verbose = FALSE,
            w.init = NULL)
plot(ICA$S, main = "ICA components")
plot(ICA$X)
ICAs=ICA$S
ICAs=as.data.frame(ICAs)
ICAs$class=df[, ncol(df)]
df=ICAs

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



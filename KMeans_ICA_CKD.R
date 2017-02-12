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
library(fastICA)


set.seed(4000) #Let's set the seed for reproducibility

df=read.csv("C:/Users/mjohnson6/Desktop/Evy_Stuff/EVY_R_CODE/CKD.csv", 
            header = TRUE, dec=",", 
            na.strings = c(""," ","NA"))

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
df=df[,-ncol(df)]

ICA=fastICA(df, 2, alg.typ = c("parallel","deflation"),
            fun = c("logcosh","exp"), alpha = 1.0, method = c("R","C"),
            row.norm = FALSE, maxit = 200, tol = 1e-04, verbose = FALSE,
            w.init = NULL)
plot(ICA$S, main = "ICA components")
plot(ICA$X)
ICAs=ICA$S
ICAs=as.data.frame(ICAs)
colnames(ICAs)=c("ICA1","ICA2")
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
Test=Test[,-classColumnID2]
wss=0

for (i in 1:15) {
  kMeans <- kmeans(Train, centers=i)
  wss[i] <- kMeans$tot.withinss
} 
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")


#picking the best k
X=data.matrix(Train, rownames.force = NA)
res <- gap(X,B=10, cl.method="kmeans", nstart=10)
par(mfrow=c(2,1))
plot(1:length(res$W), res$W, type="b", col="red", xlab="k", 
     ylab="within-cluster dispersion")
plot(1:length(res$gap), res$gap, type="b", ylim=c(min(res$gap-res$sk),
                                                  max(res$gap+res$sk)), 
     pch=19, col="green", ylab="GAP", xlab="k")
points(1:length(res$sk), res$gap+res$sk, cex=0.7, pch=8, col="green")
points(1:length(res$sk), res$gap-res$sk, cex=0.7, pch=8, col="green")
segments(x0=1:length(res$sk), y0=res$gap-res$sk, y1=res$gap+res$sk)



#Training
kMeansTrain <-kmeans(Train, centers=2)
resultTrain=kMeansTrain$cluster
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


#testing
kMeansTest <-kmeans(Test, centers=2)
resultTest=kMeansTest$cluster
actualTest=as.character(labeledTest[, classColumnID2])
for (i in 1:length(actualTest)) {
  if (actualTest[i]=="ckd") {
    actualTest[i]=2
  } else {
    actualTest[i]=1
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

labeledTest=Test
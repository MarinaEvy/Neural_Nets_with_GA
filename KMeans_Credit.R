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
df=read.csv("C:/Users/Aric.johnson/Desktop/Credit.csv", header = TRUE, dec=",", 
            na.strings = c(""," ","NA"))#read the csv file and save it to the dataframe object df

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

remove(classColumnID1)
remove(numericVectors)
remove(n)
remove(x)
remove(imputedDF)
remove(preObj)



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
  if (actualTrain[i]=="No") {
    actualTrain[i]=2
  } else {
    actualTrain[i]=1
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
  if (actualTest[i]=="No") {
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







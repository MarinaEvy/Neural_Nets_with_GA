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
install.packages("devtools")
install_github("chappers/randomProjection")


#Activate the packages
library(randomProjection)
library(devtools)
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

#let's get rid of some enironmental variables
remove(imputedDF)
remove(preObj)
remove(classColumnID1)
remove(df1)
df1=df[,-ncol(df)]
rp=RandomProjection(df1,n_features=2, eps=0.1)
dim(rp$RP)
randomp=with(rp, unclass(RP))
randomp=as.data.frame(randomp)
randomp$class=df[, ncol(df)]
df=randomp
a=ls()
a=a[a != "df"]
remove(list = a)

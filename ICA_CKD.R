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



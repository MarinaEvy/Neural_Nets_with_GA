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

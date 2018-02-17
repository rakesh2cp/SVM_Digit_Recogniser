############################ SVM Digit Recogniser Case Study ##############################
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Model Building 
#  4.1 Linear kernel
#  4.2 Polygot Kernel
#  4.3 RBF Kernel
# 5. Hyperparameter tuning and cross validation
# 6.Testing of Model against Test Dataset

############################################################################################
# Author  : Rakesh Pattanaik.                                                              #   
############################################################################################


# 1. Business Understanding: -----

# The MNIST which is a very popular database of handwritten digits. 
# It has a training set of 60,000 examples, and a test set of 10,000 examples. 
# The digits size have been normalized and centered in a fixed-size image.
# The objective of this case study is to identify rectangular pixel displays 
# as one of the digits from 0-9 in English.


# 2. Data Understanding: -----
# The MNIST which is a very popular database of handwritten digits. 
# It has a training set of 60,000 examples, and a test set of 10,000 examples. 
# Number of Instances: 60000
# Number of Attributes: 785 (784 continuous, 1 nominal class label)
# The MNIST dataset has total 785 coulmns out of which the first column
# represents the digit (e.g. 0,1,2 etc.) and rest other columns are flattened
# value of a 28*28 matrix.

# 3. Data Preparation: -----

# Loading Neccessary libraries
# The following libraries are required to run this case study.
# Please install them before running the code.
library(kernlab)
library(readr)
library(caret)
library(caTools)
library(dplyr)
library(readr)
library(ggplot2)
library(gridExtra)
library(sparklyr)
#spark_install(version = "2.1.0")
sc <- spark_connect(master = "local")
spark_disconnect(sc)



# Loading Data
# Set the working directory
# Please change the working directory as per your requirement and where
# All the data files (The provided data file in the assignment) is present.
setwd("path-to-MNIST-data-set")

#Please make sure all the data files are listed in the directory
list.files()

# Load the MNIST train data set
digit_rec_data <- read.delim("mnist_train.csv",sep = ",", stringsAsFactors = F,header = F)

# Checking structure of loaded dataset.
str(digit_rec_data)

# Checking for na values. There are no na values in dataset.
sapply(digit_rec_data, function(x) sum(is.na(x)))

# Checking dimension of data set
# data set has 60000 observations and 785 variables
dim(digit_rec_data)

# Printing First Few Rows
head(digit_rec_data)

# DATA UNDERSTANDING

# The MNIST dataset has total 785 coulmns out of which the first column
# represents the digit (e.g. 0,1,2 etc.) and rest other columns are flattened
# value of a 28*28 matrix. Hence we will convert each row in our dataset to 
# get a 28*28 matrix out of it.
# Creating a 28*28 matrix with pixel respective values.

# EDA and data sampling with plotting

# Matrix convertion of loaded dataset.
m = matrix(unlist(digit_rec_data[10,-1]), nrow = 28, byrow = TRUE)

# Function to rotate the matrix vertically
rotate <- function(x) t(apply(x, 2, rev)) 

# Plot the digits for first few lines of data set. 
par(mfrow=c(2,3))
lapply(1:6, 
       function(x) image(
         rotate(matrix(unlist(digit_rec_data[x,-1]),nrow = 28, byrow = TRUE)),
         col=grey.colors(255),
         xlab=digit_rec_data[x,1]
       )
)
# We can clearly see the plots are exactly same as the 
# letters present in the dataset.

# Converting our data set to factor
digit_rec_data$V1<-factor(digit_rec_data$V1)

# Splitting the data between train and test
set.seed(100)
indices = sample.split(digit_rec_data$V1, SplitRatio = 0.05)
digit_rec_data_n = digit_rec_data[indices,]

# The above training data set has 60,000 rows and it is expected 
# that it will need lot of computation time since svm is  Compute-intensive.
# Further dividing the sample to training and testing data set 
# in order to reduce computation time.
indices_1 = sample.split(digit_rec_data_n$V1, SplitRatio = 0.7)
digit_rec_ds = digit_rec_data_n[indices_1,]
digit_rec_ds_test = digit_rec_data_n[!(indices_1),]

# Reading the provided training DS
# we will use this data set (digit_ds_test data frame ) for testing our data set.
digit_ds_test <- digit_rec_data <- read.delim("mnist_test.csv",sep = ",", stringsAsFactors = F,header = F)

# It is observed that the test dataset has different column names.
# Hence Renaming Column names of test data set as per the training dataset.
column_names <- colnames(digit_rec_data)
colnames(digit_ds_test) <- column_names

# 4. Model Buiilding----

# Using Linear Kernel
digit_rec_mod_lin <- ksvm(V1~ ., data = digit_rec_ds, scale = FALSE, kernel = "vanilladot")
# Evaluating the Model
digit_rec_eval_lin<- predict(digit_rec_mod_lin, digit_rec_ds_test)

#confusion matrix for Linear Kernel
confusionMatrix(digit_rec_eval_lin,digit_rec_ds_test$V1)

# It is observed that the Accuracy for Lenear Kernel is 90.44%

# Polydot Kernel
digit_rec_mod_poly <- ksvm(V1~ ., data = digit_rec_ds, scale = FALSE, kernel = "polydot")
# Evaluating the Model
digit_rec_eval_poly<- predict(digit_rec_mod_poly, digit_rec_ds_test)

#confusion matrix for Polygot Kernel
confusionMatrix(digit_rec_eval_poly,digit_rec_ds_test$V1)
# Accuracy for Polygot Kernel is also 90.44%, there is no much difference from our linear model.

# Using RBF Kernel 
digit_rec_mod_rbf <- ksvm(V1~ ., data = digit_rec_ds, scale = FALSE, kernel = "rbfdot")
# Evaluating the Model
digit_rec_eval_rbf<- predict(digit_rec_mod_rbf, digit_rec_ds_test)

#confusion matrix for RBF Kernel
confusionMatrix(digit_rec_eval_rbf,digit_rec_ds_test$V1)
# Accuracy for RBF kernel is 93.56%

# 5. Hyperparameter Tuning and CV (Cross Validation) ####
# We will use the train function from caret package to perform Cross Validation. 
# traincontrol function Controls the computational nuances of the train function.
# i.e. method =  CV means  Cross Validation.  
# number = 2 implies Number of folds in CV.

trainControl <- trainControl(method="cv", number=5)

# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.
metric <- "Accuracy"

# Expand.grid functions takes set of hyperparameters, that we shall pass to our model.
set.seed(7)

# Creating grid for model Tuning with svmRadial Method. 
grid <- expand.grid(.sigma=c(0.025, 0.05), .C=c(0.1,0.5) )

# Creating grid for model Tuning with svmPoly Method.
grid2 <- expand.grid(degree=c(1,2,3,4), scale=c(0.1,0.2,0.3,0.4,0.5), C=c(1,2,3,4,5) )

# Model Tuning with svmRadial Method
# The below lines of code may take 10-12 minutes to complete.
fit.svm_rad <- train(V1~., data=digit_rec_ds, method="svmRadial", metric=metric, 
                     tuneGrid=grid, trControl=trainControl)

# svmRadial method  Model Statistics
print(fit.svm_rad)
#Plotting the Model
plot(fit.svm_rad)
# Observations from our svmRadial on the training data set are as below 
# Accuracy is 11.23%, sigma = 0.05 and  C = 0.1.

# Since the svmrdial method didn't give good accuracy we will tune our model with
# svmPoly method.

# Model Tuning with svmPoly Method
# The below lines of code may take 20-25 minutes to complete, since we have
fit.svm <- train(V1~., data=digit_rec_ds, method="svmPoly", metric=metric, 
                 tuneGrid=grid2, trControl=trainControl)

# Model statistics
print(fit.svm)
#Plotting the Model
plot(fit.svm)
# For svmPoly Accuracy is 92.42% when degree = 2, scale = 0.1 and  C = 1.

# 6. Testing of Model against Supplied Test Dataset -----

# Testing with Linear Kernel
digit_rec_eval_test_lin<- predict(digit_rec_mod_lin, digit_ds_test)

# Confusion Matrix for Linear Model
confusionMatrix(digit_rec_eval_test_lin,digit_ds_test$V1)
# Accuracy for Linear Model on test data set is  90.64% which is very close to accuracy of 
# training data set.

# Testing for Polydot Kernel
digit_rec_eval_test_poly<- predict(digit_rec_mod_poly, digit_ds_test)

# Confusion Matrix for Polygot Model
confusionMatrix(digit_rec_eval_test_poly,digit_ds_test$V1)
# Accuracy for Polygot Model on Test Data set is 90.64% 

# Testing for RBF Kernel
# The below line of code may take 4-5 minutes to complete.
digit_rec_eval_test_rbf<- predict(digit_rec_mod_rbf, digit_ds_test)

# Confusion Matrix for RBF Model
# Below line of code may take 2-3 minutes to complete.
confusionMatrix(digit_rec_eval_test_rbf,digit_ds_test$V1)
# Accuracy for RBF model on Test data set is 93.57% which is nearly equal to train data set

###################################################################################
# CONCLUSION                                                                      #
# From our obsernation we can conclude that the RBF kernel gives us the           #
# best result with Accuracy 93.57% which is very close to the percentage accuracy #
# that we have got with our training dataset.                                     #
###################################################################################
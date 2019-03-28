#----------------------------------------  Bike Renting  -------------------------------------------------------

# - - - - - - - - - - - - - - - - - - - -  Data Exploration - - - - - - - - - - - - - - - - - - - - - - - -#

# clean the environment
rm(list=ls(all=T))



#install.packages(x)

#Loading packeges
x=c("ggplot2","lattice","gridExtra","corrgram","DMwR","usdm","caret","randomForest",
    "unbalanced","caTools","C50","dummies","e1071","MASS","rpart")
#install.packages(x)
#install.packages(x)
lapply(x, require,character.only=T)


# set working Directory
setwd("C:/Users/Rohit/Desktop/Bike Renting Projects")


# Load dataset
dataset=read.csv("day.csv")

# see the dimension of data
dim(dataset)



# see the structure of data 
str(dataset)  # all the variable are in numeric and integer, so we don't need to convert
              # categorical variable into numeric.


# observe summary of the model
summary(dataset) # we can see there is no any missing value, so we don't need to impute.

# let's see uniqe value
#lapply(dataset, unique)





##########################    Missing Value Analysis  ##############################

# Create dataframe with missing value
Missing_val=data.frame(apply(dataset, 2, function(x){sum(is.na(x))}))
Missing_val   # ----> There is no missing value found.
             

######################################################################################
########################    data Exploration through Visualization   ################

# Installing important  Packages for visualization
#install.packages("lattice")
#library(lattice)
#library(ggplot2)
                                                                          

##### Plotting "cnt" with others variables


# "instant"
ggplot(data = dataset, aes(x=instant,y=cnt))+
  geom_bar(stat = 'identity', fill = "dark violet")


# dteday
ggplot(data = dataset, aes(x=dteday,y=cnt))+
  geom_bar(stat = 'identity', fill = "navy blue")

# season
ggplot(data = dataset, aes(x=season,y=cnt))+
  geom_bar(stat = 'identity', fill = "navy blue")

# "yr"
ggplot(data = dataset, aes(x=yr,y=cnt))+
  geom_bar(stat = 'identity', fill = "navy blue")


# "mnth"
ggplot(data = dataset, aes(x=mnth, y=cnt))+
  geom_bar(stat = 'identity', fill = "blue")


# "holiday"
ggplot(data = dataset, aes(x=holiday, y=cnt))+
  geom_bar(stat = 'identity', fill = "blue")


# "holiday"
ggplot(data = dataset, aes(x=holiday, y=cnt))+
  geom_bar(stat = 'identity', fill = "blue")



# "weekday"
ggplot(data = dataset, aes(x=weekday, y=cnt))+
  geom_bar(stat = 'identity', fill = "blue")




# "workingday"
ggplot(data = dataset, aes(x=workingday, y=cnt))+
  geom_bar(stat = 'identity', fill = "blue")




# "holiday"
ggplot(data = dataset, aes(x=holiday, y=cnt))+
  geom_bar(stat = 'identity', fill = "blue")




# "weathersit"
ggplot(data = dataset, aes(x=weathersit, y=cnt))+
  geom_bar(stat = 'identity', fill = "blue")




# "temp"
ggplot(data = dataset, aes(x=temp, y=cnt))+
  geom_point(aes_string(colour = dataset$cnt), size = 0.5, shape = dataset$temp)+
  theme_bw() + ggtitle(" scatter plot Analysis")




# "atemp"
ggplot(data = dataset, aes(x=atemp, y=cnt))+
  geom_point(aes_string(colour = dataset$cnt), size = 0.5, shape = dataset$temp)+
  theme_bw() + ggtitle(" scatter plot Analysis")




# "hum"
ggplot(data = dataset, aes(x=hum, y=cnt))+
  geom_point(aes_string(colour = dataset$cnt), size = 0.5, shape = dataset$temp)+
  theme_bw() + ggtitle(" scatter plot Analysis")




# "windspeed"
ggplot(data = dataset, aes(x=windspeed, y=cnt))+
  geom_point(aes_string(colour = dataset$cnt), size = 0.5, shape = dataset$temp)+
  theme_bw() + ggtitle(" scatter plot Analysis")




# "casual"
ggplot(data = dataset, aes(x=casual, y=cnt))+
  geom_point(aes_string(colour = dataset$cnt), size = 0.5, shape = dataset$temp)+
  theme_bw() + ggtitle(" scatter plot Analysis")




# "registered"
ggplot(data = dataset, aes(x=registered, y=cnt))+
  geom_point(aes_string(colour = dataset$cnt), size = 0.5, shape = dataset$temp)+
  theme_bw() + ggtitle(" scatter plot Analysis")




#################################################################################
##############################   Outlier Analysis ######################################



# seperate out numeric columns

numeric_columns = c("temp","atemp","hum","windspeed","casual","registered")



## Normality check before removing outliers
# Histogram using ggplot
# Install and load packages
#install.packages("gridExtra")
#library(gridExtra)

for(i in 1:length(numeric_columns))
{
  assign(paste0("gn",i),ggplot(dataset,aes_string(x=numeric_columns[i]))+
           geom_histogram(fill="sky blue",colour="black")+
           geom_density()+
           theme_bw()+
           labs(x=numeric_columns[i])+
           ggtitle(paste("Histogram of ",numeric_columns[i])))
}

gridExtra::grid.arrange(gn1,gn2,gn3,ncol=3)
gridExtra::grid.arrange(gn4,gn5,gn6,ncol=3)

# No any variables are  normally distributed. 





### outlier Analysis
# Boxplot distribution and outlier check

for(i in 1:length(numeric_columns)){
  assign(paste0("gn",i),ggplot(dataset,aes_string(y=numeric_columns[i],x="cnt",
                                                  fill=dataset$Absenteeism.time.in.hours))+
           geom_boxplot(outlier.colour = "red",fill="skyblue",outlier.shape = 18,
                        outlier.size = 1,notch = F)+
           theme_bw()+
           labs(y=numeric_columns[i],x="cnt")+
           ggtitle(paste("Box Plot of cnt for",numeric_columns[i])))
}
gridExtra::grid.arrange(gn1,gn2,gn3,gn4,gn5,gn6,ncol=3)




# our new numeric variable which contains outliers-
numeric_columns_outliers = c("hum","windspeed","casual")


# Our sample data is less, So we will Impute these outlies with KNN Imputation Method

for(i in numeric_columns_outliers){
  print(i)
  val=dataset[,i][dataset[,i] %in% boxplot.stats(dataset[,i])$out]
  dataset[,i][dataset[,i] %in% val]= NA
}
#library(DMwR)
dataset = knnImputation(dataset , k=5)


# Now we will check the outliers again just to be sure thay have imputed
for(i in 1:length(numeric_columns)){
  assign(paste0("gn",i),ggplot(dataset,aes_string(y=numeric_columns[i],x="cnt",
                                                  fill=dataset$Absenteeism.time.in.hours))+
           geom_boxplot(outlier.colour = "red",fill="skyblue",outlier.shape = 18,
                        outlier.size = 1,notch = F)+
           theme_bw()+
           labs(y=numeric_columns[i],x="cnt")+
           ggtitle(paste("Box Plot of cnt for",numeric_columns[i])))
}
gridExtra::grid.arrange(gn1,gn2,gn3,gn4,gn5,gn6,ncol=2)

# Almost all the Outliers have been Removed......




#########################################################################################
#################################   Feature Selection   ####################################



# - - - - - - - - - - - - - - --Checking  Multicollinearity - - - - - - - - - - - - - - - - -#


# correlation Plot
#Install library
#install.packages("corrgram")
#library(corrgram)
corrgram(dataset[,numeric_columns], order=F,
         lower.panel=panel.pie,text.panel=panel.txt,main="correlation plot")

# Here from correlaion plot "temp" and "atemp" are highly correlated,
#  so we need to remove one out of them

dataset_deleted = subset(dataset, select = -c(dteday,atemp))
                                       # "dteday" is irrelevant for prediction.




# check Multicollinearity
#install.packages("usdm")
#library(usdm)

# checking coliinearity with VIF
vifcor(dataset_deleted[,-14], th=0.95) #--> No variable from the 13 input variables has collinearity problem.



## ANOVA test for Categprical variable
summary(aov(formula = cnt~instant,data = dataset_deleted))
summary(aov(formula = cnt~season,data = dataset_deleted))
summary(aov(formula = cnt~yr,data = dataset_deleted))
summary(aov(formula = cnt~mnth,data = dataset_deleted))
summary(aov(formula = cnt~holiday,data = dataset_deleted))
summary(aov(formula = cnt~weekday,data = dataset_deleted))
summary(aov(formula = cnt~workingday,data = dataset_deleted))
summary(aov(formula = cnt~weathersit,data = dataset_deleted))

# # Discard those variables whose p value >0.05
# dataset_deleted = subset(dataset_deleted, select = -c(holiday,weekday,workingday))


#----------------------------#--------------------------------#--------------------------------------#
# we cannot rely on Anova to reject irrelevant variables. we need to also look at backward elimination
#----------------------------#--------------------------------#--------------------------------------#




##-----------------------------------  BACKWARD ELIMINATION -------------------------------------------------#

# Run regression model again to check p value , R-squared and Adjusted R-squared
LR_model = lm(cnt~., data=dataset_deleted)

summary(LR_model) # R-squared:0.9827,	Adjusted R-squared:  0.9824
# 
# 
### Dimension Reduction - Discard  "holiday" p>0.75014
dataset_deleted = subset(dataset, select = -c(dteday,atemp,holiday))

 
# #---------------------------------------------------------------------------------------------------#


# Run regression model again to check p value , R-squared and Adjusted R-squared
LR_model = lm(cnt~., data=dataset_deleted)

summary(LR_model) #R-squared: 0.9827,	Adjusted R-squared: 0.9824 

## Dimension Reduction - Discard  "season" p>0.71185
#dataset_deleted = subset(dataset_deleted, select = -c(season,holiday,windspeed,hum,weathersit))
dataset_deleted = subset(dataset, select = -c(dteday,atemp,holiday,season))

 
#------------------------------------------------------------------------------------------------------------

# Run regression model again to check p value , R-squared and Adjusted R-squared
LR_model = lm(cnt~., data=dataset_deleted)

summary(LR_model) # R-squared:0.9827,	Adjusted R-squared: 0.9824 
# 
# 
# ## Dimension Reduction - Discard  "windspeed" p> 0.63997  
dataset_deleted = subset(dataset, select = -c(dteday,atemp,holiday,season,windspeed))

#-------------------------------------------------------------------------------------------------------------

# # Run regression model again to check p value , R-squared and Adjusted R-squared
LR_model = lm(cnt~., data=dataset_deleted)

summary(LR_model) # R-squared:0.9827,	Adjusted R-squared: 0.9824 

# ## Dimension Reduction - Discard  "yr" p>0.64768
dataset_deleted = subset(dataset, select = -c(dteday,atemp,holiday,season,windspeed,yr))

#--------------------------------------------------------------------------------------------------------------

# # Run regression model again to check p value , R-squared and Adjusted R-squared
LR_model = lm(cnt~., data=dataset_deleted)
summary(LR_model) #Multiple R-squared:  0.9827,	Adjusted R-squared:  0.9824 







######################################################################################################
##########################         Feature scaling       #############################################

## Normality check before removing outliers
# Histogram using ggplot
# Instlal and load packages
#install.packages("gridExtra")
#library(gridExtra)

numeric_columns = c("temp","casual","registered")
for(i in 1:length(numeric_columns))
{
  assign(paste0("gn",i),ggplot(dataset_deleted,aes_string(x=numeric_columns[i]))+
           geom_histogram(fill="dark red",colour="black")+
           geom_density()+
           theme_bw()+
           labs(x=numeric_columns[i])+
           ggtitle(paste("Histogram of ",numeric_columns[i])))
}
gridExtra::grid.arrange(gn1,gn2,gn3,ncol=2)



# WE can see no any numeric variables are normally distributed

dataset_deleted[numeric_columns][1,] # Only "casual"  and  "registered" variables need to be scaled



##  Normalization for Non uniformly distributed features
new_numeric_columns = c( "casual","registered" )

for(i in new_numeric_columns){
  print(i)
  dataset_deleted[,i]=(dataset_deleted[,i]-min(dataset_deleted[,i]))/
    (max(dataset_deleted[,i]-min(dataset_deleted[,i])))
}





##################################################################################################
#---------------------------------   Principal component Analysis  ------------------------------#
##################################################################################################
  
#  # devide dataset into raining and test
#  library(caTools)
#  set.seed(123)
#  split = sample.split(dataset_deleted$cnt, SplitRatio = 0.8)
#  training_set = subset(dataset_deleted, split == TRUE)
#  test_set = subset(dataset_deleted, split == FALSE)
#  
#  
#  
#  library(caret)
#  library(e1071)
#  
#  #principal component analysis
#  pca = preProcess(x= training_set[-14], method = "pca", pcaComp = 14)
#  training_set_pca = predict(pca, training_set)
#  test_set_pca = predict(pca, test_set)
#  
#  
#  ### Plot of explained varience of principal components 
#  #principal component analysis
#  prin_comp = prcomp(training_set_pca)
#  
#  #compute standard deviation of each principal component
#  std_dev = prin_comp$sdev
#  
#  #compute variance
#  pr_var = std_dev^2
#  
#  #proportion of variance explained
#  prop_varex = pr_var/sum(pr_var)
#  
#  
#  #cumulative scree plot
#  plot(cumsum(prop_varex), xlab = "Principal Component",
#       ylab = "Cumulative Proportion of Variance Explained",
#       type = "b")
#  
#  
#  
#  
#  # choosing to 10 explained varience of Principal components
#  pca = preProcess(x= training_set[-14], method = "pca", pcaComp = 10)
#  training_set = predict(pca, training_set)
#  #training_set_pca = training_set_pca[c(2:21,1)]
#  test_set= predict(pca, test_set)
#  #test_set_pca = test_set_pca[c(2:21,1)]


 #---------> Not getting  accuray and RMSE upto the mark by applying PCA. So let comment this code...






#################################################################################################
#################################     Machine Learning Model      ###############################
#################################################################################################

# devide dataset into raining and test
#library(caTools)
set.seed(123)
split = sample.split(dataset_deleted$cnt, SplitRatio = 0.8)
training_set = subset(dataset_deleted, split == TRUE)
test_set = subset(dataset_deleted, split == FALSE)



#################################################################################################
#---------------------------------  Multiple Linear Regression ---------------------------------#
#################################################################################################

# Run Multiple Linear Regression
LR_model = lm(cnt~., data=training_set)

print(LR_model)

# Summary of the model
summary(LR_model)


#Lets predict for training data
pred_LR_train = predict(LR_model, training_set[,names(training_set) != "cnt"])



#Lets predict for testing data
pred_LR_test = predict(LR_model, test_set[,names(test_set) != "cnt"])


###  Error Matrics
# install.packages("caret")
# library(caret)

# For training data 
print(postResample(pred = pred_LR_train, obs = training_set$cnt))
#                   RMSE        Rsquared       MAE  
#              280.2155431     0.9785654    132.7495712     --> var = 10
#              265.0227530     0.9808267    149.4006024     --> var = 14 (with instant)
#              265.496560      0.980758     149.171203      --> var = 8 (with instant) 
#              265.2647215     0.9807916    149.2780462     --> var = 13
#              259.0394871     0.9816826    149.6771311     --> var = 32
#              265.3598716     0.9807779    149.1781339     --> var = 10 (backward elimination)


# For testing data 
print(postResample(pred = pred_LR_test, obs = test_set$cnt))
#                  RMSE        Rsquared       MAE 
#               204.1376177    0.9900847    120.3726691     --> var = 10
#               213.9104534    0.9891069    144.2161009     --> vAR = 14 (WITH INSTANT)
#               211.9147831    0.9893222    144.5222100     --> var = 8 (with instant)
#               212.1985047    0.9892815    143.8398934     --> var = 13
#               219.2690424    0.9884365    145.1527685     --> var = 32
#               212.2170503    0.9892882    143.8525429     --> var = 10 (backward elimination)


#install.packages("DMwR")
#library(DMwR)


regr.eval(test_set$cnt, pred_LR_test, stats = c('mape',"mse"))
#                 mape        mse 
#             2.582574e-02  4.167217e+04   --> VAR = 10
#             4.154263e-02  4.575768e+04   --> VAR = 14 (with instant)
#             4.155621e-02  4.490788e+04   --> var = 8 (with instant) 
#             4.100824e-02  4.502821e+04   --> var = 13
#             4.125935e-02  4.807891e+04   --> var = 32
#             4.133941e-02  4.503608e+04   --> var = 10 (backward elimination)
        






####################################################################################################
#------------------------------------    Decision Tree   ------------------------------------------#
####################################################################################################

# Load Library
# install.packages("rpart")
# install.packages("MASS")
#library(rpart)
#library(MASS)

## rpart for regression
DT_model= rpart(cnt ~ ., data = training_set, method = "anova")

summary(DT_model)

#write rules into disk
write(capture.output(summary(DT_model)), "Rules.txt")


#Lets predict for training data
pred_DT_train = predict(DT_model, training_set[,names(training_set) != "cnt"])

#Lets predict for training data
pred_DT_test = predict(DT_model,test_set[,names(test_set) != "cnt"])


# For training data 
print(postResample(pred = pred_DT_train, obs = training_set$cnt))
#                  RMSE     Rsquared       MAE 
#             503.005411   0.930932   376.613617  --> var = 10 (backward elimination)
#             503.005411   0.930932   376.613617  --> var = 14 (with instant)
#             503.005411   0.930932   376.613617  --> var = 8 (with instant)
#             278.9123169   0.9787643 231.5260682 --> pca = 29


# For testing data 
print(postResample(pred = pred_DT_test, obs = test_set$cnt))
#             RMSE         Rsquared       MAE 
#           598.3854972    0.9132051   463.0741917   --> var = 10 (backward elimination)
#           598.3854972    0.9132051    463.0741917   --> var = 14 (with instant)
#           598.3854972    0.9132051    463.0741917   --> var = 8 (with instant)
#           273.1417550    0.9821495 237.6433809      --> pca = 29


#install.packages("DMwR")


regr.eval(test_set$cnt, pred_DT_test, stats = c('mape',"mse"))
#               mape         mse 
#            1.274244e-01   3.580652e+05   --> var = 10 (backward elimination)
#            1.274244e-01   3.580652e+05   --> var = 14 (with instant)
#            1.274244e-01   3.580652e+05   --> var = 8 (with instant)
#            7.501317e-02   7.460642e+04   --> pca = 29






###########################################################################################################
#-----------------------------------------------  Random Forest -------------------------------------------
###########################################################################################################
# Fitting Random Forest Regression to the dataset
# install.packages('randomForest')
#library(randomForest)
set.seed(1234)
RF_model= randomForest(x = training_set[,names(training_set) != "cnt"],
                       y = training_set$cnt,
                       ntree = 500)



#Lets predict for training data
pred_RF_train = predict(RF_model, training_set[,names(training_set) != "cnt"])

#Lets predict for testing data
pred_RF_test = predict(RF_model, test_set[,names(test_set) != "cnt"])



# For training data 
print(postResample(pred = pred_RF_train, obs = training_set$cnt))
#                    RMSE        Rsquared       MAE   
#                 146.1574533   0.9948606  89.1758452 --> var = 10
#                 137.492876    0.995514   82.442821  --> var = 14 (with instant)
#                 162.871908    0.993787  100.288975  --> var = 8 (with instant)
#                 128.6383956   0.9961698  78.0525904 --> var = 13
#                 136.0864544   0.9955704  82.7568190 --> var = 32
#                 129.4719764   0.9960124  77.5708665 --> var = 10 (backward elimination)

# For testing data 
print(postResample(pred = pred_RF_test, obs = test_set$cnt))
#                    RMSE      Rsquared        MAE 
#                 257.9014695   0.9842691 180.7950363  --> var = 10
#                 247.8512782   0.9858284 173.6015227  --> var = 14 (with instant)
#                 280.805484    0.981592  193.479779   --> var = 8 (with instant)
#                 238.4921698   0.9869847 165.5000892  --> var = 13
#                 254.0763684   0.9848461 175.9396383  --> var = 32
#                 240.1007197   0.9865681 165.6189671  --> var = 10 (backward elimination)


#install.packages("DMwR")


regr.eval(test_set$cnt, pred_RF_test, stats = c('mape',"mse"))
#                 mape         mse 
#              5.076966e-02   6.651317e+04      --> var = 10
#              5.303189e-02   6.143026e+04      --> var = 14 (with instant)
#              6.123933e-02   7.885172e+04      --> var = 8 (with instant)
#              4.880772e-02   5.687852e+04      --> var = 13
#              5.18146e-02    6.45548e+04       --> var = 32
#              4.883611e-02   5.764836e+04      --> var = 10 (backward elimination)


#################################################################################################
# ------------------------------ Support Vector Regression --------------------------------------#

# Fitting SVR to the dataset
#install.packages('e1071')
#library(e1071)
SVR_model = svm(formula = cnt ~ .,
                data = training_set,
                type = 'eps-regression',
                kernel = 'radial')



#Lets predict for training data
pred_SVR_train = predict(SVR_model, training_set[,names(training_set) != "cnt"])

#Lets predict for testing data
pred_SVR_test = predict(SVR_model, test_set[,names(test_set) != "cnt"])


### Error Matrics
# For training data 
print(postResample(pred = pred_SVR_train, obs = training_set$cnt))
#  RMSE        Rsquared       MAE    
# 241.6418372   0.9842327 155.8147700   --> var = 10
# 218.1145448   0.9872006 145.1063399   --> var = 14 (with instant)
# 212.3873552   0.9879131 126.8806294   --> var = 8 (with instant)
# 216.5465528   0.9874096 145.0815677   --> var = 13
# 216.7178718   0.9882492 155.5045824   --> var = 32
# 159.340549    0.995279  132.604751    --> pca = 29
# 158.5614541   0.9938639 130.3602662   --> pca 23
# 245.8061856   0.9837364 157.4244467   --> var = 11(with instant)
# 225.5436843   0.9866561 138.2971798   --> var = 9
# 213.9838405   0.9876492 124.9556483   --> var = 7 (with instant)
# 214.5865148   0.9875971 143.2165188   --> var = 12 (with instant)
# 219.3112648   0.9870008 139.2240415   --> var = 11 (backward elimination)
# 216.5171408   0.9874842 132.8764906   --> var = 10 (backward elimination) -hum
# 218.790003    0.987101  138.842426    --> var = 10 (backward elimination)-c(dteday,atemp,holiday,season,windspeed,yr)


# For testing data 
print(postResample(pred = pred_SVR_test, obs = test_set$cnt))
# RMSE        Rsquared       MAE 
# 241.5497479   0.9856728  176.4378057 --> var = 10
# 264.5137424   0.9830154  177.6112711 --> var = 14 (with instant)
# 234.1161382   0.9867547  147.2719685 --> var = 8 (with instant)
# 261.1363841   0.9833744  177.5968209 --> var = 13
# 321.6040233   0.9756642  236.2556200 --> var = 32
# 400.8005538   0.9680364  271.4943015 --> pca = 29
# 319.2998289   0.9765675  215.3607476 --> pca = 23
# 250.9762949   0.9845921  183.7424607 --> var = 11(with instant)
# 238.1457851   0.9864579  157.4111288 --> var = 9
# 223.8205250   0.9879596  142.6153153  --> var = 7 (with instant)
# 240.4636824   0.9858667  162.0764282  --> var = 12 (with instant)
# 228.9709519   0.9872081  164.2722450  --> var = 11 (backward elimination)
# 209.6847996   0.9892038  145.1409615  --> var = 10 (backward elimination)-hum                          
# 221.1328272   0.9880324  159.1453353  --> var = 10 (backward elimination)-c(dteday,atemp,holiday,season,windspeed,yr)
 
#install.packages("DMwR")


regr.eval(test_set$cnt, pred_SVR_test, stats = c('mape',"mse"))
#    mape        mse 
# 5.577674e-02  5.834628e+04  --> var = 10
# 6.139980e-02  6.996752e+04  --> var = 14 (with instant)
# 4.241010e-02  5.481037e+04  --> var = 8 (with instant)
# 6.011951e-02  6.819221e+04  --> var = 13
# 8.069131e-02  1.034291e+05  --> var = 32
# 1.058810e-01  1.606411e+05  --> pca = 29
# 7.646146e-02  1.019524e+05  --> pca = 23
# 4.507340e-02  5.671341e+04  --> var = 9
# 4.098574e-02  5.009563e+04  --> var = 7 (with instant)
# 5.723978e-02  5.782278e+04  --> var = 12 (with instant)
# 4.313067e-02  4.396772e+04  --> var = 10 (backward elimination)-hum
# 4.915313e-02  4.889973e+04  --> var = 10 (backward elimination) -c(dteday,atemp,holiday,season,windspeed,yr)




####################################################################################################################
#############################################  Model selection #####################################################

#                              RMSE train      RMSE test         difference
# Multiple Linear Regression: 265.3598716     212.2170503         -53.142       
# Decision Tree             : 503.005411      598.3854972          95.38
# Random Forest             : 129.4719764     240.1007197          110.629
# SVR                       : 218.790003      221.1328272          2.342


#                              R-squqare
# Multiple Linear Regression: 0.9807779
# Decision Tree             : 0.9821495
# Random Forest             : 0.9865681
# SVR                       : 0.9880324


# SVR has less difference of RMSE test and RMSE train value as well as R^2 is also maximum.
# Therefore, SVR is working as a better pridictive model...


# writing csv file
write.csv(dataset_deleted,"dataset_output in R.csv", row.names = F)


#####################################   THANK YOU    ########################################################

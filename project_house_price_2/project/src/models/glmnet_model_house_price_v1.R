library(data.table)
library(caret)
library(Metrics)
library(glmnet)
library(plotmo)
library(lubridate)
set.seed(7)


house_data<- fread('./project/volume/data/raw/Stat_380_housedata.csv')
house_train <- fread('./project/volume/data/interim/house_train.csv')
house_test <- fread('./project/volume/data/interim/house_test.csv')
house_submit<-fread('./project/volume/data/raw/example_sub.csv')


##########################
# Prep Data for Modeling #
##########################

# subset out only the columns to model

drops<- c('Id')

house_train<-house_train[, !drops, with = FALSE]
house_test<-house_test[, !drops, with = FALSE]


#save the response var because dummyVars will remove
train_y<-house_train$SalePrice

house_test$SalePrice<-0

# work with dummies

dummies <- dummyVars(SalePrice ~ ., data = house_train)
house_train<-predict(dummies, newdata = house_train)
house_test<-predict(dummies, newdata = house_test)



house_train<-data.table(house_train)
house_test<-data.table(house_test)





########################
# Use cross validation #
########################




house_train<-as.matrix(house_train)


house_test<-as.matrix(house_test)


gl_model<-cv.glmnet(house_train, train_y, alpha = 1,family="gaussian")


bestlam<-gl_model$lambda.min

####################################
# fit the model to all of the data #
####################################


#now fit the full model

#fit a logistic model
gl_model<-glmnet(house_train, train_y, alpha = 1,family="gaussian")

plot_glmnet(gl_model)

#save model
saveRDS(gl_model,"./project/volume/models/gl_model.model")

house_test<-as.matrix(house_test)

#use the full model
pred<-predict(gl_model,s=bestlam, newx = house_test)

bestlam
predict(gl_model,s=bestlam, newx = test,type="coefficients")
gl_model



#########################
# make a submision file #
#########################


#our file needs to follow the example submission file format.
#we need the rows to be in the correct order

house_submit$SalePrice<-pred


#now we can write out a submission``
fwrite(house_submit,"./project/volume/data/processed/house_v1.csv")


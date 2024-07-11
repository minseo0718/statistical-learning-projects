#load in libraries
library(data.table)
library(caret)
library(dplyr)
set.seed(21)


house_data<-fread('./project/volume/data/raw/Stat_380_housedata.csv')
QC_table<-fread('./project/volume/data/raw/Stat_380_QC_table.csv')

house_train <- house_data[grep("train_",house_data$Id)]
house_test <- house_data[grep("test_", house_data$Id)]


#######################
# make a master table #
#######################

# First make train and test the same dim, then bind into one table so you can do the same thing to both datasets

# make a future price column for test, even though it is unknown. We will not use this, this is only to make
# them two tables the same size

house_test$SalePrice<-0

#add a column that lets you easily differentiate between train and test rows once they are together
house_test$house_train<-0
house_train$house_train<-1

#sort col
house_train$sortCol <- gsub("train_", "", house_train$Id)
house_test$sortCol <- gsub("test_", "", house_test$Id)

#now bind them together

master<-rbind(house_train,house_test)
master<-full_join(master, QC_table)

############################
# split back to train/test #
############################

# split
house_train<-master[house_train==1]
house_test<-master[house_train==0]

house_train <- house_train[order(as.integer(sortCol))]
house_test <- house_test[order(as.integer(sortCol))]


house_train$SalePrice <- as.integer(house_train$SalePrice)
house_test$SalePrice <- as.integer(house_test$SalePrice)

# clean up columns
house_train$house_train<-NULL
house_test$house_train<-NULL
#house_test$SalePrice<-NULL

house_train$sortCol <- NULL
house_test$sortCol <- NULL

########################
# write out to interim #
########################

fwrite(house_train,"./project/volume/data/interim/house_train.csv")
fwrite(house_test,"./project/volume/data/interim/house_test.csv")


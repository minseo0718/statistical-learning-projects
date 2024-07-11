library(data.table)
library(caret)

# load in data 
test<-fread("./project/volume/data/raw/kaggle_test.csv")
train<-fread("./project/volume/data/raw/kaggle_train.csv")
test_emb<-fread("./project/volume/data/raw/test_emb.csv")
train_emb<-fread("./project/volume/data/raw/train_emb.csv")

# change reddit col to numeric
train$reddit <- factor(train$reddit)
train$reddit <- as.numeric(train$reddit)

# add reddit col to test table
test$reddit <- 11

# make a master table
train$text <- NULL
test$text <- NULL
train$train <- 1
test$train <- 0

master_train <- cbind(train, train_emb)
master_test <- cbind(test, test_emb)
master <- rbind(master_train, master_test)

# write out to interim 
fwrite(master,"./project/volume/data/interim/reddit_master.csv")
fwrite(master_train,"./project/volume/data/interim/reddit_master_train.csv")
fwrite(master_test,"./project/volume/data/interim/reddit_master_test.csv")


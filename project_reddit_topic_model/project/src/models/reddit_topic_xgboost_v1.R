### xgboost model before PCA

library(data.table)
library(caret)
library(Metrics)
library(xgboost)

# load data 
train<-fread('./project/volume/data/interim/reddit_master_train.csv')
test<-fread('./project/volume/data/interim/reddit_master_test.csv')
example_sub<-fread('./project/volume/data/raw/example_sub.csv')

##########################
# Prep Data for Modeling #
##########################

drops<- c('id', 'train')
train<-train[, !drops, with = FALSE]
test<-test[, !drops, with = FALSE]

y.train <- train$reddit
y.test <- test$reddit

dummies <- dummyVars(reddit~ ., data = train)
x.train<-predict(dummies, newdata = train)
x.test<-predict(dummies, newdata = test)

# notice that I've removed label=departure delay in the dtest line, I have departure delay available to me with the in my dataset but
# you dont have price for the house prices.
dtrain <- xgb.DMatrix(x.train, label=y.train-1,missing=NA)
dtest <- xgb.DMatrix(x.test,missing=NA)

hyper_perm_tune<-NULL
########################
# Use cross validation #
########################

param <- list(  objective           = "multi:softprob",
                num_class           = 11,
                gamma               = 0.3,
                booster             = "gbtree",
                eval_metric         = "mlogloss",
                eta                 = 0.002,
                max_depth           = 5,
                min_child_weight    = 4,
                subsample           = 0.5,
                colsample_bytree    = 1.0,
                tree_method = 'hist'
)

#nrounds = n# trees
XGBm<-xgb.cv(params=param,nfold=5,nrounds=10000,missing=NA,data=dtrain,print_every_n=1,early_stopping_rounds=25)

best_ntrees<-unclass(XGBm)$best_iteration

new_row<-data.table(t(param))

new_row$best_ntrees<-best_ntrees

test_error<-unclass(XGBm)$evaluation_log[best_ntrees,]$test_rmse_mean
new_row$test_error<-test_error
hyper_perm_tune<-rbind(new_row,hyper_perm_tune)

####################################
# fit the model to all of the data #
####################################


# the watchlist will let you see the evaluation metric of the model for the current number of trees.
# in the case of the house price project you do not have the true houseprice on hand so you do not add it to the watchlist, just the dtrain
watchlist <- list( train = dtrain)

# now fit the full model
# I have removed the "early_stop_rounds" argument, you can use it to have the model stop training on its own, but
# you need an evaluation set for that, you do not have that available to you for the house data. You also should have 
# figured out the number of trees (nrounds) from the cross validation step above. 

XGBm<-xgb.train( params=param,nrounds=best_ntrees,missing=NA,data=dtrain,watchlist=watchlist,print_every_n=1)
saveRDS(XGBm,"./project/volume/models/XGB_v1.model")

pred<-predict(XGBm, newdata = dtest)
result<-matrix(pred,ncol=11,byrow=T)
result<-data.table(result)

example_sub[,2:12] <- result

#now we can write out a submission
fwrite(example_sub,"./project/volume/data/processed/reddit_topic_v1.csv")




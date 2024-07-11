### xgboost model after PCA using three tsne_dt

library(data.table)
library(caret)
library(Metrics)
library(xgboost)

tsne_dt_20<-fread('./project/volume/data/interim/reddit_topic_tsne_dt_20.csv')
tsne_dt_30<-fread('./project/volume/data/interim/reddit_topic_tsne_dt_30.csv')
tsne_dt_40<-fread('./project/volume/data/interim/reddit_topic_tsne_dt_40.csv')
example_sub<-fread('./project/volume/data/raw/example_sub.csv')

reddit <- tsne_dt_20$reddit
train <- tsne_dt_20$train

tsne_dt_20$reddit <-NULL
tsne_dt_20$train <-NULL
tsne_dt_30$reddit <-NULL
tsne_dt_30$train <-NULL
tsne_dt_40$reddit <-NULL
tsne_dt_40$train <-NULL

master_tsne_dt <- cbind(tsne_dt_20, tsne_dt_30, tsne_dt_40)
master_tsne_dt <- setNames(master_tsne_dt, c("V1", "V2", "V3", "V4","V5", "V6")) #change the col names
master_tsne_dt$reddit<-reddit
master_tsne_dt$train<-train


master_tsne_dt_train<-master_tsne_dt[train==1]
master_tsne_dt_test<-master_tsne_dt[train==0]

y.train <- master_tsne_dt_train$reddit
y.test <- master_tsne_dt_test$reddit

drops<- c('train')
master_tsne_dt_train<-master_tsne_dt_train[, !drops, with = FALSE]
master_tsne_dt_test<-master_tsne_dt_test[, !drops, with = FALSE]

dummies <- dummyVars(reddit~ ., data = master_tsne_dt_train)
x.train<-predict(dummies, newdata = master_tsne_dt_train)
x.test<-predict(dummies, newdata = master_tsne_dt_test)

# notice that I've removed label=departure delay in the dtest line, I have departure delay available to me with the in my dataset but
# you dont have price for the house prices.
dtrain <- xgb.DMatrix(x.train, label=y.train-1,missing=NA)
dtest <- xgb.DMatrix(x.test,missing=NA)

hyper_perm_tune<-NULL
param <- list(  objective           = "multi:softprob",
                num_class           = 11,
                gamma               = 0.25,
                booster             = "gbtree",
                eval_metric         = "mlogloss",
                eta                 = 0.001,
                max_depth           = 6,
                min_child_weight    = 3,
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
watchlist <- list( train = dtrain)

# now fit the full model
# I have removed the "early_stop_rounds" argument, you can use it to have the model stop training on its own, but
# you need an evaluation set for that, you do not have that available to you for the house data. You also should have 
# figured out the number of trees (nrounds) from the cross validation step above. 

XGBm<-xgb.train( params=param,nrounds=best_ntrees,missing=NA,data=dtrain,watchlist=watchlist,print_every_n=1)
saveRDS(XGBm,"./project/volume/models/XGB_v3.model")

pred<-predict(XGBm, newdata = dtest)
result<-matrix(pred,ncol=11,byrow=T)
result<-data.table(result)

example_sub[,2:12] <- result

#now we can write out a submission
fwrite(example_sub,"./project/volume/data/processed/reddit_topic_v3.csv")




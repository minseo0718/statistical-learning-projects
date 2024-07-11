library(data.table)
library(Rtsne)
library(ggplot2)
library(caret)
library(ggplot2)
library(ClusterR)
library(dplyr)

master<-fread('./project/volume/data/interim/reddit_master.csv')
example_sub<-fread('./project/volume/data/raw/example_sub.csv')

id <- master$id
reddit <- master$reddit
train <- master$train

master$id <- NULL
master$reddit <- NULL
master$train <- NULL

set.seed(3)
tsne<-Rtsne(master,pca = T,perplexity=70,check_duplicates = F)

# grab out the coordinates
tsne_dt<-data.table(tsne$Y)

tsne_dt$reddit<-reddit
tsne_dt$train<-train

# plot only for train 
ggplot(subset(tsne_dt, train == 1), aes(x = V1, y = V2, col = as.factor(reddit))) + geom_point()

# plot for both
#ggplot(tsne_dt,aes(x=V1,y=V2, col=as.factor(tsne_dt$reddit)))+geom_point()

fwrite(tsne_dt,"./project/volume/data/interim/reddit_topic_tsne_dt_70.csv")


##BUDT 758T - Data Mining and Predictive Analytics
##Practice script: lecture 19
##Author: Professor Jessica Clark
##Topic: Text Mining

#if you have a PC, this code should work:
install.packages("text2vec")

#I had some issues installing it on my Mac, so if you have a problem, try:
install.packages("devtools")
require(devtools)
install.packages("irlba")
install_version("text2vec", version = "0.5.1", repos = "http://cran.us.r-project.org")

#you'll also have to do:
install.packages("tm")
install.packages("SnowballC")
install.packages("vip")

# You may have to do library(text2vec) twice
library(tidyverse)
library(tm)
library(text2vec)
library(text2vec)
library(SnowballC)
library(glmnet)
library(vip)

set.seed(1)

# before 3/25/17, there were two types of tweets (Android and iPhone).
#hypothesis is that Android tweets were from DT, and iPhone tweets were from campaign staffers
#after that date, there were only tweets from Twitter for iPhone, so we no longer had the label to tell who was tweeting

#Goal: train a model to predict whether a tweet came from DT or not, based on the phone label.
#Then, make predictions in the "unlabeled" period.

#load the data
campaign_tweets <- read_csv("trump_campaign_tweets.csv") %>%
  mutate(source = as.factor(source))

#retrieve only the "labeled" tweets
labeled_tweets <- campaign_tweets %>%
  filter(labeled_period == 1)

unlabeled_tweets <- campaign_tweets %>%
  filter(labeled_period == 0)

#first, look at some tweets
iphone_only <- labeled_tweets %>%
  filter(source == 'iPhone') %>%
  select(text)

#head will print the top rows of a dataframe
head(iphone_only, 10)

android_only <- labeled_tweets %>%
  filter(source == 'Android') %>%
  select(text)

head(android_only, 10)

# Functions that are outside of the main word tokenizer 
# These aren't included in the text packages
# Convert all words to lowercase
prep_fun = tolower

# Define a "tokenizer" - a way to convert documents to tokens (i.e. features)
# These options are mainly described in the tm package for R
# https://cran.r-project.org/web/packages/tm/tm.pdf

cleaning_tokenizer <- function(v) {
  v %>%
    #removeNumbers %>% #remove all numbers
    #removePunctuation %>% #remove all punctuation
    removeWords(stopwords(kind="en")) %>% #remove stopwords
    #stemDocument %>%
    word_tokenizer 
}

tok_fun = cleaning_tokenizer

# Iterate over the individual documents and convert them to tokens
# Uses the functions defined above.
it_train = itoken(labeled_tweets$text, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = labeled_tweets$id, 
                  progressbar = FALSE)

# Create the vocabulary from the itoken object
vocab = create_vocabulary(it_train)

#can define your own list of stopwords to supplement the main one, if you want
stop_words = c("will", "new", "us")
vocab2 <- create_vocabulary(it_train, stopwords = stop_words)

#Include ngrams
vocab3 <- create_vocabulary(it_train, stopwords = stop_words, ngram = c(1L, 2L))

#Prune vocabulary
#Try a few values here - what happens?
vocab = prune_vocabulary(vocab, term_count_min = 10, doc_proportion_max = 0.5)

# Create a vectorizer object using the vocabulary we learned
vectorizer = vocab_vectorizer(vocab)

# Convert the training documents into a DTM
dtm_train = create_dtm(it_train, vectorizer) # A  documentterm frequency matrix constists of a lot of terms.Hence we create sparse matrices. 
dim(dtm_train)

# Make a binary BOW matrix
dtm_train_bin <- dtm_train>0+0

# Make a TFIDF DTM
tfidf = TfIdf$new()
dtm_train_tfidf = fit_transform(dtm_train, tfidf)


# Can also use the defined vectorizer to convert other documents (not in the training sample)
it_unlabeled = tok_fun(prep_fun(unlabeled_tweets$text))
it_unlabeled = itoken(it_unlabeled, ids = unlabeled_tweets$id, progressbar = FALSE)
dtm_unlabeled = create_dtm(it_unlabeled, vectorizer)


# you could stop here and just use dtm (maybe selecting a few columns at a time)
# dtm_train and dtm_valid are a SPARSE MATRICES
# they would have more than 1 million entries if stored as a normal, dense matrix

# Not all modeling algorithms we've talked about so far can handle sparse matrices
# More on this next time.


#split labeled into train/valid
train_rows <- sample(nrow(labeled_tweets),.7*nrow(labeled_tweets))
tr_dtm <- dtm_train[train_rows,]
va_dtm <- dtm_train[-train_rows,]

# Get the y values
tr_y <- labeled_tweets[train_rows,]$source
va_y <- labeled_tweets[-train_rows,]$source


# Train a ridge model
grid <- 10^seq(7,-7,length=100)
k<-5
cv.out.ridge <- cv.glmnet(tr_dtm, tr_y, family="binomial", alpha=0, lambda=grid, nfolds=k)
plot(cv.out.ridge)
bestlam_ridge <- cv.out.ridge$lambda.min
pred_ridge <- predict(cv.out.ridge, s=bestlam_ridge, newx = va_dtm,type="response")
class_ridge <- ifelse(pred_ridge > 0.5, "iPhone", "Android")
acc_ridge = mean(ifelse(class_ridge == va_y, 1, 0))

# Make a variable importance plot
vip(cv.out.ridge, num_features = 20)

#Make predictions in the "unlabeled" tweets (which all came from iPhone so we can't tell who wrote them)
pred_ridge_unlabeled <- predict(cv.out.ridge, s=bestlam_ridge, newx = dtm_unlabeled ,type="response")
class_ridge_unlabeled <- ifelse(pred_ridge_unlabeled > 0.5, "iPhone", "Android")

#our hypothesis was that Android tweets were from DT and iPhone tweets were from campaign staffers
#print the prediction for the first 100 "unlabeled" tweets
#do the results match our hypothesis?
for (i in c(1:100)){
  print(paste(unlabeled_tweets$text[i], ', ', class_ridge_unlabeled[i]))
}


#TRY IT YOURSELF

#try some other text featurization techniques. 
#Can you improve on the validation accuracy in lines 61-62?
#Ideas: incorporate bigrams/trigrams, don't lowercase, don't remove stop words, change the vocabulary pruning parameters, think about numbers/punctuation, use TFIDF...
#or use some of the feature engineering ideas we talked about a few weeks ago.

# Inclass activity
tfidf <- TfIdf$new()
dtm_train_tfidf = fit_transform(dtm_train, tfidf)
dtm_train_tfidf

# Installing required packages.

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")

# Loading required library

library(tidyverse)
library(caret)
library(data.table)
library(recosystem)
library(knitr)
library(ggthemes)
library(scales)
library(lubridate)
library(tinytex)
library(rmarkdown)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Loading the data set

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Here, a ratings dataset is created with four columns using fread, which is a fast and friendly 
# file finagler similar to read.table but faster and more convenience to auto detect all controls 
# such as colClasses or nrows.

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

# if using R 3.5 or earlier, use `set.seed(1)`
# I'm using later version of r; so, i will set seed as follow.

set.seed(1, sample.kind="Rounding") 

# Creating a validation set. Validation set will be 10% of MovieLens data.

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Partitioning the data set into a train_set and a test_set

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Matching userId and movieId in both train and test sets

test_set <- temp %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Adding back rows into train set

removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)

# Data Exploratory Anaysis

# Number of rows and columns in the edx dataset?

nrow(edx)
ncol(edx)

# Number of zeros given as ratings in the edx dataset?  

edx %>% filter(rating == 0) %>% tally()

# Number of threes given as ratings in the edx dataset?

edx %>% filter(rating == 3) %>% tally()

# Different movies in the edx dataset?

n_distinct(edx$movieId)

# Different users in the edx data set?

n_distinct(edx$userId)

# Detecting the structure of the edx data set.

edx %>% group_by(genres) %>% 
  summarise(n=n()) %>%
  head()

# genres in ascending order

tibble(count = str_count(edx$genres, fixed("|")), genres = edx$genres) %>% 
  group_by(count, genres) %>%
  summarise(n = n()) %>%
  arrange(count) %>% 
  head()

# In general, half star ratings are less common than whole star ratings 

edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line()

# Distributing ratings per year

edx %>% mutate(year = year(as_datetime(timestamp, origin="1970-01-01"))) %>%
  ggplot(aes(x=year)) +
  geom_histogram(color = "pink") + 
  ggtitle("Distributed: Rating Per Year") +
  xlab("Year") +
  ylab("Ratings in Thousand") +
  scale_y_continuous(labels = comma) + 
  theme_economist()

# Movie with the greatest number of ratings?

edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

# The ten most given ratings in order from most to least?

edx %>% group_by(rating) %>% summarize(count = n()) %>% top_n(10) %>%
  arrange(desc(count))

# Method and Evaluation

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}
# Building Model

# Model 1: Just the average of the data set for all movies and users, 
# The simplest model assumes a random distribution of error from movie to movie variations, when predicting that
# all users will rate all movie the same.Considering statistics theory, the mean, which is just the average of 
# all observed ratings, minimizes the RMSE, as described in the formula below.
#    Ŷ u,i=μ+ϵi,u


mu <- mean(train_set$rating)
rmse1 <- RMSE(test_set$rating, mu)

# Let's see how prediction is at this moment by replicating the x value at 2 and halt times of the 
# test set up to maximum of 1,799,966 rows, and counting the number of rows in the predicted test 
# set and then predicting the RMSE of its rating.

predictions <- rep(2.5, nrow(test_set))
RMSE(test_set$rating, predictions)

# creating a table to store the rmse results for every model's result along the way.

naive_rmse <- RMSE(test_set$rating, mu)

rmse_outputs <- tibble(Method = "Model 1: The Overall Average", RMSE = naive_rmse)

# Model 2: the movie effect on ratings
# From exploratory data analysis, it was observed that some movies are more popular than others and receive 
# higher ratings. Considering the movie effect, this model will be improved by adding the term bi to the 
# formula used to determine the average of all movies like this;
# Yu,i = μ + bi + ϵu,i

bi <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# Visual description of the movie effect

bi %>% ggplot(aes(b_i)) +
  geom_histogram(color = "pink", fill = "darkgrey", bins = 12) +
  xlab("Movie Effect") +
  ylab("Count") +
  theme_bw()

# Normal distribution for the movie effect

bi %>% ggplot(aes(x = b_i)) + 
  geom_histogram(bins=12, col = I("pink")) +
  ggtitle("Distributed: Movie Effect") +
  xlab("Movie effect") +
  ylab("Count") +
  scale_y_continuous(labels = comma) + 
  theme_economist()

# Predicting the rmse of the movie effect on the test set considering the mean as well and then using left_join 
# to return all row column by movieId. 

predicted_ratings <- mu + test_set %>%
  left_join(bi, by = "movieId") %>%
  .$b_i
movie_effect_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_outputs <- bind_rows(rmse_outputs, tibble(Method = "Model 2: The Movie Effect", RMSE = movie_effect_rmse))
rmse_outputs %>% knitr::kable()

# Model 3: the user's specific effect on ratings.
# Considering the user's effect, this model can be improved by adding the term "bu" to the formula used in 
# previous model like this;
# Yu,i = μ + bi + bu + ϵu,i
# Here the user effect model is built on the train_set by returning all rows and columns from the movie effect
# by movieId and then taking the mean of the rating excluding the overall average and the movie effect

bu <- train_set %>%
  left_join(bi, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Normal distribution for the user effect

train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(color = "pink") + 
  ggtitle("Distributed: User Effect") +
  xlab("Bias for User") +
  ylab("Count") +
  scale_y_continuous(labels = comma) + 
  theme_economist()

# plotting the movie and user matrix

users <- sample(unique(edx$userId), 100)
edx %>% filter(userId %in% users) %>%
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% 
  select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies in Number", ylab="Users") %>%
  abline(h=0:100+0.5, v=0:100+0.5, col = "gold") %>%
  title("Matrix: User & Movie")

# Predicting the rmse of the user effect

predicted_ratings <- test_set %>%
  left_join(bi, by = "movieId") %>%
  left_join(bu, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
user_effect_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_outputs <- bind_rows(rmse_outputs, tibble(Method = "Model 3: The Movie & User Effect", RMSE = user_effect_rmse))
rmse_outputs %>% knitr::kable()

# Model 4: regularizing the movie and user effects on rating using the best parameters from lanbdas to penalize 
# or reduce noisy data. Here,three sets of lambdas are defined to tune lambdas beforehand.

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(x){
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+x))
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+x))
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

# plotting lambdas vs. RMSE

qplot(lambdas, rmses, color = I("pink"))

# Picking lambdas with the lowest RMSE to be used for regularizing the movie and user effects.

lamb <- lambdas[which.min(rmses)]
lamb

# Predicting the rmse from a regularized movie and user effects

rmse_outputs <- bind_rows(rmse_outputs, tibble(Method = "Model 4: Regularizing - the Movie and User Effects", RMSE = min(rmses)))
rmse_outputs %>% knitr::kable()

# Model 5: matrix factorization - alternatively using the recosystem for tuning due to memory gap.
# Matrix Factorization - the alternative Recosystem will be used instead due to the memory gap on commercial 
# computer currently in use. Here, the best tuning parameters is used from an R suggested class object called 
# Reco(). The train() method allows for a set of parameters inside the function and then, the $predict() is used 
# for predicted values.

set.seed(1, sample.kind="Rounding")
train_reco <- with(train_set, data_memory(user_index = userId, item_index = movieId, rating = rating))
test_reco <- with(test_set, data_memory(user_index = userId, item_index = movieId, rating = rating))
rec <- Reco()

alt_reco <- rec$tune(train_reco, opts = list(dim = c(20, 30),
                                             lrate = c(0.01, 0.1),
                                             costp_l1 = c(0.01, 0.1),
                                             costq_l1 = c(0.01, 0.1),
                                             nthread = 4,
                                             niter = 10))

rec$train(train_reco, opts = c(alt_reco$min, nthread = 4, niter = 40))
results_alt_reco <- rec$predict(test_reco, out_memory())

mat_factor_rmse <- RMSE(results_alt_reco, test_set$rating)
rmse_outputs <- bind_rows(rmse_outputs, tibble(Method = "Model 5: Matrix factorization - alternative recosystem", RMSE = mat_factor_rmse))
rmse_outputs %>% knitr::kable()

# Finalizing rmse prediction on the validation set. The lowest thus far, has been obtained on the fourth of 
# four models using matrix factorization with the recosystem. Finally, the edx data set will be used to train 
# result fromm the fourth model, while the validation set will be used to test for accuracy.

set.seed(1, sample.kind="Rounding")
edx_reco_sys <- with(edx, data_memory(user_index = userId, item_index = movieId, rating = rating))
valid_reco <- with(validation, data_memory(user_index = userId, item_index = movieId, rating = rating))
rec <- Reco()

alt_reco <- rec$tune(edx_reco_sys, opts = list(dim = c(20, 30),
                                               lrate = c(0.01, 0.1),
                                               costp_l2 = c(0.01, 0.1),
                                               costq_l2 = c(0.01, 0.1),
                                               nthread = 4,
                                               niter = 10))

rec$train(edx_reco_sys, opts = c(alt_reco$min, nthread = 4, niter = 40))

valid_reco <- rec$predict(valid_reco, out_memory())

valid_final_rmse <- RMSE(valid_reco, validation$rating)
rmse_outputs <- bind_rows(rmse_outputs, tibble(Method = "Final validation: Matrix factorization - alternative recosystem", RMSE = valid_final_rmse))
rmse_outputs %>% knitr::kable()
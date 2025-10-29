# scripts/02_supervised_sentiment_model.R
source("scripts/00_project_setup.R")

library(text2vec)
library(data.table)
library(Matrix)
library(glmnet)
library(caret)
library(pROC)
library(dplyr)

# 1. Load data & basic cleaning
data <- read.csv(DATA_PATH, header = TRUE, stringsAsFactors = FALSE)
data$Review <- iconv(data$Review, to = "UTF-8", sub = "byte")
data$Review <- gsub("\t|\r|\n", " ", data$Review)
data$Review <- trimws(data$Review)
data <- data[!is.na(data$Review) & nchar(data$Review) > 0, ]
message("Rows after cleaning: ", nrow(data))

# 2. Create binary labels from Rating (same mapping as before)
df <- data %>% select(Review, Rating) %>%
  mutate(label = case_when(Rating >= 4 ~ "positive",
                           Rating <= 2 ~ "negative",
                           TRUE ~ NA_character_)) %>%
  filter(!is.na(label))

df$label <- factor(df$label, levels = c("negative", "positive"))
table(df$label)

# 3. Train/Test split (stratified)
set.seed(42)
trainIndex <- createDataPartition(df$label, p = 0.8, list = FALSE)
train_df <- df[trainIndex, ]
test_df  <- df[-trainIndex, ]
message("Train rows: ", nrow(train_df), " Test rows: ", nrow(test_df))

# 4. Load saved vocabulary and tfidf transformer (from exploratory step)
vocab <- readRDS(file.path(OUTPUT_MODELS, "vocabulary.rds"))
tfidf_transformer <- readRDS(file.path(OUTPUT_MODELS, "tfidf_transformer.rds"))

# 5. Build itoken iterators for train and test (use same preprocessing/tokenizer)
prep_fun <- tolower
tok_fun <- word_tokenizer

it_train <- itoken(train_df$Review, preprocessor = prep_fun, tokenizer = tok_fun, progressbar = TRUE)
it_test  <- itoken(test_df$Review, preprocessor = prep_fun, tokenizer = tok_fun, progressbar = TRUE)

# 6. Create vectorizer from vocab and generate DTM (sparse)
vectorizer <- vocab_vectorizer(vocab)
dtm_train <- create_dtm(it_train, vectorizer)    # sparse matrix (n_train x n_terms)
dtm_test  <- create_dtm(it_test, vectorizer)     # sparse matrix (n_test x n_terms)

# 7. Apply the same TF-IDF transformer (fitted earlier)
dtm_train_tfidf <- tfidf_transformer$transform(dtm_train)
dtm_test_tfidf  <- tfidf_transformer$transform(dtm_test)

# 8. Convert labels to numeric for glmnet: positive=1 negative=0
y_train <- ifelse(train_df$label == "positive", 1L, 0L)
y_test  <- ifelse(test_df$label == "positive", 1L, 0L)

# 9. Train glmnet logistic regression with CV (use sparse input)
# alpha = 0 -> ridge, alpha = 1 -> lasso; choose alpha via experiments.
set.seed(42)
cvfit <- cv.glmnet(x = dtm_train_tfidf, y = y_train, family = "binomial",
                   type.measure = "auc", nfolds = 5, parallel = FALSE)
best_lambda <- cvfit$lambda.min
message("Best lambda: ", best_lambda)

# 10. Evaluate on test set
pred_prob <- predict(cvfit, dtm_test_tfidf, s = "lambda.min", type = "response")[,1]  # probabilities
pred_class <- ifelse(pred_prob >= 0.5, "positive", "negative")
pred_class <- factor(pred_class, levels = c("negative", "positive"))

conf <- confusionMatrix(pred_class, factor(test_df$label, levels = c("negative","positive")), positive = "positive")
print(conf)

roc_obj <- roc(response = y_test, predictor = as.numeric(pred_prob))
message("AUC (test): ", round(auc(roc_obj), 4))

# 11. Save ROC plot
png(file.path(OUTPUT_FIGURES, "roc_glmnet.png"), width = 800, height = 600)
plot(roc_obj, main = paste0("GLMNET ROC AUC: ", round(auc(roc_obj), 4)))
dev.off()

# 12. Save model and artifacts
saveRDS(cvfit, file = file.path(OUTPUT_MODELS, "glmnet_cvfit.rds"))
saveRDS(vocab, file = file.path(OUTPUT_MODELS, "vocab_for_model.rds"))

# 13. Save test predictions for error analysis
results_df <- data.frame(Review = test_df$Review,
                         true = test_df$label,
                         pred_prob = pred_prob,
                         pred_class = pred_class,
                         stringsAsFactors = FALSE)
write.csv(results_df, file = file.path("reports", "test_predictions_glmnet.csv"), row.names = FALSE)
message("Saved test predictions to reports/test_predictions_glmnet.csv")
message("02_supervised_sentiment_model.R completed. Model saved to models/")

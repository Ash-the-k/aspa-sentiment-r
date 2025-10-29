# scripts/01_exploratory_sentiment.R
# Exploratory sentiment analysis using text2vec (memory-efficient)
source("scripts/00_project_setup.R")

library(text2vec)
library(data.table)
library(ggplot2)
library(wordcloud)
library(RColorBrewer)
library(syuzhet)
library(Matrix)    # explicit for sparse matrix helpers

# 1. Load and basic clean
data <- read.csv(DATA_PATH, header = TRUE, stringsAsFactors = FALSE)
message("Rows: ", nrow(data), " Columns: ", paste(names(data), collapse = ", "))

# remove NA or empty reviews
data$Review <- iconv(data$Review, to = "UTF-8", sub = "byte")
data$Review <- gsub("\t|\r|\n", " ", data$Review)
data$Review <- trimws(data$Review)
data <- data[!is.na(data$Review) & nchar(data$Review) > 0, ]
message("After dropping NA/empty reviews: ", nrow(data), " rows")

# 2. Prepare text vector
texts <- data$Review

# 3. Create itoken iterator (preprocessing + tokenization)
prep_fun <- tolower
tok_fun  <- word_tokenizer

it_all <- itoken(texts, preprocessor = prep_fun, tokenizer = tok_fun, progressbar = TRUE)

# 4. Build vocabulary (prune to keep memory reasonable)
vocab <- create_vocabulary(it_all, stopwords = stopwords("en"))
vocab <- prune_vocabulary(vocab,
                          term_count_min = 5,
                          doc_proportion_min = 0.0005,
                          doc_proportion_max = 0.8)

message("Vocabulary size after pruning: ", nrow(vocab))

# Save vocabulary for reproducibility / later use
vocab_path <- file.path(OUTPUT_MODELS, "vocabulary.rds")
dir.create(OUTPUT_MODELS, showWarnings = FALSE)
saveRDS(vocab, file = vocab_path)
message("Saved vocab to: ", vocab_path)

# 5. Create vectorizer and sparse DTM (documents x terms)
vectorizer <- vocab_vectorizer(vocab)
it_all <- itoken(texts, preprocessor = prep_fun, tokenizer = tok_fun, progressbar = TRUE) # re-create iterator
dtm <- create_dtm(it_all, vectorizer)  # sparse dgCMatrix (n_docs x n_terms)

# 6. TF-IDF transform
tfidf_transformer <- TfIdf$new(norm = "l2")   # keep sparse format
dtm_tfidf <- tfidf_transformer$fit_transform(dtm)

# -------------------------
# Helper: robust column sums
# -------------------------
compute_col_sums_safe <- function(obj) {
  # Try Matrix::colSums for sparse Matrix
  try({
    if (inherits(obj, "dgCMatrix") || inherits(obj, "dgTMatrix") || inherits(obj, "sparseMatrix")) {
      cs <- Matrix::colSums(obj)
      if (!is.null(cs) && length(cs) > 0) return(as.numeric(cs))
    }
  }, silent = TRUE)

  # Try base colSums if it's a matrix-like object
  try({
    if (is.matrix(obj)) {
      cs <- base::colSums(obj)
      if (!is.null(cs) && length(cs) > 0) return(as.numeric(cs))
    }
  }, silent = TRUE)

  # If obj is numeric vector with names
  if (is.numeric(obj) && !is.null(names(obj))) {
    return(as.numeric(obj))
  }

  # Try coercing to matrix and then colSums
  try({
    m <- as.matrix(obj)
    if (is.matrix(m)) {
      cs <- base::colSums(m)
      if (!is.null(cs) && length(cs) > 0) return(as.numeric(cs))
    }
  }, silent = TRUE)

  # Fallback: NULL (caller should try other sources)
  return(NULL)
}

# --- Robust replacement with safe attempts ---
term_scores <- NULL

# 1) Try dtm_tfidf
if (exists("dtm_tfidf")) {
  term_scores <- compute_col_sums_safe(dtm_tfidf)
  if (!is.null(term_scores)) {
    names(term_scores) <- colnames(dtm_tfidf) %||% names(term_scores)
    message("Using dtm_tfidf column sums for term scores.")
  } else {
    message("dtm_tfidf exists but column-sum extraction failed; will try dtm.")
  }
}

# 2) Try dtm if previous failed
if (is.null(term_scores) && exists("dtm")) {
  term_scores <- compute_col_sums_safe(dtm)
  if (!is.null(term_scores)) {
    names(term_scores) <- colnames(dtm) %||% names(term_scores)
    message("Using dtm column sums for term scores.")
  } else {
    message("dtm exists but column-sum extraction failed; will try vocab fallback.")
  }
}

# 3) Fallback to vocabulary counts
if (is.null(term_scores) && exists("vocab") && !is.null(vocab$term_count)) {
  term_scores <- setNames(vocab$term_count, vocab$term)
  message("Falling back to vocab$term_count for term scores.")
}

# If still NULL -> stop with clear error
if (is.null(term_scores)) {
  stop("Cannot compute term scores: dtm_tfidf and dtm both failed and vocab fallback not available.")
}

# Ensure names present
if (is.null(names(term_scores)) && exists("dtm") && !is.null(colnames(dtm))) {
  names(term_scores) <- colnames(dtm)
}

# proceed to sort and plot
term_scores_sorted <- sort(term_scores, decreasing = TRUE)
top_n <- min(length(term_scores_sorted), 50)
top_terms_df <- data.frame(term = names(term_scores_sorted)[1:top_n],
                           score = as.numeric(term_scores_sorted[1:top_n]),
                           stringsAsFactors = FALSE)

# plot top terms
p_top <- ggplot(top_terms_df[1:min(20, nrow(top_terms_df)),], aes(reorder(term, score), score)) +
  geom_col() + coord_flip() + labs(title = "Top terms (TF-IDF or counts)", x = "", y = "Score")
ggsave(file.path(OUTPUT_FIGURES, "top_terms_text2vec.png"), p_top, width = 8, height = 6)
message("Saved top terms plot to figures/")

# wordcloud
png(file.path(OUTPUT_FIGURES, "wordcloud_text2vec.png"), width = 900, height = 700)
set.seed(123)
wordlist <- names(term_scores_sorted)[1:min(length(term_scores_sorted), 150)]
wordfreqs <- as.numeric(term_scores_sorted[1:length(wordlist)])
wordcloud(wordlist, freq = wordfreqs, min.freq = 2, scale = c(3, 0.4), max.words = length(wordlist), random.order = FALSE, colors = brewer.pal(8, "Dark2"))
dev.off()
message("Saved wordcloud to figures/")

# 9. Sentiment scores using syuzhet/bing/afinn (for exploratory insight)
syuzhet_scores <- get_sentiment(texts, method = "syuzhet")
bing_scores <- get_sentiment(texts, method = "bing")
afinn_scores <- get_sentiment(texts, method = "afinn")

# Save histogram of syuzhet
p_hist <- ggplot(data.frame(score = syuzhet_scores), aes(score)) + geom_histogram(bins = 40) +
  labs(title = "Syuzhet sentiment score distribution", x = "Syuzhet score", y = "Count")
ggsave(file.path(OUTPUT_FIGURES, "syuzhet_hist.png"), p_hist, width = 8, height = 5)
message("Saved syuzhet histogram to figures/")

# 10. NRC emotions if desired (may take a bit)
nrc_data <- get_nrc_sentiment(texts)
emotion_sum <- colSums(nrc_data)
emotion_df <- data.frame(emotion = names(emotion_sum), count = as.numeric(emotion_sum), stringsAsFactors = FALSE)
p_em <- ggplot(emotion_df, aes(reorder(emotion, count), count)) + geom_col() + coord_flip() + labs(title = "NRC emotion counts")
ggsave(file.path(OUTPUT_FIGURES, "nrc_emotions.png"), p_em, width = 8, height = 6)
message("Saved NRC emotions plot to figures/")

# 11. Save small sample CSV (for manual inspection)
sample_out <- data.frame(Review = texts[1:500],
                         syuzhet = syuzhet_scores[1:500],
                         bing = bing_scores[1:500],
                         afinn = afinn_scores[1:500],
                         stringsAsFactors = FALSE)
write.csv(sample_out, file = file.path("reports", "sentiment_sample_full.csv"), row.names = FALSE)
message("Saved sentiment sample to reports/")

# 12. Persist helpful objects for supervised use
saveRDS(tfidf_transformer, file = file.path(OUTPUT_MODELS, "tfidf_transformer.rds"))
# Optionally save dtm_tfidf if you have disk space:
# saveRDS(dtm_tfidf, file = file.path(OUTPUT_MODELS, "dtm_tfidf_full.rds"))

message("01_exploratory_sentiment.R completed. Figures saved to: ", OUTPUT_FIGURES)
message("Vocabulary and tfidf transformer saved to: ", OUTPUT_MODELS)

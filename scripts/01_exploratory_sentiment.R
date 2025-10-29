# scripts/01_exploratory_sentiment.R
# Exploratory sentiment analysis using text2vec (memory-efficient)
# - High-quality figures (PNG + SVG)
# - Lexicon comparison (syuzhet, bing, afinn)
# - Robust term-score extraction with safe fallbacks
# - Exports CSVs for reporting

source("scripts/00_project_setup.R")  # sets PROJECT_ROOT, DATA_PATH, OUTPUT_FIGURES, OUTPUT_MODELS

# ---- libraries ----
library(text2vec)
library(data.table)
library(ggplot2)
library(wordcloud)
library(RColorBrewer)
library(syuzhet)
library(Matrix)
library(dplyr)
library(reshape2)
library(scales)
library(tm)        # for stopwords
library(ggtext)
# svg export (optional). If not installed, we'll skip SVG.
svglite_available <- requireNamespace("svglite", quietly = TRUE)

# ensure outputs exist
dir.create(OUTPUT_FIGURES, showWarnings = FALSE, recursive = TRUE)
dir.create(OUTPUT_MODELS, showWarnings = FALSE, recursive = TRUE)
dir.create("reports", showWarnings = FALSE, recursive = TRUE)

# ---- 1. Load and basic clean ----
data <- read.csv(DATA_PATH, header = TRUE, stringsAsFactors = FALSE)
message("Rows: ", nrow(data), " Columns: ", paste(names(data), collapse = ", "))

data$Review <- iconv(data$Review, to = "UTF-8", sub = "byte")
data$Review <- gsub("\t|\r|\n", " ", data$Review)
data$Review <- trimws(data$Review)
data <- data[!is.na(data$Review) & nchar(data$Review) > 0, ]
message("After dropping NA/empty reviews: ", nrow(data), " rows")

texts <- data$Review

# ---- 2. Create itoken iterator ----
prep_fun <- tolower
tok_fun  <- word_tokenizer
it_all <- itoken(texts, preprocessor = prep_fun, tokenizer = tok_fun, progressbar = TRUE)

# ---- 3. Build vocabulary with tm stopwords ----
vocab <- create_vocabulary(it_all, stopwords = tm::stopwords("english"))
vocab <- prune_vocabulary(vocab,
                          term_count_min = 5,
                          doc_proportion_min = 0.0005,
                          doc_proportion_max = 0.8)
message("Vocabulary size after pruning: ", nrow(vocab))

# save vocab
saveRDS(vocab, file = file.path(OUTPUT_MODELS, "vocabulary.rds"))
message("Saved vocab to: ", file.path(OUTPUT_MODELS, "vocabulary.rds"))

# ---- 4. create DTM and TF-IDF ----
vectorizer <- vocab_vectorizer(vocab)
it_all <- itoken(texts, preprocessor = prep_fun, tokenizer = tok_fun, progressbar = TRUE)
dtm <- create_dtm(it_all, vectorizer)                 # sparse dgCMatrix
message("DTM dims: ", paste(dim(dtm), collapse = " x "))

tfidf_transformer <- TfIdf$new(norm = "l2")
dtm_tfidf <- tfidf_transformer$fit_transform(dtm)
message("TF-IDF dims: ", paste(dim(dtm_tfidf), collapse = " x "))

# save transformer
saveRDS(tfidf_transformer, file = file.path(OUTPUT_MODELS, "tfidf_transformer.rds"))
message("Saved TF-IDF transformer to: ", file.path(OUTPUT_MODELS, "tfidf_transformer.rds"))

# ---- Helper: robust column sums ----
compute_col_sums_safe <- function(obj) {
  # try Matrix::colSums for sparse
  try({
    if (inherits(obj, "dgCMatrix") || inherits(obj, "dgTMatrix") || inherits(obj, "sparseMatrix")) {
      cs <- Matrix::colSums(obj)
      if (!is.null(cs) && length(cs) > 0) return(as.numeric(cs))
    }
  }, silent = TRUE)
  # try base colSums if matrix
  try({
    if (is.matrix(obj)) {
      cs <- base::colSums(obj)
      if (!is.null(cs) && length(cs) > 0) return(as.numeric(cs))
    }
  }, silent = TRUE)
  # numeric vector with names
  if (is.numeric(obj) && !is.null(names(obj))) return(as.numeric(obj))
  # try coercion
  try({
    m <- as.matrix(obj)
    if (is.matrix(m)) {
      cs <- base::colSums(m)
      if (!is.null(cs) && length(cs) > 0) return(as.numeric(cs))
    }
  }, silent = TRUE)
  return(NULL)
}

# ---- 5. Compute term scores for plotting ----
term_scores <- compute_col_sums_safe(dtm_tfidf)
if (!is.null(term_scores)) {
  if (is.null(names(term_scores)) && !is.null(colnames(dtm_tfidf))) names(term_scores) <- colnames(dtm_tfidf)
  message("Using TF-IDF column sums for term scores.")
} else {
  term_scores <- compute_col_sums_safe(dtm)
  if (!is.null(term_scores)) {
    if (is.null(names(term_scores)) && !is.null(colnames(dtm))) names(term_scores) <- colnames(dtm)
    message("Using DTM counts for term scores (fallback).")
  } else if (!is.null(vocab$term_count)) {
    term_scores <- setNames(vocab$term_count, vocab$term)
    message("Falling back to vocabulary term_count for term scores.")
  } else {
    stop("Cannot compute term scores from dtm_tfidf, dtm, or vocab.")
  }
}

term_scores_sorted <- sort(term_scores, decreasing = TRUE)
top_terms_df <- data.frame(term = names(term_scores_sorted),
                           score = as.numeric(term_scores_sorted),
                           stringsAsFactors = FALSE)
# save full top terms table
write.csv(top_terms_df, file = file.path("reports", "top_terms_tfidf_full.csv"), row.names = FALSE)
message("Saved full top-terms table to reports/top_terms_tfidf_full.csv")

# ---- 6. Wordcloud (no artificial strict cap; use reasonable max based on vocab) ----
max_words_wc <- max( min(250, length(term_scores_sorted)), 100 )  # allow many words but bounded
wordlist_all <- names(term_scores_sorted)[1:max_words_wc]
wordfreqs_all <- as.numeric(term_scores_sorted[1:length(wordlist_all)])


# Square PNG
png(file.path(OUTPUT_FIGURES, "wordcloud_top_terms.png"),
    width = 2300, height = 2300, res = 300, bg = "transparent")
par(mar = c(0,0,0,0))
set.seed(123)
wordcloud(words = wordlist_all, freq = wordfreqs_all, min.freq = 2,
          scale = c(5, 0.6), max.words = max_words_wc, random.order = FALSE,
          rot.per = 0.2, use.r.layout = FALSE, colors = brewer.pal(8, "Dark2"))
dev.off()

message("Saved wordcloud: wordcloud_top_terms.png")

# ---- 7. Top terms bar plot (top 20) with explanatory text and labels ----
top20 <- head(top_terms_df, 20)
p_top_pretty <- ggplot(top20, aes(x = reorder(term, score), y = score, fill = score)) +
  geom_col() +
  coord_flip() +
  scale_fill_gradient(low = "#fde0dd", high = "#c51b8a") +
  geom_text(aes(label = round(score, 2)), hjust = -0.1, size = 3.5) +
  labs(
    title = "Top 20 Terms by TF-IDF (sum across corpus)",
    subtitle = "TF-IDF sum across documents — higher values indicate more informative terms",
    x = NULL, y = "TF-IDF sum",
    caption = "Vocabulary pruned: term_count_min=5, doc_proportion_min=0.0005, doc_proportion_max=0.8"
  ) +
  theme_minimal(base_size = 14) +
  theme(axis.text.y = element_text(size = 11),
        plot.subtitle = element_text(size = 11, color = "gray30"),
        plot.caption = element_text(size = 9, color = "gray50")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)))

ggsave(file.path(OUTPUT_FIGURES, "top_terms_tfidf_top20.png"), p_top_pretty, width = 10, height = 7, dpi = 300)
message("Saved top-terms TF-IDF plot: top_terms_tfidf_top20.png")

# ---- 8. Lexicon-based sentiment scores (syuzhet / bing / afinn) ----
syuzhet_scores <- get_sentiment(texts, method = "syuzhet")
bing_scores <- get_sentiment(texts, method = "bing")
afinn_scores <- get_sentiment(texts, method = "afinn")

# save sample CSV
sample_n <- min(500, length(texts))
sample_out <- data.frame(Review = texts[1:sample_n],
                         syuzhet = syuzhet_scores[1:sample_n],
                         bing = bing_scores[1:sample_n],
                         afinn = afinn_scores[1:sample_n],
                         stringsAsFactors = FALSE)
write.csv(sample_out, file = file.path("reports", "sentiment_sample_full.csv"), row.names = FALSE)
message("Saved sentiment sample to reports/sentiment_sample_full.csv")

# build comparison data.frame and labels
sentiment_df <- data.frame(syuzhet = syuzhet_scores, bing = bing_scores, afinn = afinn_scores, stringsAsFactors = FALSE)
cor_table <- cor(sentiment_df, use = "complete.obs")
message("Correlation (syuzhet,bing,afinn):")
print(round(cor_table, 3))

label_from_numeric <- function(x) {
  ifelse(x > 0, "positive", ifelse(x < 0, "negative", "neutral"))
}
sentiment_df$label_syuzhet <- label_from_numeric(sentiment_df$syuzhet)
sentiment_df$label_bing    <- label_from_numeric(sentiment_df$bing)
sentiment_df$label_afinn   <- label_from_numeric(sentiment_df$afinn)

message("Counts (bing labels):"); print(table(sentiment_df$label_bing))
message("Counts (syuzhet labels):"); print(table(sentiment_df$label_syuzhet))
message("Counts (afinn labels):"); print(table(sentiment_df$label_afinn))

# ---- 9. Sentiment histogram comparison (faceted) ----
tmp <- sentiment_df[, c("syuzhet","bing","afinn")]
tmp$id <- seq_len(nrow(tmp))
tmp_m <- reshape2::melt(tmp, id.vars = "id", variable.name = "method", value.name = "score")

p_hist_compare <- ggplot(tmp_m, aes(x = score)) +
  geom_histogram(aes(y = ..density..), bins = 60, fill = "#377eb8", alpha = 0.45, color = "black") +
  geom_density(color = "#e41a1c", size = 0.6, alpha = 0.6) +
  facet_wrap(~method, scales = "free") +
  labs(title = "Sentiment Score Distribution (Syuzhet / Bing / Afinn)",
       subtitle = "Faceted distributions; scales differ because lexicon ranges differ",
       x = "Sentiment score", y = "Density",
       caption = "Thresholding: >0 = positive, <0 = negative, =0 neutral") +
  theme_minimal(base_size = 14)

ggsave(file.path(OUTPUT_FIGURES, "sentiment_hist_lexicons.png"), p_hist_compare, width = 12, height = 6, dpi = 300)
message("Saved sentiment histogram comparison: sentiment_hist_lexicons.png")

# ---- Pie chart: white background + boxed title/subtitle/legend (ggtext) ----
# requires ggtext
if (!requireNamespace("ggtext", quietly = TRUE)) {
  message("ggtext not found. Install with: install.packages('ggtext') to get boxed title/subtitle styling.")
  library(ggtext)
} else {
  library(ggtext)
}

# ensure all three sentiment labels exist (even if count = 0)
raw_dist <- as.data.frame(table(sentiment_df$label_bing), stringsAsFactors = FALSE)
names(raw_dist) <- c("label", "count")
all_labels <- c("positive", "neutral", "negative")

# Use complete from tidyr to ensure all categories exist
dist_bing <- tidyr::complete(raw_dist, label = all_labels, fill = list(count = 0))

# Make sure label is a factor with all levels
dist_bing$label <- factor(dist_bing$label, levels = all_labels)

# compute percentages
dist_bing$percent <- round(100 * dist_bing$count / sum(dist_bing$count), 1)
dist_bing$label_pct <- paste0(dist_bing$label, " (", dist_bing$percent, "%)")

# explicit palette for the three fixed labels
fill_cols <- c("positive" = "#1B9E77", "neutral" = "#D95F02", "negative" = "#7570B3")

p_pie_bing_boxed <- ggplot(dist_bing, aes(x = "", y = count, fill = label)) +
  geom_bar(width = 1, stat = "identity", color = "white", linewidth = 0.4) +
  coord_polar(theta = "y") +
  geom_label(
    data = subset(dist_bing, count > 0),  # Only label slices with data
    aes(label = paste0(percent, "%")),
    position = position_stack(vjust = 0.5),
    size = 4.2, fontface = "bold",
    color = "black",
    fill = alpha("white", 0.95),
    label.size = 0.25,
    label.r = unit(4, "pt")
  ) +
  scale_fill_manual(
    values = fill_cols, 
    breaks = all_labels,  # Force all labels to appear
    drop = FALSE          # Don't drop unused levels
  ) +
  theme_void() +
  labs(
    title = "Sentiment Distribution (Bing lexicon)",
    subtitle = "Percentages of positive, neutral, and negative reviews",
    fill = ""
  ) +
  theme(
    # boxed title (semi-opaque white chip)
    plot.title = ggtext::element_textbox_simple(
      size = 16, face = "bold", halign = 0.5,
      margin = margin(b = 6, t = 4),
      fill = alpha("white", 0.95), color = "black", r = unit(4, "pt"),
      padding = margin(4, 8, 4, 8)
    ),
    plot.subtitle = ggtext::element_textbox_simple(
      size = 11, halign = 0.5,
      fill = alpha("white", 0.95), color = "black", r = unit(3, "pt"),
      padding = margin(2, 6, 2, 6)
    ),
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.title = element_blank(),  # Remove title since fill = ""
    legend.text = element_text(size = 11),  # Use regular text element
    legend.key.size = unit(1, "cm"),
    plot.margin = margin(10, 10, 10, 10)
  ) +
  guides(fill = guide_legend(override.aes = list(size = 5)))

# save with white background for global readability
ggsave(
  file.path(OUTPUT_FIGURES, "sentiment_distribution_bing.png"),
  p_pie_bing_boxed,
  width = 7, height = 7, dpi = 300, bg = "white"
)
message("Saved boxed pie chart with white background: sentiment_distribution_bing.png")


# ---- 11. NRC emotions (may take a bit) and improved bar plot ----
nrc_data <- get_nrc_sentiment(texts)   # might be slow on ~20k reviews
emotion_sum <- colSums(nrc_data)
emotion_df <- data.frame(emotion = names(emotion_sum), count = as.numeric(emotion_sum), stringsAsFactors = FALSE)
emotion_df <- emotion_df %>% arrange(count)
emotion_df$emotion <- factor(emotion_df$emotion, levels = emotion_df$emotion)

p_emotion <- ggplot(emotion_df, aes(x = emotion, y = count, fill = emotion)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_fill_brewer(palette = "Paired") +
  geom_text(aes(label = count), hjust = -0.1, size = 3.5) +
  labs(title = "NRC Emotion Counts", subtitle = "Total mentions of each NRC emotion across the corpus",
       x = "", y = "Total mentions") +
  theme_minimal(base_size = 14) +
  theme(axis.text.y = element_text(size = 12)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)))

ggsave(file.path(OUTPUT_FIGURES, "nrc_emotion_counts.png"), p_emotion, width = 9, height = 7, dpi = 300)
message("Saved NRC emotion counts plot: nrc_emotion_counts.png")

# ---- 12. final messages ----
message("01_exploratory_sentiment.R completed. Figures saved to: ", OUTPUT_FIGURES)
message("Vocabulary and tfidf transformer saved to: ", OUTPUT_MODELS)

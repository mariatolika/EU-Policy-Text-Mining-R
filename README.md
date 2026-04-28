# Loading necessary libraries
library(tidyverse)
library(tidytext)
library(tm)
library(topicmodels)
library(wordcloud)
library(SnowballC)

# 1. Data entry – e.g. local authority reports, NSRF project notices or policy statements
texts <- data.frame(
  doc_id = c(1, 2, 3),
  text = c(
    "The municipality implemented digital governance reforms improving transparency and citizen participation.",
    "EU cohesion funds have fostered sustainable tourism and local economic competitiveness in the Peloponnese region.",
    "Public administration policies focused on innovation, simplification and e-governance transformation."
  )
)

# 2.Text pre-processing
corpus <- VCorpus(VectorSource(texts$text))
corpus <- corpus %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords("english")) %>%
  tm_map(stripWhitespace)

# 3. Creation Document-Term Matrix
dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, 0.9)

# 4. Analysis of Thematic Areas with LDA (Machine Learning)
set.seed(123)
lda_model <- LDA(dtm, k = 2, control = list(seed = 123))
topics <- tidy(lda_model, matrix = "beta")

# 5. Visualization of words by topic
top_terms <- topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

ggplot(top_terms, aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  labs(title = "Θεματική ανάλυση πολιτικών κειμένων (LDA)",
       x = "Όροι", y = "Πιθανότητα (beta)") +
  theme_minimal()

# 6. Create a Wordcloud for an overall perspective
m <- as.matrix(dtm)
word_freq <- sort(colSums(m), decreasing = TRUE)
wordcloud(names(word_freq), word_freq, max.words = 50, colors = brewer.pal(8, "Dark2"))

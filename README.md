# Fake-News-Detection

## Introduction

The aim of this research is to analyze existing studies on fake news detection, focusing on optimizing established models to enhance the ability to detect and classify fake information in the security context. This research employs two main methods: LSTM (Long Short-Term Memory) and BiLSTM (Bidirectional Long Short-Term Memory), combined with Hyperband, an efficient hyperparameter optimization algorithm. The results of the study indicate that Bi-LSTM outperforms other models, achieving the highest accuracy of 99.84%, clearly demonstrating the advantage of processing context in both directions. Bi-LSTM's ability to extract information from both forward and backward sequences of data allows the model to gain a deeper understanding of the relationships between words in the text, thereby improving accuracy in fake news classification. Meanwhile, the LSTM model also demonstrates strong performance, achieving an accuracy of 99.77%, due to its ability to capture sequential dependencies, making it a robust choice for detecting patterns in textual data. Therefore, this research contributes to enhancing the performance of deep learning models in security applications.

## Methodology

### Proposed Framework

The framework for fake news detection starts with two datasets (fake and true) from Kaggle. The data undergoes preprocessing, including text cleaning and processing. After splitting into training (80%) and testing (20%) sets, feature engineering is applied: text is tokenized, padded for consistent length, and labels are encoded numerically. The framework employs BiLSTM and LSTM models to capture long-term and bidirectional dependencies. The models are trained, evaluated, and optimized, with the best-performing model selected for deployment.

### Workflow description of the proposed framework

**Data Cleaning:** The datasets are combined into one, shuffled to prevent ordering bias, and cleaned by removing 6251 duplicate entries. Only the "text" and "label" columns are retained for further processing, while "title," "subject," and "date" columns are discarded as they do not contribute to the classification task.

**Text Processing:** The text data is processed using a custom function that removes extra whitespace, special characters, and single-character words. It converts all text to lowercase, tokenizes it into words, and removes stopwords. The remaining words are lemmatized to their root forms, and words shorter than three characters are discarded. This preprocessing optimizes the dataset for training deep learning models.

**Data Splitting:** The processed text data is split into training (80%) and testing (20%) sets to evaluate the model's performance on unseen data and assess its generalization ability.

**Tokenizing and Padding Text:** Tokenization converts each text sample into a sequence of integers, where each word is assigned a unique ID. For a text sequence T of length L, each word wᵢ is represented as tᵢ = token_index(wᵢ). Padding is then applied to standardize sequence length, with shorter sequences padded with zeros and longer ones truncated. In this case, sequences are padded to a maximum length of 150 tokens.

**Transforming Labels:** The labels are encoded as integers (0 for 'fake news' and 1 for 'true news'), then one-hot encoded using Keras' to_categorical function. For binary classification, the labels are represented as [1, 0] for 'fake news' and [0, 1] for 'true news', ensuring compatibility with deep learning models.

### BiLSTM
The BiLSTM model enhances text processing by analyzing sequences in both forward and backward directions, capturing contextual dependencies from both the past and future. This improves the model's ability to distinguish real from fake news.

To optimize performance, the Hyperband algorithm is used for hyperparameter tuning, focusing on:

- Neurons: {32, 64, 96, 128}
  
- Dropout rate: {0.2, 0.3, 0.4, 0.5}
  
- Learning rate: {0.01, 0.001, 0.0001}

Hyperband runs 10 iterations with early stopping (patience = 3) to identify the optimal hyperparameters efficiently.

The BiLSTM model consists of two LSTM layers: one processes input sequences in the forward direction, while the other processes them in reverse. The outputs of these two layers are concatenated at each time step to capture contextual information from both perspectives.
The forward LSTM updates its states as follows:

### LSTM

# Book-Recommendation-System
This repository implements a machine learning-based **Book Recommendation System** that leverages **Natural Language Processing (NLP)** techniques to recommend book titles based on descriptions. It uses supervised learning to train models on a dataset of book titles, descriptions, and genres.

---

## **Key Features**
- Preprocesses book descriptions with **tokenization**, **stop-word removal**, and **lemmatization**.
- Converts descriptions into 300-dimensional vectors using **Word2Vec** and **Spacy**.
- Implements and evaluates multiple machine learning models for book genre classification.
- Recommends books by matching the input description to trained models.

---

## **Dataset Information**
- **Source**: Custom dataset with 14 genres and 1400+ book entries.
- **Columns**: `title`, `description`, `genre`.
- **Genres**:
  - Action, Adventure, Ghost, Romance, Science Fiction, and more.
- Each genre is labeled numerically (e.g., Action = 1, Romance = 13).

---

## **Preprocessing and Vectorization**
1. **Text Preprocessing**:
   - Descriptions are tokenized, stop-words and punctuation are removed, and words are lemmatized.
2. **Word2Vec Embeddings**:
   - Each description is converted into a 300-dimensional vector.
3. **Scaling**:
   - Negative values in vectors are scaled to a [0, 1] range using **MinMaxScaler**.

---

## **Exploratory Data Analysis**
- **Word Clouds**:
  - Visualized most frequent terms for different genres using WordCloud.
- **Class Distribution**:
  - Ensured balanced representation across genres for effective model training.

---

## **Models and Results**
### **Models Evaluated**
1. **Support Vector Classifier (SVC)**: Best-performing model with:
   - **Accuracy**: 0.95 (Cross-validation), 0.94 (Testing).
2. **Random Forest Classifier**:
   - **Accuracy**: 0.92
3. **Multinomial Naive Bayes**:
   - **Accuracy**: 0.89
4. **K-Nearest Neighbors (KNN)**:
   - **Accuracy**: 0.83 (Not ideal for high-dimensional data).

### **Evaluation Metrics**
- Classification Report:
  - **Precision, Recall, F1-Score** for each genre.
- Confusion Matrix:
  - Visualized genre-wise prediction performance.

### **Model Comparison**
| Model                 | Accuracy |
|------------------------|----------|
| Support Vector Classifier (SVC) | 0.95     |
| Random Forest          | 0.92     |
| Multinomial Naive Bayes| 0.89     |
| KNN                    | 0.83     |

---

## **Usage**
### **Training and Testing**
1. Split the dataset:
   - 80% training and 20% testing, stratified by genre.
2. Train models on scaled feature vectors.
3. Evaluate models using cross-validation and classification metrics.

### **Input and Recommendation**
- Input a description of your choice.
- The system preprocesses the input and predicts a matching genre.
- Recommends 5 book titles from the same genre.

---

## **Code Highlights**
### Preprocessing Function
```python
def preprocess_and_vectorize(text):
    doc = nlp(text)
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return wv.get_mean_vector(filtered_tokens)

# Personalized Hybrid Recommendation System

A personalized recommendation system that predicts and ranks items for users based on their historical interactions and item-level content information.

The project compares four recommendation approaches:

- Popularity-Based Recommendation
- Content-Based Recommendation
- Collaborative Filtering using Matrix Factorization
- Hybrid Neural Recommendation

The models are evaluated using Top-K recommendation metrics such as Precision@K, Recall@K, and NDCG@K.

---

## 📌 Project Overview

Recommendation systems are designed to help users discover relevant items from a large collection of available items.

In this project, the goal is to recommend items that a user is likely to interact with in the future based on their historical behavior.

The project follows this overall pipeline:

Raw Interaction Data
        ↓
Data Preprocessing
        ↓
Temporal Train / Validation / Test Split
        ↓
User & Item Indexing
        ↓
Item Content Feature Engineering
        ↓
Positive Interactions + Negative Sampling
        ↓
Recommendation Models
        ├── Popularity-Based
        ├── Content-Based
        ├── Collaborative Filtering
        └── Hybrid Neural
        ↓
Top-K Recommendation
        ↓
Evaluation
        ├── Precision@K
        ├── Recall@K
        └── NDCG@K

---

## 📊 Dataset

The dataset contains:

- **150,000 interactions**
- **5,000 users**
- **2,000 items**
- **8 columns**

### Dataset Features

| Feature | Description |
|---|---|
| `User_ID` | Unique user identifier |
| `Item_ID` | Unique item identifier |
| `Category` | Item category |
| `Rating` | Rating associated with the interaction |
| `Timestamp` | Time of interaction |
| `Price` | Item price |
| `Platform` | Platform associated with the interaction |
| `Location` | User/context location |

---

## 🧹 Data Preprocessing

Several preprocessing steps are performed before model training.

### 1. Timestamp Conversion

The `Timestamp` column is converted into a datetime format.

This is required for chronological ordering and temporal splitting.

### 2. User and Item Indexing

Original user and item IDs are converted into integer indices.

For example:

```text
User_1 → 0
User_2 → 1
User_3 → 2

3. Positive Interactions

User-item interactions are stored separately for:

Training
Validation
Testing

These are used to determine which items were actually interacted with by each user.

4. Item Content Features

Item-level content features are created using:

Category
Platform
Location
Price

Categorical variables are converted using one-hot encoding.

Price is standardized using StandardScaler.

The resulting item-content matrix contains:

2,000 items × 20 features
⏱️ Temporal Train / Validation / Test Split

Instead of randomly splitting the data, the project uses a temporal split.

For each user, interactions are ordered chronologically.

Conceptually:

Past                                      Future
 │                                           │
 ├──────────── Training ────────┤ Validation ├── Test

For users with at least three interactions:

Earlier interactions → Training
Second-last interaction → Validation
Last interaction → Test

The resulting dataset sizes are:

Split	Interactions
Training	140,000
Validation	5,000
Test	5,000
Why Temporal Splitting?

A recommendation system should predict future behavior using information available in the past.

A random split could allow future interactions to influence training, resulting in data leakage.

🤖 Recommendation Models
1. Popularity-Based Recommendation

The popularity model is used as a simple baseline.

The number of training interactions is counted for each item.

Items are then ranked by interaction frequency.

Most Interacted Item
        ↓
Second Most Interacted
        ↓
Third Most Interacted
        ↓
...

The most popular items are recommended.

Advantage
Very simple
Fast
Provides a baseline for comparison
Limitation

It is not personalized.

Different users may receive the same recommendations.

2. Content-Based Recommendation

The content-based model recommends items based on their item features.

Item features include:

Category
Platform
Location
Scaled Price

The system creates a user profile by averaging the content features of items the user has interacted with.

User's Previous Items
        ↓
Average Item Features
        ↓
User Content Profile
        ↓
Compare Against All Items
        ↓
Rank Similar Items
Similarity Measure

The project uses cosine similarity to compare the user's content profile with item feature vectors.

Higher cosine similarity means the item is more similar to the user's learned content preferences.

3. Collaborative Filtering

Collaborative filtering uses user-item interaction behavior rather than item content.

The project implements collaborative filtering using Matrix Factorization.

The model learns:

User embeddings
Item embeddings
User bias
Item bias

The embeddings have a dimension of 32.

Conceptually:

User ID
   ↓
User Embedding
   ↓
     ┐
     ├── Interaction Score
     ┘
Item Embedding
   ↑
Item ID

The prediction score is based on the interaction between the user and item embeddings, together with user and item biases.

Training

The model is trained using:

Positive interactions
Randomly sampled negative interactions
BCEWithLogitsLoss
Adam optimizer

Two negative samples are generated for each positive interaction.

4. Hybrid Neural Recommendation

The hybrid model combines:

User Embedding
      +
Item Embedding
      +
Item Content Features

The content features are projected into a 32-dimensional representation.

The resulting representations are concatenated and passed through a multilayer neural network.

Architecture
User ID
   ↓
User Embedding (32)

Item ID
   ↓
Item Embedding (32)

Item Content Features (20)
   ↓
Linear Layer
   ↓
ReLU
   ↓
Dropout
   ↓
32-dimensional representation

        ↓
   Concatenation
        ↓
       96
        ↓
    Linear 64
        ↓
      ReLU
        ↓
    Dropout
        ↓
    Linear 32
        ↓
      ReLU
        ↓
    Dropout
        ↓
     Linear 1
        ↓
     Prediction Score
Training

The hybrid model uses:

PyTorch
Adam optimizer
BCEWithLogitsLoss
Mini-batch training
Dropout
Weight decay
Early stopping

The model is trained for up to 50 epochs with a batch size of 512.

Early stopping is used when validation loss stops improving.

🎯 Top-K Recommendation

After training, each model generates a score for candidate items.

For example:

Item A → 0.91
Item B → 0.73
Item C → 0.86
Item D → 0.42

The items are sorted by score:

1. Item A → 0.91
2. Item C → 0.86
3. Item B → 0.73
4. Item D → 0.42

The Top-K items are returned as recommendations.

Items that the user has already interacted with are excluded from the recommendation list.

📏 Evaluation Metrics

The project evaluates recommendation quality using:

Precision@K
Recall@K
NDCG@K
Precision@K

Measures how many of the recommended items are relevant.

Precision@K =
Relevant Recommended Items / K
Recall@K

Measures how many of the relevant items were successfully recommended.

Recall@K =
Relevant Recommended Items
--------------------------
Total Relevant Items
NDCG@K

NDCG evaluates both relevance and ranking position.

Relevant items appearing near the top of the recommendation list receive greater importance.

📈 Results

The project evaluates models at:

K = 5
K = 10
Results at K = 10
Model	Precision@10	Recall@10	NDCG@10
Popularity	0.00052	0.0052	0.002498
Content-Based	0.00050	0.0050	0.002324
Collaborative Filtering	0.00060	0.0060	0.002907
Hybrid Neural	0.00046	0.0046	0.001726
Results at K = 5
Model	Precision@5	Recall@5	NDCG@5
Popularity	0.00064	0.0032	0.001847
Content-Based	0.00060	0.0030	0.001671
Collaborative Filtering	0.00084	0.0042	0.002335
Hybrid Neural	0.00028	0.0014	0.000698
🏆 Model Comparison

Based on the recorded evaluation results, Collaborative Filtering using Matrix Factorization achieved the strongest performance among the evaluated approaches.

At both K=5 and K=10, collaborative filtering achieved the highest Precision, Recall, and NDCG among the four models.

An important observation is that the Hybrid Neural model did not outperform collaborative filtering in this experiment.

This demonstrates that increasing model complexity does not automatically guarantee better recommendation performance.

🔬 Ablation Study

The project also defines an ablation setup to investigate the contribution of different components.

Three configurations are considered:

Model	Components
User + Item	Collaborative embeddings only
User + Item + Content	Collaborative + content features
User + Item + Content + Reward	Full hybrid model

The purpose of an ablation study is to understand whether adding individual components contributes useful information to the model.

🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
PyTorch
Jupyter Notebook / Google Colab
Important Libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
📁 Project Structure
Personalized-Hybrid-Recommendation-System/
│
├── Untitled8 (1).ipynb
├── README.md
└── dataset/
    └── interactions.csv

Update the dataset filename if your actual dataset has a different name.

🚀 How to Run
1. Clone the Repository
git clone <your-repository-url>
cd Personalized-Hybrid-Recommendation-System
2. Install Dependencies
pip install pandas numpy scikit-learn torch jupyter
3. Open the Notebook
jupyter notebook

Open:

Untitled8 (1).ipynb
4. Run the Notebook

Execute the cells sequentially.

The notebook performs:

Data Loading
      ↓
Preprocessing
      ↓
Temporal Split
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Recommendation
      ↓
Evaluation
💡 Key Concepts Demonstrated

This project demonstrates practical understanding of:

Recommendation Systems
Personalized Ranking
Top-K Recommendation
Temporal Data Splitting
Data Leakage Prevention
Feature Engineering
One-Hot Encoding
Feature Scaling
Content-Based Filtering
Cosine Similarity
Collaborative Filtering
Matrix Factorization
User Embeddings
Item Embeddings
Negative Sampling
Neural Networks
ReLU Activation
Dropout
BCEWithLogitsLoss
Adam Optimization
Mini-Batch Training
Early Stopping
Precision@K
Recall@K
NDCG@K
Model Comparison
Ablation Analysis
⚠️ Limitations

Based on the recorded experiment, the hybrid neural model does not outperform the collaborative filtering model.

Possible areas for future investigation include:

Hyperparameter tuning
Alternative negative-sampling strategies
Improved content features
Different neural architectures
Better ranking objectives
Additional interaction/context features
Larger and more diverse datasets
More extensive experimentation
🔮 Future Improvements

Potential improvements include:

Experiment with different embedding dimensions.
Tune learning rate, dropout, batch size, and regularization.
Explore additional item and user features.
Experiment with different negative-sampling ratios.
Test ranking-specific loss functions.
Investigate why the hybrid model underperforms collaborative filtering.
Evaluate the system on larger datasets.
Develop an online recommendation API.
Add a recommendation interface.
Monitor recommendation quality after deployment.
📌 Conclusion

This project demonstrates a complete recommendation-system pipeline, starting from raw user-item interactions and ending with personalized Top-K recommendations.

Four approaches were implemented and compared:

Popularity
    ↓
Content-Based
    ↓
Collaborative Filtering
    ↓
Hybrid Neural Recommendation

Based on the recorded results, Collaborative Filtering using Matrix Factorization performed best among the evaluated approaches.

The project also demonstrates an important machine-learning principle:

A more complex model does not necessarily produce better results.

Model performance must be validated experimentally using appropriate evaluation metrics.

👨‍💻 Author

Your Name

GitHub: [https://github.com/Brahmareddy-Ambavarapu]
Email: [brahmareddy.a271002@gmail.com]

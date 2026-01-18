# ESRAG-LLM-RS
**Exploration-Enhanced Retrieval-Augmented Generation for Recommendation**

---

## 1. Installation

We recommend creating a separate Conda environment before installing dependencies to ensure reproducibility:

'''bash
# Create a new Conda environment
conda create -n esrag-llm-rs python=3.10

# Activate the environment
conda activate esrag-llm-rs

# Install required packages
python -m pip install -r requirements.txt
'''

---

## 2. Repository Overview

This repository provides the complete implementation of **ESRAG-LLM-RS**, including:
1) a rating prediction model (**Recommendation Critic**),
2) an exploration-enhanced RAG retriever (**ESRAG**), and
3) an LLM-based recommendation inference and evaluation pipeline.

### 2.1 Core Scripts

- 'critic.py'  
  Training code for the **Recommendation Critic**, which learns to predict a user’s rating for a target item given their historical interactions. The trained critic provides rating signals for ESRAG training.

- 'esrag.py'  
  Training code for **ESRAG (Exploration-Enhanced RAG)**. It learns a retriever with a **stochastic exploration mechanism (e.g., ε-greedy)**, enabling retrieval beyond purely similarity-based neighbors.

- 'esrag_llm_rs.py'  
  The main inference script of **ESRAG-LLM-RS**. It uses the trained ESRAG retriever to retrieve neighbor evidence from the retrieval database, injects the evidence into the LLM context, and generates final Top-K recommendation lists.

### 2.2 Data and Resources

- 'critic_data_sample.json'  
  A small de-identified sample of training data for the Recommendation Critic, illustrating the three-component input format: user history, target item, and ground-truth rating.

- 'esrag_data_sample.jsonl'  
  A small de-identified sample of **ESRAG training data**. The data extraction logic is **identical to that used during ESRAG-LLM-RS inference**, but the data are used solely to train the retriever (no LLM parameters are trained).

- 'user_movie_history_database_sample.jsonl'  
  A sample **retrieval database** file illustrating the format of the retrieval corpus. The full retrieval database consists of the complete interaction histories of all users (see Section 3.2).

- 'user_movie_history_sample.jsonl'  
  A small sample of inference and evaluation input data for ESRAG-LLM-RS, containing user–item interaction histories in JSONL format.

- 'movie.json'  
  Item metadata file containing structured attributes such as title, director, and cast.

- 'metric/'  
  Evaluation scripts for computing standard Top-K recommendation metrics (HR@K, NDCG@K, Precision@K).

- 'requirements.txt'  
  The list of Python dependencies required to run the project.

---

## 3. Data Collection and Processing

We conduct experiments on two public user–item interaction datasets:

- **Movie domain:** MovieLens (Tag Genome 2021)  
  https://grouplens.org/datasets/movielens/tag-genome-2021

- **Book domain:** Book-Crossing (2022)  
  https://github.com/Bionic1251/Revisiting-the-Tag-Relevance-Prediction-Problem

### 3.1 Raw Fields

- **Movie dataset (MovieLens 2021):** 'user_id', 'title', 'director', 'main actors', 'user rating'
- **Book dataset (Book-Crossing 2022):** 'user_id', 'title', 'URL', 'authors', 'language', 'year published', 'book description', 'user rating'

### 3.2 Dataset Partitioning

After preprocessing, each domain-specific dataset is split into **three non-overlapping subsets** with **strict user-level disjointness**:

#### (1) Recommendation Critic Training Set

This subset is used to train the **Recommendation Critic** and contains **30,000 users**, each with at least **10 interactions**.

For each user:
- One item is randomly selected as the **target item**;
- All remaining items are treated as the user’s **historical interactions**.

Each training instance consists of:
1. **User interaction history** (e.g., title, directedBy, starring, rating);
2. **Target item information** (e.g., title, directedBy, starring);
3. **Ground-truth rating** of the target item.

The data are further split into **training / validation / test = 7 : 2 : 1**.  
A de-identified sample file ('critic_data_sample.json') is provided.

#### (2) ESRAG Training Data and Retrieval Database

After training the Recommendation Critic, we construct the **ESRAG training data** and the **retrieval database**:

- **ESRAG training data:**  
  3,000 users are randomly sampled from users not included in the Critic training set. The data extraction logic is identical to that used during ESRAG-LLM-RS inference.  
  Sample file: 'esrag_data_sample.json'.

- **Retrieval database:**  
  The retrieval database consists of the **complete interaction histories of all users** in the dataset (not limited to the 3,000 training users).  
  Sample file: 'user_movie_history_database_sample.json'.

#### (3) ESRAG-LLM-RS Inference and Evaluation Set

After ESRAG training, the same user sampling and data extraction logic are used for **ESRAG-LLM-RS inference and evaluation**.  
A sample input file ('user_movie_history_sample.jsonl') illustrates the standard inference format.

This partitioning strategy enforces strict separation between:
- Recommendation Critic training;
- ESRAG retriever training;
- ESRAG-LLM-RS inference and evaluation;
thereby preventing information leakage and ensuring fair and reproducible evaluation.

---

## 4. How to Run

All main scripts support the '--help' flag to list available command-line arguments. For example:

'''bash
python esrag_llm_rs.py --help
'''

### Step 1: Train the Recommendation Critic

'''bash
python critic.py
'''

This step trains the rating prediction model used to provide supervision signals for ESRAG training.

---

### Step 2: Train ESRAG (Exploration-Enhanced RAG Retriever)

'''bash
python esrag.py
'''

This step trains the exploration-enhanced retriever using the ESRAG training data (3,000 users) and the global retrieval database.

---

### Step 3: Generate ESRAG-LLM-RS Recommendations

'''bash
python esrag_llm_rs.py
'''

This step retrieves neighbor evidence using the trained ESRAG retriever and calls the LLM to generate Top-K recommendation lists.

---

### Step 4: Evaluate ESRAG-LLM-RS Results

All evaluation scripts are located in the 'metric/' directory.  
**In our experiments, we adopt Option B (Real Rating–Only) as the primary evaluation protocol.**

#### Option B: Real Rating–Only (Primary Setting)

1. **Extract and clean recommended items**
   '''bash
   python metric/extract_filter.py --input outputs/esrag_llm_movie.jsonl
   '''

2. **Attach ground-truth ratings**
   '''bash
   python metric/real_rating.py
   '''

3. **Compute Top-K metrics**
   '''bash
   python metric/metric.py
   '''

This protocol evaluates recommendation quality solely based on **ground-truth user ratings**, without involving auxiliary scoring models.

> *Option A (Critic + Real) is provided in the codebase for ablation or analysis purposes, but is not used as the main evaluation setting.*

---

## 5. Important Notes

- **Both the Recommendation Critic and ESRAG are trainable modules** and must be trained before inference.
- **The LLM itself is not trained**; it is only used at inference time.
- ESRAG optimizes the **retrieval strategy**, not the language model parameters.
- The repository is designed to support reproducible research and fair experimental evaluation.

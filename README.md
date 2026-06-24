# Word Sense Disambiguation (WSD) Comparison Framework

A comparative benchmarking suite for **Word Sense Disambiguation (WSD)**. This repository implements and evaluates multiple classical WSD paradigms, ranging from knowledge-based dictionary methods to supervised machine learning models. 

The framework is built to evaluate performance metrics (Accuracy, Precision, Recall, F1 Score) and training scalability (learning curves) across different configurations, such as the utilization of Part-of-Speech (POS) tagging.

---

## Table of Contents
- [Implemented Algorithms](#implemented-algorithms)
  - [Knowledge-Based / Dictionary Methods](#knowledge-based--dictionary-methods)
  - [Supervised Machine Learning Models](#supervised-machine-learning-models)
- [Dataset Requirements](#dataset-requirements)
- [Project Directory Structure](#project-directory-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
  - [Running the Benchmark Suite](#running-the-benchmark-suite)
  - [Running Individual Models](#running-individual-models)
- [Evaluation Details](#evaluation-details)

---

## Implemented Algorithms

### Knowledge-Based / Dictionary Methods
These approaches utilize **WordNet** lexical database relations and definitions to determine the correct sense of an ambiguous word:

1. **Simple Lesk (`lesk.py`)**
   - Disambiguates by finding the WordNet synset whose definition has the highest token overlap (intersection) with the context sentence.
2. **Extended Lesk (`lesk.py`)**
   - Expands the signature of each candidate synset to include its examples, as well as the definitions and examples of its related WordNet nodes (hypernyms, hyponyms, part meronyms, and part holonyms) to reduce the data-sparsity problem of dictionary overlap.
3. **TF-IDF Lesk (`lesk.py`)**
   - Weighs dictionary overlap using Inverse Document Frequency (IDF) computed over the training set corpus, prioritizing rare/informative overlapping terms over common function words.
4. **Embedding Lesk (`lesk.py`)**
   - Matches semantic signatures using vector space representations. It averages pre-trained Word Embeddings (`glove-wiki-gigaword-50` via Gensim) of the context sentence and candidate synset signatures, resolving the correct sense via cosine similarity.

### Supervised Machine Learning Models
These models learn statistical mappings from context features to sense labels using a supervised training split:

1. **Most Frequent Sense (MFS) Baseline (`baseline.py`)**
   - A standard heuristic baseline in WSD. It maps each lemma (or `lemma, POS` pair) to its most frequent sense label observed in the training corpus.
2. **Naive Bayes WSD (`nb.py`)**
   - Computes the prior probability of each sense and the conditional probability of context words (bag-of-words) given the sense using Laplace smoothing:
     $$P(\text{Sense} \mid \text{Context}) \propto P(\text{Sense}) \prod_{w \in \text{Context}} P(w \mid \text{Sense})$$
3. **Support Vector Machine WSD (`svm.py`)**
   - Trains individual Linear SVMs (`LinearSVC`) for each target lemma or `lemma, POS` pair. It represents context sentences using bag-of-words frequency counts mapped via a `DictVectorizer`.

---

## Dataset Requirements

Supervised and evaluation scripts load data from a local `data/` directory at the project root. This directory should contain sense-annotated corpora formatted as CSV files (e.g., converted SemCor corpus formats).

The CSV files must contain the following columns:
- **`tagfile`**: Identifier of the original source file.
- **`pnum`**: Paragraph index within the document.
- **`snum`**: Sentence index within the paragraph.
- **`value`**: The actual token string in the text.
- **`lemma`**: The lemmatized representation of the token.
- **`wnsn`**: The WordNet Sense Number (sense label). If empty or `'None'`, the token is treated as unannotated.
- **`pos`**: The Part-of-Speech tag (e.g., Penn Treebank format).

> [!NOTE]
> The POS tags are simplified internally: POS tags starting with `N` map to `'n'` (noun), `V` to `'v'` (verb), `J` to `'a'` (adjective), and `R` to `'r'` (adverb).

---

## Project Directory Structure

```
.
├── baseline.py      # Most Frequent Sense (MFS) baseline class
├── lesk.py          # Lesk algorithm family (Simple, Extended, TF-IDF, Embedding)
├── nb.py            # Naive Bayes classifier & standalone test script
├── svm.py           # Linear SVM classifier & standalone test script
├── stats.py         # Multi-threaded benchmark suite & learning curve analysis
├── pyproject.toml   # Project configuration and dependency declarations
├── uv.lock          # Locked dependency graph
└── data/            # Ignored directory for storing the training/testing CSV files
```

---

## Installation & Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast, robust package management.

1. **Install `uv`** (if not already installed):
   ```bash
   curl -LsSf https://astral-sh.uv.tech/install.sh | sh
   ```

2. **Clone the Repository** and navigate to it:
   ```bash
   git clone <repository-url>
   cd Disambiguation-comparison
   ```

3. **Install Dependencies**:
   ```bash
   uv sync
   ```

> [!TIP]
> The system will automatically download necessary NLTK datasets (like WordNet) and Gensim word embedding models (`glove-wiki-gigaword-50`) during the first run of the relevant models.

---

## Usage

### Running the Benchmark Suite
The primary script for comparing all models is [stats.py](file:///home/irishbumfuzzle/Coding/Disambiguation-comparison/stats.py). It runs a parallelized evaluation pipeline and outputs comparative metrics, followed by a learning curve assessment.

```bash
uv run python stats.py
```

#### What `stats.py` does:
1. Loads all CSV datasets from `data/`.
2. Shuffles and splits the sentences: **80% training** and **20% testing**.
3. Takes an evaluation slice (default: 500 sentences) to perform fast validation.
4. Spawns a parallel thread pool (`ThreadPoolExecutor`) to evaluate the following:
   - Naive Bayes (with and without POS constraints)
   - Support Vector Machine (with and without POS constraints)
   - Simple Lesk
   - Extended Lesk
   - TF-IDF Lesk
   - Embedding-based Lesk
   - Most Frequent Sense Baseline
5. Computes and displays **Accuracy**, **Precision**, **Recall**, and **F1 Score** (weighted) for each algorithm.
6. Conducts a **Learning Curve Analysis** by training Naive Bayes and SVM on subsets (10%, 25%, 50%, and 100%) of the training data to inspect sample complexity.

### Running Individual Models
You can run [nb.py](file:///home/irishbumfuzzle/Coding/Disambiguation-comparison/nb.py) or [svm.py](file:///home/irishbumfuzzle/Coding/Disambiguation-comparison/svm.py) directly to train and evaluate only those respective models on the full test split:

```bash
uv run python nb.py
uv run python svm.py
```

---

## Evaluation Details

All evaluation processes follow these standard practices:
- **POS Simplification**: Reduces linguistic tag sets down to WordNet's four syntactic categories ('n', 'v', 'a', 'r') to maximize overlap potential.
- **Out-of-Vocabulary (OOV) Sense fallback**: If an algorithm cannot confidently assign a sense (or the word is unseen), it defaults to assigning sense label `"0"` (or falls back to none) for strict calculation of evaluation metrics.
- **Weighted Metrics**: Precision, Recall, and F1 score are computed using `scikit-learn`'s `'weighted'` average parameter to handle class imbalance across different senses.

# Turkish Spam Mail Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)

Machine learning project for classifying Turkish emails as spam or ham using **K-Nearest Neighbors (KNN)** algorithm. Features custom KNN implementation, Turkish text preprocessing, and comprehensive visualizations.

**Dataset:** Turkish Spam V01 (825 emails) | **Accuracy:** 81.85% with K=3

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Dataset](#dataset)
- [License](#license)

## ✨ Features

- Custom KNN algorithm implementation from scratch
- Turkish language support with stopwords filtering
- Comprehensive text preprocessing pipeline
- Multiple evaluation metrics (accuracy, precision, recall, F1-score)
- Rich visualizations (confusion matrix, histograms, word frequency charts)
- Interactive Jupyter notebook with detailed explanations
- Professional web showcase page

## 📁 Project Structure

```
turkish-spam-mail-detection/
├── data/
│   ├── trspam.csv              # Turkish Spam V01 dataset
│   └── stopwords-tr.txt        # Turkish stopwords list
├── src/
│   └── spam_detection.py       # Main Python script
├── spam_detection.ipynb         # Jupyter notebook
├── index.html                   # Project showcase page
├── requirements.txt             # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

**Note:** `outputs/plots/` directory is generated when running the code and contains visualizations (not tracked in git).

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone repository
git clone https://github.com/yusufarbc/turkish-spam-mail-detection.git
cd turkish-spam-mail-detection

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- scikit-learn >= 1.0.0
- seaborn >= 0.11.0

## 💻 Usage

### Run Python Script

```bash
python src/spam_detection.py
```

### Run Jupyter Notebook

```bash
jupyter notebook spam_detection.ipynb
```

### Output

The program will:
1. Load and preprocess 825 Turkish emails
2. Generate word count histograms (ham vs spam)
3. Analyze word frequency patterns
4. Test multiple K values (3, 5, 7, 9, 11, 15, 19, 24, 30)
5. Train final model with optimal K=3
6. Display detailed metrics and classification report
7. Save visualizations to `outputs/plots/`

## 🔬 Methodology

### 1. Data Preprocessing

- **Punctuation Removal:** Strip all punctuation marks
- **Lowercase Conversion:** Normalize text
- **Stopwords Filtering:** Remove common Turkish words (ve, için, bir, etc.)

### 2. Feature Extraction

Convert emails to word frequency vectors using bag-of-words approach.

### 3. KNN Classification

- **Distance Metric:** Euclidean distance between word frequency vectors
- **K Optimization:** Tested values 3-30, optimal K=3
- **Train/Test Split:** 70/30 with stratification

### 4. Evaluation

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Per-class performance analysis

## 📊 Results

### Model Performance

| Metric | Value |
|--------|-------|
| **Optimal K** | 3 |
| **Accuracy** | 81.85% |
| **Training Size** | 577 emails |
| **Test Size** | 248 emails |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Ham** | 0.82 | 0.92 | 0.87 | 146 |
| **Spam** | 0.91 | 0.78 | 0.84 | 102 |
| **Weighted Avg** | 0.87 | 0.82 | 0.82 | 248 |

### Key Insights

- ✅ Lower K values (3-5) perform better for this dataset
- ✅ Model excels at identifying ham emails (92% recall)
- ✅ High spam precision (91%) minimizes false positives
- ✅ Balanced performance across both classes

## 📈 Visualizations

Generated visualizations (saved to `outputs/plots/`):

1. **confusion_matrix.png** - Model performance matrix
2. **hist_ham.png** - Ham email word count distribution
3. **hist_spam.png** - Spam email word count distribution
4. **plot_ham.png** - Most frequent words in ham emails
5. **plot_spam.png** - Most frequent words in spam emails

View all visualizations on the [project showcase page](https://yusufarbc.github.io/turkish-spam-mail-detection/).

## 📚 Dataset

**Turkish Spam V01**

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Turkish+Spam+V01)
- **Total Emails:** 825
- **Classes:** ham (legitimate), spam
- **Language:** Turkish
- **Format:** CSV (UTF-8)
- **Features:** Email text and classification label

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**yusufarbc**
- GitHub: [@yusufarbc](https://github.com/yusufarbc)

## 🙏 Acknowledgments

- Turkish Spam V01 dataset contributors
- UCI Machine Learning Repository
- Turkish NLP community

---

**[View Live Demo](https://yusufarbc.github.io/turkish-spam-mail-detection/)** | Made with ❤️ for Turkish NLP Community

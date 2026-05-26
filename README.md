# Sentiment Analysis From Scratch

Project này làm bài toán sentiment analysis nhị phân cho positive / negative theo kiểu học cơ bản, tự viết các bước chính và giữ pipeline ngắn gọn.

## Mục tiêu hiện tại

- Chỉ dùng 2 model: Naive Bayes và Decision Tree.
- Chỉ dùng CountVectorizer cho feature extraction.
- Chỉ dùng dữ liệu CSV nằm trong `data/raw`.

## Pipeline

1. Load dataset từ `data/raw/imdb` hoặc `data/raw/sst`.
2. Clean text.
3. Tokenize đơn giản.
4. Remove stopwords.
5. Vectorize bằng CountVectorizer.
6. Train/test split.
7. Train Naive Bayes.
8. Train Decision Tree.
9. Evaluate bằng accuracy, confusion matrix, và classification report.

## Files chính

- `main.py`: chạy baseline từ dòng lệnh.
- `experiments/run_baseline.py`: chạy theo cấu hình YAML.
- `notebook/exploration.ipynb`: notebook khám phá và chạy end-to-end.
- `src/preprocessing/text_processor.py`: clean và tokenize.
- `src/feature_extraction/count_vectorizer.py`: Bag of Words.
- `src/models/naive_bayes/naive_bayes.py`: Naive Bayes from scratch.
- `src/models/decision_tree/decision_tree.py`: Decision Tree from scratch.
- `src/evaluation/accuracy.py` và `src/evaluation/confusion_matrix.py`: metric cơ bản.
- `src/utils/helper.py`: train/test split.

## Datasets

Repo chỉ dùng dữ liệu đã có sẵn trong `data/raw`:

- IMDB: `data/raw/imdb/IMDB_dataset.csv`
- SST: `data/raw/sst/train.csv`

## Chạy project

Chạy CLI baseline:

```bash
python main.py
```

Chạy experiment YAML:

```bash
python experiments/run_baseline.py
```

Mặc định output được lưu trong `results/`.

## Notebook

Notebook `notebook/exploration.ipynb` được giữ ở mức dễ đọc và dễ chạy lại. Notebook chỉ nên chứa đúng các bước của pipeline hiện tại, không thêm thử nghiệm ngoài phạm vi.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Dyu20705/processing-sentiment-with-ml-model.git
cd ml-models-sentiment
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the main baseline

```bash
python main.py --dataset sst --limit 300
```

### 5. Launch the notebook

```bash
jupyter notebook notebook/exploration.ipynb
```

## Results

The repository does not yet present a single canonical benchmark table for the full datasets. However, executable baseline outputs do exist.

### Example Output

The following numbers are from a **small smoke run** of:

```bash
python main.py --dataset sst --limit 300
```

| Run Type | Dataset Slice | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---|---:|---:|---:|---:|
| Example smoke run | SST, first 300 rows | Naive Bayes | 0.5902 | 0.5794 | 0.5786 | 0.5788 |
| Example smoke run | SST, first 300 rows | Decision Tree | 0.5738 | 0.5441 | 0.5247 | 0.4877 |

These numbers should be interpreted as **sanity-check outputs**, not as final benchmark claims.

## License

This repository is released under the **MIT License**. See [LICENSE](LICENSE).
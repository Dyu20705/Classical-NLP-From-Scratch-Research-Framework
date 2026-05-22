# Báo Cáo Bài Tập: Phân Tích Cảm Xúc Bằng Các Thuật Toán ML Cổ Điển

## Abstract

Báo cáo này mô tả trạng thái hiện tại của baseline phân tích cảm xúc được triển khai theo hướng from-scratch trong kho mã. Trọng tâm của dự án là hai mô hình cổ điển phù hợp với văn bản thưa là Multinomial Naive Bayes và Decision Tree, cùng với các thành phần hỗ trợ như tiền xử lý, trích xuất đặc trưng, tách tập stratified, đánh giá và sinh biểu đồ. Tài liệu ưu tiên mô tả đúng hiện trạng kỹ thuật của code hơn là nói theo hướng lý thuyết chung: ma trận thưa được dùng ở tầng vector hóa, run baseline được điều phối qua cấu hình, và báo cáo kết quả được lưu dưới dạng artifact để có thể tái tạo. Các số liệu trong kho hiện phản ánh toy dataset hoặc baseline nhỏ, không nên hiểu là kết quả tối ưu trên SST.

## 1. Giới thiệu

Phân tích cảm xúc (Sentiment Analysis) là một nhiệm vụ nền tảng trong xử lý ngôn ngữ tự nhiên (NLP) với nhiều ứng dụng: phân tích ý kiến khách hàng, giám sát mạng xã hội, và hệ thống khuyến nghị. Văn bản tự nhiên có đặc trưng phân tán cao — không gian đặc trưng rất thưa, thứ bậc từ vựng lớn, và mối tương tác từ vựng phức tạp. Mặc dù các mô hình deep learning hiện đại (LSTM, Transformer/BERT) đạt hiệu năng hàng đầu, các phương pháp ML cổ điển như Naive Bayes và Decision Tree vẫn có giá trị giáo dục và thực tế: chúng minh hoá các giả thiết thống kê, dễ diễn giải và ít tốn tài nguyên khi kết hợp với các biểu diễn dạng bag-of-words.

Dự án trong kho mã nhắm tới mục tiêu thực hành và kiểm chứng thuật toán — hiểu cơ chế bên trong, hạn chế black-box ở mức tối thiểu, và giữ quy trình chạy thử nhất quán giữa mã nguồn, README và báo cáo.

## 2. Related Work

- Naive Bayes: thuật toán kinh điển cho phân lớp văn bản (Multinomial NB), phù hợp với đặc trưng đếm từ (BoW). Ưu điểm: nhanh, nhạy với tính tần suất từ, hiệu quả với dữ liệu thưa và nhiều chiều.
- Decision Trees: mô hình dựa trên luật rẽ nhánh (entropy / information gain). Trực quan nhưng dễ overfit trong không gian đặc trưng cao và thưa.
- Deep Learning: kiến trúc embedding + LSTM/Transformer cho phép học biểu diễn ngữ nghĩa, nhưng yêu cầu nhiều dữ liệu và tài nguyên; không phù hợp cho bài tập tập trung vào hiểu thuật toán từ gốc.

## 3. Mô tả Dữ liệu

- Kho chứa dữ liệu SST ở [data/raw/sst/train.csv](data/raw/sst/train.csv) (các cột `id,target,sentence`). Cột `target` sử dụng nhãn dạng số (ví dụ 0 và 1) theo chuẩn nhãn nhị phân. Tập `test.csv` và `sample_submission.csv` cũng có trong cùng thư mục.
- Trong mã ví dụ (`main.py`), khi không truyền đường dẫn dữ liệu, pipeline sử dụng một "toy dataset" gồm 12 nhận xét (có trong `main.py`) để minh hoạ, do đó các kết quả in ra mặc định là kết quả trên tập rất nhỏ này.
- Mã hiện tại không tự động tải hoặc hợp nhất file SST vào pipeline chính — `main.py` mặc định dùng dữ liệu minh hoạ; tuy nhiên các file gốc sẵn có để thử nghiệm thủ công.
- Phân phối lớp: các file CSV tồn tại nhưng không có bước tự động tính và lưu phân phối lớp trong pipeline mặc định; do đó không có con số phân phối lớp thực thi được xuất sẵn trong mã (ngoại trừ khi chạy trên tập toy đã nêu).

## 4. Methodology

Toàn bộ pipeline như được cài đặt xuất phát từ các thành phần trong `src/`.

### 4.1 Tiền xử lý dữ liệu

- Tập trung trong file [src/preprocessing/text_processor.py](src/preprocessing/text_processor.py). Hàm `TextProcessor` thực hiện:
  - Loại bỏ thẻ HTML: `re.sub(r'<.*?>','', text)`
  - Chuyển về chữ thường (`text.lower()`)
  - Loại bỏ URL và email
  - Tùy chọn loại bỏ số (tham số `remove_numbers` mặc định True)
  - Giữ lại chỉ ký tự chữ cái, dấu nháy đơn, và khoảng trắng (`re.sub(r"[^a-zA-Z\s']", '', text)`)
  - Token hóa đơn giản theo whitespace (`text.split()`)

Lưu ý thực thi: lớp `TextProcessor` hiện đã tách rõ cờ `remove_stopwords` và phương thức lọc stopword, nên chức năng này ổn định hơn trước. Các hạn chế còn lại chủ yếu nằm ở việc mô hình custom vẫn chuyển sparse sang dense và chưa tối ưu cho vocabulary lớn.

### 4.2 Trích xuất đặc trưng

- Bag-of-Words (CountVectorizer): cài đặt trong [src/feature_extraction/count_vectorizer.py](src/feature_extraction/count_vectorizer.py).
  - Token hóa theo whitespace; hỗ trợ n-gram (tham số `ngram_range`) và `max_features` để giới hạn từ vựng
  - Xây dựng `vocabulary_` dựa trên tần suất token trên tập huấn luyện
  - Trả về ma trận thưa dạng `scipy.sparse.csr_matrix` (kiểu `float32`)

- TF–IDF: cài đặt trong [src/feature_extraction/tfidf_vectorizer.py](src/feature_extraction/tfidf_vectorizer.py).
  - Tính IDF với smoothing mặc định (idf = log((N+1)/(df+1)) + 1)
  - Nhân cột bởi hệ số IDF và chuẩn hoá L2 cho mỗi hàng (document)

### 4.3 Multinomial Naive Bayes

- Cài đặt trong [src/models/naive_bayes/naive_bayes.py](src/models/naive_bayes/naive_bayes.py).
- Thuật toán sử dụng công thức log để tránh underflow:

  - Công thức nền tảng:

    $$P(c\\mid d) \\propto P(c) \\prod_{i} P(w_i\\mid c)^{x_i}$$

  - Trong thực thi: tính log-space

    $$\\log P(c\\mid d) = \\log P(c) + \\sum_i x_i \\log P(w_i\\mid c)$$

  - Ưu/nhược:
    - Áp dụng Laplace smoothing (`alpha` tham số, mặc định = 1.0)
    - Chuyển ma trận thưa sang dense bằng `toarray()` trước khi tính toán (xem phần hạn chế)

### 4.4 Decision Tree

- Cài đặt trong [src/models/decision_tree/decision_tree.py](src/models/decision_tree/decision_tree.py).
- Sử dụng entropy làm hàm mất mát và information gain để chọn ngưỡng phân tách. Cụ thể:

  - Entropy:

    $$H(Y) = -\\sum_{k} p_k \\log_2 p_k$$

  - Information gain của một phép tách = entropy(parent) − weighted average entropy(children)

- Thuộc tính thực thi:
  - Hỗ trợ dữ liệu dense (nếu nhận sparse sẽ chuyển qua dense với `toarray()` trước khi huấn luyện)
  - Tham số dừng: `min_samples_split` và `max_depth`
  - Trong môi trường đặc trưng thưa và chiều cao, cây quyết định có xu hướng overfit; mã cũng cung cấp hàm `print_tree()` để in cấu trúc (nhưng hàm này chứa đoạn in đệ quy dư thừa và không tối ưu)

## 5. Thực nghiệm (Thực hiện trong mã)

- Pipeline chính được điều phối bởi [main.py](main.py):
  - Tiền xử lý: `TextProcessor(remove_stopwords=False)` (mặc định không loại stopwords)
  - Chia tập: `train_test_split(..., test_size=0.2, random_state=42, stratify=True)` (hàm định nghĩa trong `src/utils/helper.py`)
  - Vectorization: `CountVectorizer(lowercase=True, ngram_range=(1,1), max_features=1000)`
  - Huấn luyện mô hình:
    - `NaiveBayes(alpha=1.0)`
    - `DecisionTree(max_depth=4, min_samples_split=2)` (trên raw counts)
    - `DecisionTree` trên TF–IDF (sử dụng `TfidfTransformer` mặc định)

- Thư mục `tests/test_pipeline.py` chứa một bộ test toàn diện mô phỏng nhiều kịch bản nhỏ (toy datasets) để kiểm tra:
  - CountVectorizer (unigrams & bigrams)
  - TfidfTransformer (kiểm tra chuẩn L2)
  - train_test_split (stratified)
  - DecisionTree và NaiveBayes với ma trận thưa
  - Một end-to-end test pipeline với dữ liệu mẫu 6 tài liệu

### Hyperparameters quan sát được trong mã

- `NaiveBayes.alpha = 1.0` (Laplace smoothing)
- `CountVectorizer.max_features = 1000` (mặc định trong `main.py`), nhưng nhiều test dùng `max_features=None` hoặc nhỏ hơn
- `DecisionTree.max_depth` được đặt 4 trong `main.py` (một số test dùng 2 hoặc 3)
- `TfidfTransformer.smooth_idf = True` (mặc định)

## 6. Kết quả

Lưu ý quan trọng: kho mã in/ghi các kết quả cho các tập dữ liệu mẫu (toy) được sử dụng trong `main.py` và trong `README.md`. Các con số sau đây phản ánh kết quả trên dữ liệu minh hoạ nhỏ chứ không phải đánh giá trên SST thực tế.

Experimental results (theo nội dung README / main.py chạy trên toy dataset):

| Model | Features | Accuracy |
|-------|----------:|---------:|
| Naive Bayes | Raw Counts | 1.0000 |
| Decision Tree | Raw Counts | 1.0000 |
| Decision Tree | TF–IDF | 1.0000 |

Những giá trị trên được ghi trong README như kết quả khi chạy pipeline với toy dataset (12 mẫu) — không nên hiểu là hiệu năng thực tế trên bộ dữ liệu SST.

Nếu chạy pipeline trên tập dữ liệu lớn (ví dụ [data/raw/sst/train.csv](data/raw/sst/train.csv)), mã hiện tại sẽ xử lý được nhưng cần chú ý giới hạn bộ nhớ do việc chuyển ma trận thưa sang dense trong các mô-đun huấn luyện.

Nếu có confusion matrix (được tính bằng `src/evaluation/confusion_matrix.py`), nó được lưu trong kết quả và dùng để vẽ heatmap bởi `src/evaluation/visualization.py`.

## 7. Thảo luận — Phân tích kỹ thuật sâu

Trong phần này tôi mở rộng phân tích để đề cập các trao đổi kỹ thuật (computational tradeoffs), tính thưa của không gian đặc trưng, hiện tượng curse of dimensionality, cơ chế overfitting của Decision Tree, vì sao Naive Bayes thường biểu hiện tốt trên bài toán NLP, so sánh giữa cài đặt thủ công và abstraction của `scikit-learn`, và giá trị giáo dục khi thực hiện cài đặt từ gốc.

### 7.1 Trao đổi tính toán (Computational tradeoffs)

- Bộ công việc hiện tại tối ưu ở mức thiết kế: lưu trữ ma trận từ văn bản dưới dạng CSR giúp tiết kiệm bộ nhớ cho các thao tác lưu trữ và truyền dữ liệu. Tuy nhiên, các quyết định triển khai bên trong mô hình (chẳng hạn gọi `toarray()` trong `NaiveBayes` và `DecisionTree`) phá hỏng lợi ích này. Kết quả là:
  - Lợi ích của sparse representation: lưu trữ $O(z)$ thay vì $O(nm)$, với $z$ là số phần tử khác không. Phép biến đổi, lưu trữ và một số phép toán đại số (nhân cột bởi idf, cộng đếm) có thể thực hiện trực tiếp trên CSR với chi phí tuyến tính theo $z$.
  - Chi phí thay đổi khi ép sang dense: khi gọi `toarray()` ta tiêu tốn ngay $O(nm)$ bộ nhớ và $O(nm)$ thời gian cho thao tác sao chép, làm cho pipeline không mở rộng tới vocab lớn.

- Thực tế kỹ thuật: hai chiến lược khả thi là (i) giữ nguyên tính sparse và viết toán tử cho NB/DT với sparse (ví dụ tính feature_counts bằng `.sum(axis=0)` trên CSR), hoặc (ii) áp dụng giảm chiều trước khi chuyển sang dense (feature selection / hashing) để kiểm soát $m$.

### 7.2 Không gian đặc trưng thưa và hệ quả kỹ thuật

- Trong biểu diễn BoW, mỗi văn bản chỉ kích hoạt một phần rất nhỏ của từ vựng toàn cục. Điều này gây ra:
  - Lưu trữ: ưu tiên CSR/COO thay vì ma trận dày.
  - Tính toán: nhiều phép toán có thể tối ưu bằng thao tác theo hàng/điểm không-zero; các thuật toán dựa trên duyệt toàn bộ cột (ví dụ tìm ngưỡng cho mỗi feature) trở nên tốn kém.
  - Ảnh hưởng đến mô hình: các thuật toán tuyến tính (Naive Bayes, Logistic Regression) tận dụng được tính thưa (dot-product sparse), trong khi cây rẽ nhánh truyền thống cần so sánh ngưỡng trên từng feature và vì vậy không tận dụng tốt sparsity.

### 7.3 Curse of dimensionality (Lời nguyền chiều cao)

- Khi $m$ (số đặc trưng) tăng, khoảng cách và mật độ dữ liệu trong không gian $\mathbb{R}^m$ hành xử khác hẳn: điểm dữ liệu trở nên cô lập, nguyên tắc thống kê dựa trên láng giềng hay ranh giới đơn giản kém hiệu quả. Hậu quả cụ thể cho NLP BoW:
  - Tỷ lệ non-zero mỗi hàng giảm, khiến ước lượng xác suất các tham số trở nên nhiễu nếu số mẫu cho mỗi đặc trưng quá ít.
  - Cây quyết định có thể tìm được nhiều phép phân tách làm cho entropy giảm mạnh trên tập huấn luyện nhưng không tổng quát được: đây là biểu hiện overfitting.
  - Biện pháp khắc phục thường gặp: regularization (logistic regression with penalty), feature selection, dimensionality reduction (SVD, PCA trên TF–IDF) hoặc embedding.

### 7.4 Overfitting trong Decision Trees — phân tích cơ chế

- Cây quyết định tìm một loạt phép tách để tối đa hóa information gain cục bộ. Với dữ liệu thưa và chiều cao:
  - Mỗi từ có thể mang tín hiệu rời rạc, một node có thể chọn một token hiếm làm điểm phân tách khiến node con có phân phối nhãn gần như đồng nhất trên dữ liệu huấn luyện.
  - Quá trình đòi hỏi kiểm soát: `max_depth`, `min_samples_split` hiện có trong mã là biện pháp cơ bản; nhưng thiếu pruning hậu nghiệm (post-pruning) hoặc regularization tinh vi.
  - Ensemble (Random Forest) giảm phương sai bằng cách trung bình hóa nhiều cây với tính ngẫu nhiên — đây là đề xuất trực tiếp để giảm overfitting so với single tree.

### 7.5 Vì sao Naive Bayes vẫn hiệu quả trong NLP thực hành

- Lý giải kỹ thuật:
  - Mô hình Multinomial NB trực tiếp mô hình hóa tần suất từ conditioned on class — khi dữ liệu là từ đếm, giả thiết về phân phối rời rạc của đếm từ là phù hợp.
  - Công thức log và Laplace smoothing giúp ổn định ước lượng trong miền thưa.
  - NB tối ưu cho các bài toán có nhiều đặc trưng yếu: tổng các đóng góp log P(w|c) tạo ra một chỉ số mạnh mẽ ngay cả khi các token độc lập về mặt thống kê không hoàn toàn đúng trên thực tế.
  - Tính toán đơn giản (đếm + log + dot-product) cho phép chạy nhanh và mở rộng với sparse representations.

### 7.6 So sánh: cài đặt from-scratch vs `scikit-learn` abstraction

- Lợi ích của cài đặt thủ công (như kho mã):
  - Minh bạch thuật toán — sinh viên/engineer nhìn thấy từng bước tính toán, easy to debug and teach.
  - Tính tùy biến cao — dễ sửa đổi thành phần nội bộ (ví dụ thay hàm smoothing, thay cách tính entropy).

- Hạn chế so với `scikit-learn`:
  - Thiếu các tối ưu hóa thấp cấp (C/Fortran-optimized code), sparse-kernel tối ưu, và thuật toán robust production-ready (pruning, early stopping, cross-validated hyperparameter search).
  - `scikit-learn` cung cấp interoperable APIs (fit/transform/predict) cùng pipelines, serialization, và rộng rãi các solver sparse-friendly; sử dụng nó giúp tiết kiệm thời gian engineering.

### 7.7 Giá trị giáo dục của việc cài đặt thủ công

- Việc triển khai từ gốc buộc người học phải giải quyết các vấn đề thực tế: numerical stability (log-space), representation choice (sparse vs dense), algorithmic complexity, và tradeoffs engineering. Những hiểu biết này không thể thu được chỉ bằng cách gọi `fit()` trên một abstraction. Cụ thể:
  - Sinh viên hiểu vì sao Laplace smoothing cần thiết, nguồn gốc của underflow, và cách chuẩn hoá TF–IDF.
  - Việc bắt tay can thiệp vào representation giúp nhận ra điểm mạnh/yếu của từng mô hình và khi nào nên chọn ensemble hay embedding.

Tổng kết: phần mở rộng này nhấn mạnh rằng bài toán NLP cổ điển là một bài toán kỹ thuật sâu — không chỉ chọn mô hình nào cho điểm số cao mà còn là tối ưu representation, thuật toán và quy trình thực thi để đảm bảo hiệu năng và khả năng mở rộng thực tế.

## 8. Phân tích Độ phức tạp

- Với $n$ mẫu và $m$ đặc trưng (vocabulary size):
  - CountVectorizer.fit: đếm token trên toàn bộ corpus $O(\\sum |d_i|)$ (tổng độ dài token), bộ nhớ để lưu `Counter` tạm thời là $O(m)$.
  - Transform sang CSR: mỗi non-zero được lưu, tổng non-zero là $z$, kích thước ma trận thưa là $O(z)$.
  - NaiveBayes.fit (hiện tại): do chuyển sang dense, lưu ma trận $n\\times m$ trong bộ nhớ: $O(nm)$; tính toán feature counts $O(nm)$; inference cost $O(nm)$ (dot-products) nếu không tối ưu.
  - DecisionTree.fit: thuật toán tìm ngưỡng duyệt mọi đặc trưng (m) và mọi ngưỡng khả dĩ (trong worst-case số lượng khác nhau ≤ n) dẫn tới độ phức tạp lớn $O(m n \\log n)$ hoặc tệ hơn cho mỗi node — tổng thể có thể rất đắt đỏ trên không gian đặc trưng lớn.

## 9. Hạn chế (Limitations)

- Một số hạn chế thực tế quan sát được trong kho mã (không phỏng đoán):
  - Mô hình `NaiveBayes` và `DecisionTree` chuyển ma trận thưa sang dense bằng `.toarray()` dẫn tới giới hạn bộ nhớ khi chạy trên tập dữ liệu lớn (SST có hàng chục nghìn câu).
  - `DecisionTree.print_tree()` hữu ích cho debug nhưng chưa phải là cơ chế xuất cây tối ưu cho báo cáo lớn hoặc cây sâu.
  - Pipeline thực nghiệm vẫn phụ thuộc mạnh vào cấu hình trong `experiments/config.yaml`; khi đổi nguồn dữ liệu hoặc cột đầu vào cần kiểm tra lại tên cột và tiền xử lý.
  - Dù `experiments/run_baseline.py` đã thực hiện được thí nghiệm và lưu kết quả, bộ dữ liệu SST thật vẫn có thể cần thêm tối ưu nếu muốn chạy toàn bộ tập với cấu hình mặc định.

## 10. Đề xuất Công việc Tương Lai

- Sửa lỗi và cải thiện:
  - Tránh chuyển toàn bộ ma trận thưa sang dense: thực hiện các phép toán cho NB và DT trực tiếp trên ma trận thưa hoặc dùng sparse-friendly algorithms.

- Mở rộng mô hình:
  - Random Forest / Gradient Boosting để giảm overfitting so với single Decision Tree.
  - Logistic Regression hoặc Linear SVM với tối ưu hoá cho dữ liệu thưa (liblinear / sparse solvers).
  - Thử embedding (Word2Vec / FastText) hoặc fine-tune Transformer (BERT) cho biểu diễn ngữ nghĩa sâu.

- Kỹ thuật khác:
  - Thêm cơ chế pruning cho DecisionTree hoặc sử dụng Early stopping/Min leaf size để giảm overfitting.
  - Kết hợp feature selection (chi2, mutual information) để giảm chiều và tăng hiệu quả.
  - Mở rộng `experiments/run_baseline.py` để xuất thêm báo cáo so sánh per-class metrics và lưu figure vào `report/images/` theo chuẩn trong [report/IMAGES_GUIDE.md](report/IMAGES_GUIDE.md).
  - Chuẩn hóa đầu ra báo cáo cuối cùng thành một quy trình tự động: chạy thí nghiệm, sinh bảng số liệu, và chèn hình minh họa vào `BaoCaoBaiTap.md`.

## 11. Kết luận

Kho mã là một bài tập kỹ thuật giáo dục tốt: các thành phần chính (tiền xử lý, BoW, TF–IDF, Multinomial Naive Bayes, Decision Tree, đánh giá và trực quan hoá) đều được cài đặt rõ ràng từ cơ bản. Điều này giúp người học hiểu mô tả toán học, những quyết định kỹ thuật (ví dụ log-space trong NB), và những thách thức khi làm việc với dữ liệu văn bản thưa. Tuy nhiên, để áp dụng vào thực nghiệm trên SST hay dữ liệu lớn cần sửa một số vấn đề kỹ thuật (xử lý sparse đúng cách, sửa lỗi tiền xử lý, viết script thí nghiệm), đồng thời cân nhắc các mô hình hiện đại hơn hoặc kỹ thuật ensemble để cải thiện tổng quát hoá.

## References

- Các file nguồn chính tham chiếu trong báo cáo:
  - [src/preprocessing/text_processor.py](src/preprocessing/text_processor.py)
  - [src/feature_extraction/count_vectorizer.py](src/feature_extraction/count_vectorizer.py)
  - [src/feature_extraction/tfidf_vectorizer.py](src/feature_extraction/tfidf_vectorizer.py)
  - [src/models/naive_bayes/naive_bayes.py](src/models/naive_bayes/naive_bayes.py)
  - [src/models/decision_tree/decision_tree.py](src/models/decision_tree/decision_tree.py)
  - [src/utils/helper.py](src/utils/helper.py)
  - [src/evaluation/accuracy.py](src/evaluation/accuracy.py)
  - [src/evaluation/confusion_matrix.py](src/evaluation/confusion_matrix.py)
  - [main.py](main.py)
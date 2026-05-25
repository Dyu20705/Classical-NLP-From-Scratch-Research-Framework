# Dive Deep: Sentiment Pipeline, Naive Bayes, and Decision Tree

Tài liệu này giải thích pipeline sentiment trong repo theo cách thực dụng: dữ liệu đi qua đâu, vì sao phải tiền xử lý như vậy, tại sao Naive Bayes thường hợp với text counts, và khi nào Decision Tree cần feature engineering cẩn thận hơn.

## 1. Mục tiêu của bài toán

Bài toán ở đây là phân loại sentiment nhị phân: một câu hoặc một review sẽ được gán vào một trong hai nhãn, thường hiểu là negative hoặc positive. Trong repo này, nhãn gốc có thể là số hoặc chuỗi tùy dataset, nên nguyên tắc đúng là giữ nguyên encoding của dữ liệu và chỉ diễn giải nó nhất quán trong toàn bộ pipeline.

Điểm quan trọng nhất không nằm ở việc chọn model đầu tiên, mà nằm ở việc giữ pipeline không bị leakage: chia train/test trước, fit vectorizer trên train בלבד, sau đó mới transform test.

## 2. Pipeline tổng thể

Luồng thực tế của repo là:

Raw text -> clean text -> tokenization -> stopword removal -> vectorization -> model training -> evaluation

Trong code hiện tại, pipeline này được hiện thực bằng các khối sau:

- `src/preprocessing/pipeline.py` để làm sạch dữ liệu thô và ghi ra bản processed.
- `src/preprocessing/text_processor.py` để tokenize và lọc stopwords.
- `src/feature_extraction/count_vectorizer.py` để biến text thành bag-of-words sparse matrix.
- `src/feature_extraction/tfidf_vectorizer.py` để đổi count matrix sang TF-IDF.
- `src/models/naive_bayes/naive_bayes.py` và `src/models/decision_tree/decision_tree.py` cho hai model chính.
- `src/evaluation/accuracy.py`, `src/evaluation/confusion_matrix.py`, và `src/evaluation/visualization.py` để đo lường kết quả.

## 3. Tiền xử lý: vì sao cần, và cần đến đâu

Tiền xử lý trong repo không phải để làm text “đẹp” hơn một cách mơ hồ. Nó nhằm giảm nhiễu trước khi biến text thành feature rời rạc.

### 3.1 Những bước đang có

- Lowercase để giảm số lượng token trùng nghĩa hình thức.
- Loại HTML, URL, email để bỏ artefact không mang nghĩa sentiment trực tiếp.
- Loại số nếu bài toán không dùng thông tin numeric như rating cụ thể.
- Tokenize bằng whitespace vì pipeline này ưu tiên đơn giản, dễ kiểm soát, và nhất quán với vectorizer từ scratch.
- Remove stopwords để giảm từ chức năng như “the”, “is”, “and”.

Lưu ý thực tế trên snapshot này: một số câu trong SST export vẫn mang tiền tố giống byte-string như `b'...'`. Đây là artefact của nguồn dữ liệu, không phải tín hiệu sentiment. Notebook dùng đúng pipeline hiện tại của repo để bạn nhìn thấy artefact đó đi qua các bước cleaning thay vì che nó đi.

### 3.2 Vì sao không nên tiền xử lý quá tay

Sentiment thường phụ thuộc vào những tín hiệu rất nhỏ như “not good”, “hardly enjoyable”, hoặc “too slow”. Nếu tiền xử lý quá mạnh, bạn dễ làm mất cấu trúc đó.

Ví dụ:

- Giữ lại bigram giúp bắt mẫu phủ định như “not good”.
- Bỏ hết stopword có thể làm mất dấu của phủ định nếu bạn lọc quá agresive.
- Stemming có thể giúp giảm kích thước vocabulary, nhưng cũng có thể làm mờ sắc thái ngôn ngữ.

Nguyên tắc thực dụng là: clean đủ để giảm nhiễu, nhưng không làm text mất nghĩa sentiment.

## 4. Feature engineering: phần quyết định rất lớn đến chất lượng

Với text classification cổ điển, feature engineering thường quan trọng không kém model.

### 4.1 CountVectorizer: bag-of-words thuần count

`CountVectorizer` biến mỗi document thành vector đếm token.

Ưu điểm:

- Đơn giản, dễ giải thích.
- Hợp với Naive Bayes vì NB cần số lần xuất hiện của token để ước lượng xác suất có điều kiện.
- Sparse output giúp tiết kiệm bộ nhớ.

Nhược điểm:

- Từ xuất hiện nhiều chưa chắc quan trọng.
- Từ rất phổ biến có thể lấn át tín hiệu quan trọng nếu không có bước weighting.

### 4.2 TF-IDF: giảm ảnh hưởng của token quá phổ biến

TF-IDF gán trọng số cao hơn cho từ xuất hiện thường xuyên trong một document nhưng hiếm trong toàn corpus.

Trực giác của nó là:

- Term frequency tăng khi token lặp lại trong một câu.
- Inverse document frequency giảm nếu token xuất hiện ở quá nhiều câu.

Hệ quả là những từ rất phổ biến như “movie”, “film”, hoặc các từ rất chung chung sẽ bị giảm trọng số, còn từ mang sắc thái cụ thể sẽ nổi bật hơn.

### 4.3 N-gram: giữ được ngữ cảnh cục bộ

`ngram_range=(1, 2)` cho phép bạn giữ cả unigram và bigram.

Đây là cấu hình rất đáng thử cho sentiment vì:

- “good” và “not good” không có cùng ý nghĩa.
- Bigram giúp model nhìn được cấu trúc phủ định và những cụm sentiment cố định.

Đổi lại, vocabulary tăng nhanh, nên `max_features` phải được kiểm soát.

### 4.4 Max features: giới hạn để không nổ chiều

Text thường có không gian feature cực lớn. Trong notebook, việc cắt `max_features` là cần thiết để giữ pipeline chạy được, nhất là với Decision Tree từ scratch.

Quy tắc thực dụng:

- Naive Bayes có thể chịu feature space lớn hơn.
- Decision Tree nên dùng ít feature hơn vì chi phí split search tăng rất nhanh.

## 5. Naive Bayes: model nào hợp với text counts

Repo đang dùng Multinomial Naive Bayes.

### 5.1 Ý tưởng cốt lõi

Mô hình ước lượng:

P(class | x) proportional to P(class) x product P(token_i | class)^{count_i}

Trong thực tế code dùng log space để tránh underflow.

### 5.2 Vì sao NB thường mạnh cho sentiment text

- Text sparse, high-dimensional, và token counts rất phù hợp với giả định multinomial.
- NB không cần nhiều dữ liệu để học một baseline khá tốt.
- Khi token mang tín hiệu mạnh, NB thường học rất nhanh.

### 5.3 Laplace smoothing

`alpha` trong Naive Bayes là smoothing parameter.

Nó xử lý tình huống một token chưa từng xuất hiện trong class nào đó ở train set. Nếu không smoothing, xác suất có thể bằng 0 và phá hỏng toàn bộ posterior.

### 5.4 Khi NB thất bại

- Khi ngữ cảnh quan trọng hơn tần suất token rời rạc.
- Khi negation và composition ngữ nghĩa phức tạp hơn bag-of-words.
- Khi feature engineering kém, ví dụ bỏ mất bigram hoặc giữ quá ít feature.

### 5.5 Cách đọc kết quả NB

Nếu NB tốt hơn Decision Tree, thường điều đó có nghĩa là:

- Dữ liệu text đủ “bag-of-words friendly”.
- Token counts đang giữ đủ tín hiệu sentiment.
- Tree chưa khai thác được cấu trúc feature do giới hạn độ sâu hoặc feature cap.

## 6. Decision Tree: trực quan nhưng dễ overfit với text

### 6.1 Ý tưởng cốt lõi

Decision Tree chọn feature và threshold để tối đa hóa information gain dựa trên entropy.

Đây là model rất dễ giải thích trên tabular data, nhưng với text sparse thì không phải lúc nào cũng là lựa chọn tự nhiên nhất.

### 6.2 Vì sao tree khó hơn NB trên text

- Text có quá nhiều chiều.
- Nhiều feature hiếm và sparse.
- Split search phải thử nhiều threshold trên từng feature.
- Nếu độ sâu lớn, tree dễ memorization.

Trong repo này, tree là implementation từ scratch nên chi phí càng nhạy cảm hơn.

### 6.3 Vì sao notebook dùng TF-IDF cho tree

TF-IDF giúp làm dịu ảnh hưởng của token quá phổ biến và đưa feature về dạng có cân nhắc trọng số tốt hơn.

Nhưng cần hiểu đúng: TF-IDF không làm tree “thành model text lý tưởng”. Nó chỉ làm feature giàu thông tin hơn và thường ổn hơn count thô trong một số trường hợp.

### 6.4 Khi nào tree hợp lý

- Khi bạn muốn diễn giải split một cách trực quan.
- Khi feature space đã được giới hạn hợp lý.
- Khi bạn chủ động chấp nhận độ chính xác thấp hơn để đổi lấy khả năng giải thích hoặc học pipeline cơ bản.

### 6.5 Khi nào tree không phải lựa chọn tốt

- Khi feature space quá rộng.
- Khi dữ liệu text lớn và sparse.
- Khi bạn cần baseline mạnh, ổn định, và ít tuning hơn Naive Bayes.

## 7. Cách lựa chọn pipeline cho từng model

### 7.1 Cho Naive Bayes

Khuyến nghị thực dụng:

- Clean vừa đủ.
- Dùng CountVectorizer.
- Thử `ngram_range=(1, 2)` nếu dữ liệu đủ lớn.
- Giữ `max_features` ở mức vừa phải đến khá cao.
- Tune `alpha` nếu muốn cân bằng giữa smoothing và độ nhạy.

### 7.2 Cho Decision Tree

Khuyến nghị thực dụng:

- Clean giống NB để so sánh công bằng ở bước đầu.
- Dùng ít feature hơn.
- Cân nhắc TF-IDF thay vì count thô.
- Giữ `max_depth` thấp để tránh overfitting và giảm thời gian train.
- Đừng cho tree vào feature space quá lớn nếu implementation là from scratch.

## 8. Cách đọc confusion matrix đúng cách

Confusion matrix trả lời câu hỏi rất cụ thể:

- Model đang nhầm positive thành negative bao nhiêu lần?
- Model đang nhầm negative thành positive bao nhiêu lần?

Trong sentiment, hai loại lỗi này có thể không cân xứng về hậu quả. Nếu bạn quan tâm đến trải nghiệm người dùng, false negative và false positive có thể mang ý nghĩa khác nhau.

Vì vậy, đừng nhìn accuracy một mình. Hãy đọc thêm confusion matrix để hiểu lỗi thật sự nằm ở đâu.

## 9. Cách dùng notebook `notebooks/test.ipynb`

Notebook này được thiết kế theo các bước sau:

1. Tìm project root và import đúng module từ repo.
2. Đọc raw data từ `data/raw/sst/train.csv`.
3. Clean và lưu processed data ra `data/processed/`.
4. Split train/test trước khi fit vectorizer.
5. Train Naive Bayes trên count features.
6. Train Decision Tree trên TF-IDF features có feature budget nhỏ hơn.
7. So sánh accuracy, confusion matrix, và ví dụ dự đoán sai.

Nếu notebook chạy chậm, ưu tiên giảm `MAX_ROWS`, giảm `max_features` cho tree, hoặc giảm `max_depth`.

## 10. Kết luận thực dụng

Nếu mục tiêu của bạn là một baseline sentiment cổ điển, thì:

- Naive Bayes thường là lựa chọn đầu tiên tốt nhất vì nhanh, ổn định, và hợp với text counts.
- Decision Tree hữu ích để học trực giác về split và entropy, nhưng không phải model mặc định mạnh cho NLP sparse.
- Tiền xử lý phải đủ sạch nhưng không nên phá mất cấu trúc sentiment như negation.
- Feature engineering, đặc biệt là n-gram, max_features, và lựa chọn giữa count/TF-IDF, thường quyết định chất lượng nhiều hơn bạn nghĩ.

Nếu muốn đi xa hơn, bước tiếp theo hợp lý là so sánh baseline này với Logistic Regression hoặc Linear SVM trên cùng pipeline để xem classical linear models thắng tree bao nhiêu trong bài toán text classification.
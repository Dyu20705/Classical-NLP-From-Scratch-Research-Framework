# Research Upgrade Plan (Current State + Next Actions)

## Mục tiêu

Chuyển repo từ trạng thái implementation collection sang scientific investigation artifact với 4 trụ cột:

- Methodology rõ ràng
- Experiment discipline
- Reproducibility mạnh
- Analysis có chiều sâu

## Current State Snapshot (2026-05-28)

### Hoàn thành

- Research framing trong README đã chuyển sang hướng framework nghiên cứu cổ điển NLP:
  - [README.md](README.md)
- Đã có metadata tracking tự động cho mỗi run:
  - [main.py](main.py)
  - [experiments/run_baseline.py](experiments/run_baseline.py)
  - [src/utils/experiment_tracking.py](src/utils/experiment_tracking.py)
- Đã có script tổng hợp benchmark từ artifacts:
  - [scripts/aggregate_results.py](scripts/aggregate_results.py)
- Đã có script chạy ablation có kiểm soát:
  - [scripts/run_ablations.py](scripts/run_ablations.py)
- Đã sửa lỗi CI import/dependency gần nhất:
  - thêm seaborn trong [requirements.txt](requirements.txt)
  - phục hồi [src/feature_extraction/vocabulary.py](src/feature_extraction/vocabulary.py)

### Đang thiếu

- Chưa có bộ benchmark thật (được aggregate từ nhiều run) đưa vào README.
- Chưa có kết quả ablation thật (stopword/ngram/max_features) để kết luận có bằng chứng.
- Chưa có section error analysis dựa trên prediction artifacts.
- Chưa có profiling runtime/memory sparse vs dense dạng report.
- Chưa có technical report dạng paper trong docs/paper.

## Phase Roadmap (Updated)

## Phase 1: Scientific Framing

Status: Completed

Deliverables đã đạt:

- README có Abstract, Motivation, Research Questions, Methodology, Reproducibility, Limitations.
- Scope chính đã rõ: sparse classical NLP baseline.
- Bộ metric chính đã chuẩn hóa: Accuracy, Precision, Recall, Macro F1, Confusion Matrix.

Acceptance evidence:

- [README.md](README.md)

## Phase 2: Experiment Discipline + Tracking

Status: In Progress (70%)

Deliverables đã đạt:

- Output theo experiment run id trong luồng config-driven.
- Metadata per-run gồm commit hash, python version, timestamp, dataset checksum, config path.
- Benchmark aggregation script đã có.

Deliverables còn thiếu:

- Chạy thực tế nhiều run để tạo benchmark_summary.csv/md có ý nghĩa.
- Chuẩn hóa thêm metadata policy cho tất cả run paths (CLI + experiments + ablations) trong CI.

Acceptance criteria:

- Mỗi run có đủ metadata và có thể truy vết.
- Có benchmark summary sinh tự động từ artifacts thật.

## Phase 3: Analysis Depth

Status: Not Started

Deliverables mục tiêu:

- Error analysis theo category (negation, sarcasm, long dependency, OOV).
- Ablation tối thiểu:
  - stopword on/off
  - n-gram (1,1), (1,2), (1,3)
  - max_features (1000, 5000, 10000)
- Runtime/memory profiling sparse vs dense.

Acceptance criteria:

- Có section What we discovered với kết luận dựa trên dữ liệu.
- Có ít nhất 3 biểu đồ phân tích.

## Phase 4: Research-style Report

Status: Not Started

Deliverables mục tiêu:

- Báo cáo kỹ thuật dạng paper trong docs/paper:
  - Abstract
  - Introduction
  - Methodology
  - Experiments
  - Results
  - Discussion
  - Limitations
  - Future Work
- Bảng benchmark chính thức và reproducibility appendix.

Acceptance criteria:

- Reviewer ngoài dự án đọc được claim, setup, kết quả và giới hạn.
- Dùng được cho portfolio nghiên cứu/internship.

## Experiment Discipline Checklist (Operational)

Mỗi lần chạy experiment bắt buộc:

- [ ] Ghi hypothesis trong notes của run.
- [ ] Đóng băng config và seed.
- [ ] Lưu đầy đủ artifacts + metadata.
- [ ] Aggregate vào benchmark summary.
- [ ] Viết interpretation ngắn (không chỉ ghi số).

## Scientific Writing Checklist

- [ ] Nêu claim có điều kiện, không overclaim.
- [ ] Tách observation và interpretation.
- [ ] Có failure analysis cho model tốt nhất.
- [ ] Có section limitations rõ ràng.

## Scope Guardrail

Ưu tiên chiều sâu hơn độ rộng:

- Focus chính: Naive Bayes, Decision Tree, Count/TF-IDF, sparse NLP.
- Model mở rộng chỉ dùng khi phục vụ câu hỏi nghiên cứu.

Nguyên tắc:

- Ít model hơn nhưng phân tích sâu hơn.
- Ít claim hơn nhưng bằng chứng chắc hơn.

## Next Actions (Dự án hiện tại)

## Priority 1: Chạy benchmark thật và aggregate

Mục tiêu:

- Tạo 5-10 runs trong results/experiments/exp_xxx.
- Sinh benchmark tổng hợp bằng script.

Done khi:

- Có [results/experiments/benchmark_summary.csv](results/experiments/benchmark_summary.csv)
- Có [results/experiments/benchmark_summary.md](results/experiments/benchmark_summary.md)

## Priority 2: Chạy ablation batch đầu tiên

Mục tiêu:

- Chạy ablation stopword/ngram/max_features trên cùng dataset slice.

Done khi:

- Có [results/ablations/ablation_summary.csv](results/ablations/ablation_summary.csv)
- Có [results/ablations/ablation_summary.md](results/ablations/ablation_summary.md)

## Priority 3: Cập nhật README bằng evidence thật

Mục tiêu:

- Thay kết quả mẫu/range bằng số liệu thật từ benchmark + ablation.

Done khi:

- Section Key Findings trong [README.md](README.md) chỉ dùng số liệu từ artifacts.

## Priority 4: Error Analysis vòng 1

Mục tiêu:

- Tạo tài liệu phân tích lỗi đầu tiên từ predictions artifacts.

Done khi:

- Có file docs mới (ví dụ docs/ERROR_ANALYSIS.md) với ít nhất 15-20 mẫu lỗi có phân loại.

## Priority 5: Systems Profiling sơ bộ

Mục tiêu:

- Ghi runtime và memory cho một số cấu hình điển hình (NB vs DT, ngram 1 vs 2).

Done khi:

- Có bảng profiling trong docs và liên kết từ README.

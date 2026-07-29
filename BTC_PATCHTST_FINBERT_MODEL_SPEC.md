# Đặc tả mô hình dự báo RV Bitcoin bằng PatchTST, sentence embedding và FinBERT sentiment

## 1. Mục tiêu bài toán

Với mỗi ngày UTC \(t\), dự báo kỳ vọng có điều kiện của Realized Variance:

\[
\widehat{RV}_t = E[RV_t \mid \mathcal F_{t-1}]
\]

Realized Variance ngày được tính từ log-return intraday 5 phút:

\[
r_{t,i} = \log P_{t,i} - \log P_{t,i-1}
\]

\[
RV_t = \sum_{i=1}^{288} r_{t,i}^{2}
\]

Target dùng để biểu diễn đầu ra là:

\[
y_t = \log RV_t
\]

Mô hình xuất:

\[
\widehat{y}_t = \widehat{\log RV}_t
\]

và dự báo trên thang RV là:

\[
\widehat{RV}_t = \exp(\widehat{y}_t)
\]

Thời điểm dự báo là đầu ngày \(t\). Toàn bộ dữ liệu đầu vào chỉ được sử dụng đến hết ngày \(t-1\).

Mô hình học sâu dự báo trực tiếp, không sử dụng dự báo HAR hoặc các đặc trưng `logRV_d`, `logRV_w`, `logRV_m` bên trong kiến trúc chính.

---

## 2. Dữ liệu

### 2.1. Dữ liệu thị trường

File:

```text
data/BTCUSDT_5min_2018_2025_present.csv
```

Phạm vi sử dụng:

```text
2018-01-01 00:00 UTC
đến
2025-06-30 23:55 UTC
```

Chỉ giữ một ngày nếu thỏa mãn, sau khi áp dụng duy nhất chính sách bảo trì đã khóa tại §2.1.1:

- Có đủ 288 nến 5 phút.
- Có close của nến `23:55` ngày trước để tạo return đầu tiên lúc `00:00`.
- Có đủ 288 log-return liên tục theo lưới 5 phút.
- Không có khoảng trống intraday ngoài các no-trade bar bảo trì đã xác minh tại §2.1.1.
- \(RV_t > 0\).

Return được tính trên chuỗi close toàn cục trước khi group theo UTC date:

\[
r_{t,00{:}00}
=
\log P_{t,00{:}00}
-
\log P_{t-1,23{:}55}
\]

và tương tự cho các nến còn lại. Một ngày có 288 close nhưng chỉ sai phân bên trong ngày sẽ chỉ có 287 return và không đúng định nghĩa này. Ngày \(t\) chỉ hợp lệ khi chuỗi từ close `23:55` của \(t-1\) đến close `23:55` của \(t\) liên tục đúng 5 phút sau verified-maintenance fill. Loader được phép giữ nến `2017-12-31 23:55` chỉ làm boundary bar cho ngày đầu phạm vi; nó không tạo target ngoài giai đoạn nghiên cứu.

#### 2.1.1. Khoảng đóng cửa bảo trì đã xác minh

Audit dữ liệu xác nhận hai khoảng thiếu nến trùng chính xác với thời gian Binance tạm dừng toàn bộ spot trading để nâng cấp hệ thống:

```text
2021-08-13 02:00 – 06:25 UTC: 54 bar
2021-09-29 07:00 – 08:55 UTC: 24 bar
```

Nguồn đối chiếu lịch bảo trì:

- https://bitnoticias.com.br/noticias/exchange-binance-realiza-atualizacao-e-trava-negociacoes-por-no-minimo-4-horas/
- https://help.bituniverse.org/announcements/2021-09-29

Đây là hai khoảng duy nhất được phép tạo no-trade bar. Với từng timestamp 5 phút bị thiếu trong các khoảng trên:

```text
Open = High = Low = Close = close quan sát cuối cùng ngay trước khi đóng cửa
Volume = Quote Asset Volume = 0
Number of Trades = 0
Taker Buy Base Asset Volume = 0
Taker Buy Quote Asset Volume = 0
```

Quy tắc này nhân quả: không được đọc giá mở cửa lại hoặc bất kỳ quan sát tương lai nào để dựng bar bảo trì. Return trong thời gian đóng cửa bằng 0; nến thực đầu tiên sau khi mở lại nhận toàn bộ biến động từ close cuối trước bảo trì. Loader chỉ chèn timestamp thực sự thiếu, không ghi đè bar có sẵn.

Mọi khoảng thiếu khác vẫn làm ngày không hợp lệ theo quy tắc gốc; không forward-fill chung, không nội suy và không chuyển thành “60 ngày hợp lệ gần nhất”. Loader phải xuất audit theo từng interval gồm số bar dự kiến, số bar đã có, số bar tổng hợp, timestamp và giá close tham chiếu. `maintenance_synthetic` chỉ là mask audit, không phải channel/feature của mô hình.

Các cột intraday dự kiến sử dụng:

- `Open`
- `High`
- `Low`
- `Close`
- `Volume`
- `Number of Trades`
- `Taker Buy Base Asset Volume`

`Taker Buy Quote Asset Volume` chỉ được đọc để QC tính nhất quán với quote volume/giá, không tạo channel hay feature mô hình. Kênh taker-buy tại §7 chỉ dùng `Taker Buy Base Asset Volume`.

### 2.2. Dữ liệu tin tức

Tên file thực tế:

```text
data/news_clusters.json
```

Phạm vi:

```text
2018-01-01
đến
2025-06-30
```

Các trường được phép dùng để tạo feature:

- `canonical_publication_time`
- `canonical_title`
- `canonical_article_text`
- `canonical_source`
- `news_cluster_id`

Các trường cluster hồi cứu chỉ dùng cho audit, không dùng làm feature/trọng số:

- `source_count`
- `member_count`
- `republication_offsets_minutes`
- `all_sources`, `all_urls` và các danh sách member/variant

`canonical_publication_time` phải được chuẩn hóa về UTC trước khi lọc và tổng hợp.

Lý do: cluster có thể nhận thêm member sau canonical time. Ví dụ audit hiện tại có cluster `news_1f3c4dcc5ebd0c7ca464`, canonical time `2021-06-17`, nhưng một bản đăng lại xuất hiện sau 2.031 phút. Dùng `source_count/member_count` cuối cùng tại ngày canonical sẽ đưa thông tin tương lai vào feature. Nếu sau này có snapshot point-in-time thì mới được tái dựng các count này theo cutoff; với file hiện tại phải loại.

---

## 3. Làm sạch và lọc tin liên quan Bitcoin

Không tìm từ khóa trên toàn bộ nội dung chưa làm sạch. Nhiều nguồn có footer hoặc danh sách bài đề xuất chứa từ Bitcoin dù nội dung chính không liên quan.

### 3.1. Làm sạch nội dung

Cắt bỏ hoặc loại bỏ:

- Nội dung sau `More From The Motley Fool`.
- Nội dung sau `More from Fortune.com`.
- Nội dung sau `See original article`.
- Related stories.
- Disclosure.
- Author biography.
- Danh sách bài đề xuất.
- HTML tags.
- Ký tự lỗi và khoảng trắng bất thường.

### 3.2. Từ khóa trực tiếp

```python
PRIMARY_KEYWORDS = [
    r"\bbitcoin\b(?!\s+cash)",
    r"\bbtc\b",
    r"\bxbt\b",
]
```

Từ khóa loại riêng:

```python
BITCOIN_CASH_TERMS = [
    r"\bbitcoin cash\b",
    r"\bbch\b",
]
```

### 3.3. Thuật ngữ đặc thù Bitcoin

```python
BITCOIN_SPECIFIC_TERMS = [
    r"\bsatoshi nakamoto\b",
    r"\bsatoshis?\b",
    r"\blightning network\b",
    r"\bbitcoin core\b",
    r"\bsegwit\b",
    r"\btaproot\b",
    r"\bbitcoin halv(?:ing|en)\b",
    r"\bbitcoin miners?\b",
    r"\bbitcoin mining\b",
    r"\bbitcoin etf\b",
    r"\bspot bitcoin etf\b",
    r"\bbitcoin futures?\b",
    r"\bmt\.?\s*gox\b",
]
```

### 3.4. Luật giữ tin

Giữ một cụm tin nếu thỏa mãn ít nhất một điều kiện:

1. Tiêu đề chứa `Bitcoin`, `BTC` hoặc `XBT`.
2. Tiêu đề chứa thuật ngữ đặc thù Bitcoin.
3. Phần đầu nội dung sạch chứa từ khóa Bitcoin ít nhất hai lần.
4. Từ khóa Bitcoin xuất hiện trong cả tiêu đề và nội dung.
5. Nội dung có thuật ngữ đặc thù Bitcoin rõ ràng.

Không giữ nếu:

- Chỉ có các từ chung như `crypto`, `cryptocurrency` hoặc `blockchain`.
- Chỉ nói về ETH, XRP hoặc altcoin.
- Bitcoin chỉ xuất hiện trong footer hoặc related links.
- Chỉ chứa `Bitcoin Cash` hoặc `BCH` và không có từ khóa BTC độc lập.

### 3.5. Điểm liên quan Bitcoin

Có thể tạo:

```text
bitcoin_relevance_score
```

| Điều kiện | Điểm |
|---|---:|
| Bitcoin/BTC trong tiêu đề | +3 |
| Bitcoin/BTC trong 500 ký tự đầu | +2 |
| Thuật ngữ Bitcoin đặc thù | +2 |
| Từ khóa xuất hiện ít nhất ba lần | +1 |
| Chỉ xuất hiện sau 2.000 ký tự | 0 |
| Chỉ xuất hiện trong footer | Loại |

Ngưỡng khởi đầu:

```text
bitcoin_relevance_score >= 2
```

Ngưỡng này cần được kiểm tra thủ công trên một mẫu tin chỉ lấy từ development period trước khi chạy embedding toàn bộ; không điều chỉnh luật lọc dựa trên tin hoặc kết quả dự báo trong final test.

---

## 4. Semantic embedding và FinBERT sentiment

Không dùng hidden state mean-pooling của `ProsusAI/finbert` làm semantic sentence embedding chính. FinBERT được giữ cho nhiệm vụ mà nó đã được fine-tune trực tiếp: phân loại sentiment tài chính.

### 4.1. Semantic sentence encoder

Model chính:

```text
BAAI/bge-base-en-v1.5
```

Cấu hình:

- Tối đa 512 token.
- Embedding 768 chiều.
- Freeze và chạy offline.
- Pool bằng hidden state của token đầu tiên (`last_hidden_state[:, 0]`, CLS), sau đó chuẩn hóa L2 embedding đầu ra.

Nếu dùng `SentenceTransformer.encode`, phải khóa `normalize_embeddings=True` và assert output 768 chiều; không được âm thầm đổi sang mean pooling giữa các lần chạy.

Nguồn:

- https://huggingface.co/BAAI/bge-base-en-v1.5
- https://huggingface.co/BAAI/bge-base-en-v1.5/blob/main/config.json

Model ablation:

```text
intfloat/e5-base-v2
```

E5 cũng tạo embedding 768 chiều; nếu dùng E5 thì input cần tuân theo prefix được model card yêu cầu, ví dụ `passage:`.

### 4.2. Ghép tiêu đề và nội dung

Đầu vào semantic encoder:

```text
Title: <canonical_title>
Content: <cleaned canonical_article_text>
```

Tổng chuỗi sau tokenization tối đa 512 token, đã bao gồm special tokens. Nội dung bị cắt ở phần cuối; tiêu đề luôn phải được giữ.

Không dùng 768 token đầu vào. Trong các model base nêu trên, 768 là chiều embedding, không phải độ dài chuỗi.

### 4.3. FinBERT sentiment

Model:

```text
ProsusAI/finbert
```

FinBERT được freeze và chạy offline trên cùng `title + cleaned content lead`, tối đa 512 token.

Chỉ lấy ba xác suất:

- `positive_probability`
- `negative_probability`
- `neutral_probability`

Không lấy mean-pooled hidden state của FinBERT làm semantic embedding chính.

Nguồn:

- https://huggingface.co/ProsusAI/finbert/blob/main/config.json
- https://huggingface.co/docs/transformers/model_doc/bert

Với bài \(i\):

\[
e_i^{semantic} \in \mathbb R^{768}
\]

\[
s_i =
[p_i^{pos},p_i^{neg},p_i^{neu}]
\in \mathbb R^3
\]

Hai nhóm feature này được xử lý riêng. Sentiment không được ghép vào embedding trước PCA.

---

## 5. Tổng hợp tin theo ngày

Nếu ngày \(t\) có \(n_t\) cụm tin:

\[
\bar e_t =
\frac{\sum_{i=1}^{n_t} w_i e_i^{semantic}}
{\sum_{i=1}^{n_t} w_i}
\]

Trọng số:

\[
w_i =
bitcoin\_relevance\_score_i
\]

Không dùng `source_count_i`, `member_count_i` hoặc republishing offsets trong \(w_i\), vì đây là trạng thái cuối của cluster và có thể chứa các member xuất hiện sau ngày canonical.

Sentiment ngày dùng cùng trọng số, không lấy sentiment của một bài đại diện:

\[
sentiment_t
=
\frac{\sum_{i=1}^{n_t} w_i
[p^{pos}_i,p^{neg}_i,p^{neu}_i]}
{\sum_{i=1}^{n_t} w_i}
\]

Nếu ngày không có tin:

```python
daily_semantic_embedding_t = missing
no_news_t = 1
news_count_t = 0
```

Không đặt zero vector trong không gian embedding thô 768 chiều và không forward-fill embedding của ngày trước.

Các scalar news feature:

- `news_count`
- `canonical_source_count`: số `canonical_source` phân biệt trong các canonical cluster có timestamp thuộc ngày đó
- `negative_ratio`: tỷ lệ bài có nhãn argmax FinBERT là `negative`
- `negative_count_070`: số bài có `negative_probability > 0.70`
- `negative_probability_max`
- `negative_probability_std`
- `positive_probability_max`
- `sentiment_entropy_mean`
- `semantic_dispersion`: khoảng cách trung bình của embedding bài với centroid ngày
- `mean_relevance`
- `no_news_dummy`

Các feature độ phân tán được giữ vì trung bình embedding có thể làm mất bất đồng giữa các bài trong những ngày nhiều tin.

Định nghĩa chính xác:

\[
negative\_ratio_t
=
\frac{1}{n_t}
\sum_{i=1}^{n_t}
\mathbf 1
\{
\arg\max(p^{pos}_i,p^{neg}_i,p^{neu}_i)=neg
\}
\]

`negative_probability_std` dùng population standard deviation (`ddof=0`); nếu \(n_t=1\) thì bằng 0. `sentiment_entropy_mean` là weighted mean entropy của ba xác suất với trọng số \(w_i\). `semantic_dispersion` là weighted mean cosine distance từ embedding bài đến centroid ngày. `mean_relevance` là arithmetic mean của `bitcoin_relevance_score`.

Trong kiến trúc chính, toàn bộ danh sách trên được dùng trong `daily_scalar_t`, ngoại trừ raw `news_count` được thay bằng `news_intensity` tại §5.1. `canonical_source_count` và `negative_count_070` được biến đổi bằng `log1p`; các feature liên tục còn lại được giữ trên thang tự nhiên trước fold-specific scaling.

Với ngày không có tin:

- `news_intensity` vẫn được tính hợp lệ từ `news_count=0`.
- Các scalar sentiment/dispersion không xác định được gán missing, impute bằng median của core train rồi mới transform bằng scaler của fold; sau transform đặt chúng bằng 0, tức giá trị trung tính trong không gian đã chuẩn hóa.
- `no_news_dummy=1`.

Không đưa raw zero của các scalar không xác định qua scaler, vì nó có thể trở thành một điểm giả xa tâm train.

### 5.1. News-count drift và chuẩn hóa nhân quả

Audit sơ bộ của bộ lọc hiện tại cho thấy `news_count` thay đổi mạnh theo năm:

| Năm | Median bài/ngày | Mean bài/ngày |
|---|---:|---:|
| 2018 | 10 | 10,91 |
| 2019 | 15 | 17,22 |
| 2020 | 11 | 10,56 |
| 2021 | 33 | 31,75 |
| 2022 | 26 | 25,89 |
| 2023 | 17 | 17,36 |
| 2024 | 19 | 19,14 |
| 2025 đến 30/06 | 22 | 20,97 |

Hai dòng có ngày sau 2024-04-16 chỉ là QC mô tả sau khi đã khóa quy tắc; không được dùng để chọn cửa sổ hoặc threshold. Quyết định dùng rolling median 365 ngày phải được tái hiện trên development period.

Vì vậy không dùng raw `news_count` trực tiếp. Tạo:

\[
c_j=\log(1+news\_count_j)
\]

\[
news\_intensity_j =
c_j -
median(c_{j-364:j})
\]

Rolling median chỉ dùng lịch sử đến ngày đang tạo feature, nên vẫn nhân quả. Cửa sổ được khóa là 365 ngày lịch bao gồm ngày \(j\): `rolling(window=365, min_periods=30).median()`. Trước khi đủ 30 ngày, dùng expanding median từ `2018-01-01` đến \(j\). Với target ngày \(t\), model chỉ nhận `news_intensity_j` cho \(j\le t-1\); không dùng count của chính ngày \(t\).

Trước khi train phải xuất:

- Đồ thị `news_count` theo ngày và theo tháng.
- Median, mean, P90 và maximum theo tháng/năm.
- Tỷ lệ no-news theo tháng/năm.
- Số `canonical_source` phân biệt và tỷ trọng từng canonical source theo tháng/năm; không dùng member source hồi cứu.
- Cosine drift của monthly semantic centroid.

`canonical_publication_time` được giả định là thời điểm bài được xuất bản, nhưng aggregator có thể lưu thời điểm crawl hoặc nội dung đã được cập nhật sau khi đăng. Không thể loại bỏ hoàn toàn rủi ro này từ schema hiện tại; phải ghi nó thành limitation và audit các timestamp bất thường.

Ngay cả việc chọn canonical title/text và hình thành cluster cũng có thể được thực hiện hồi cứu. Việc cấm member/count metadata loại được đường leakage trực tiếp đã thấy, nhưng không biến dataset thành snapshot point-in-time hoàn hảo; kết quả phải nêu rõ giới hạn này.

---

## 6. PCA và phân rã slow/fast

Semantic embedding 768 chiều quá lớn so với số ngày huấn luyện.

Trong mỗi walk-forward fold:

1. Chỉ lấy các ngày thuộc phần train thực sự có tin để fit semantic scaler.
2. Fit PCA chỉ trên semantic embedding của phần train, không dùng validation hoặc test.
3. Giảm xuống 8 hoặc 16 thành phần.
4. Dùng đúng scaler/PCA của fold để transform mọi ngày có tin từ ngày bắt đầu cố định của chuỗi đến hết test fold.
5. Tính lại toàn bộ recursion slow/fast từ ngày bắt đầu cố định trong cơ sở PCA của chính fold đó.

Gọi vector sau PCA là:

\[
Z_t \in \mathbb R^k
\]

với \(k \in \{8,16\}\).

Ba xác suất sentiment không đi qua semantic PCA. Chuỗi sentiment ngày được phân rã slow/fast riêng bằng cùng quy tắc recursion. Các scalar dispersion không phân rã; chúng được chuẩn hóa riêng bằng scaler của train fold.

### 6.1. Tái tính theo từng fold

PCA của các fold có components khác nhau, nên \(Z_t\), `slow_t` và `fast_t` của fold này không được tái sử dụng cho fold khác.

Với mỗi fold \(f\):

```text
fit scaler_f/PCA_f trên core train_f
→ transform lịch sử từ 2018-01-01 đến hết test_f
→ khởi tạo lại state tại 2018-01-01
→ chạy recursion tuần tự đến hết test_f
```

Quy tắc bắt buộc:

- Recursion luôn bắt đầu tại cùng ngày lịch `2018-01-01` cho mọi fold.
- Khi đi từ train sang validation và test, trạng thái `slow` tiếp tục tuần tự; không reset tại ranh giới fold segment.
- Feature dự báo ngày \(t\) chỉ lấy trạng thái đã cập nhật đến hết \(t-1\).
- Khi bước qua test block, news quan sát được ở ngày test trước được phép cập nhật state cho dự báo một bước trước của ngày test sau.

Lưu SHA-256 của:

```text
semantic scaler parameters
PCA components
PCA mean
PCA explained variance
PCA n_components
```

vào metadata feature. Hash dùng để sinh slow/fast phải khớp với hash preprocessor của model checkpoint trong cùng fold.

### 6.2. PCA drift diagnostic

Với standardized semantic vector \(x_i^*\), PCA mean \(\mu_f\) và projection \(P_{f,k}\) của fold \(f\), báo cáo captured variance ratio trên block \(B\):

\[
CVR_f(B)
=
\frac{
\sum_{i\in B}
\left\|
P_{f,k}(x_i^*-\mu_f)
\right\|_2^2
}{
\sum_{i\in B}
\left\|
x_i^*-\mu_f
\right\|_2^2
}
\]

Tính \(CVR_f\) riêng trên core train, validation và test, chỉ với ngày có tin, bằng đúng scaler/PCA core-train. Báo cả `CVR_test`, `CVR_core` và:

\[
relative\_drop_f
=
1-\frac{CVR_f(test)}{CVR_f(core)}
\]

Đây chỉ là diagnostic drift. Không refit PCA trong test và không đổi \(k\) sau khi xem final test. Với final block từ 2024-04-17 đến 2025-06-30 là 440 ngày lịch, mức giảm mạnh phải được nêu như limitation của cơ sở PCA đã fit đến 2024-01-17.

### 6.3. Khởi tạo và ngày có tin

Trước ngày có tin đầu tiên:

```python
slow_semantic = zero_vector_in_fold_pca_space
slow_sentiment = zero_vector
initialized = False
```

Tại ngày có tin đầu tiên, khởi tạo:

```python
slow_semantic = Z_t
slow_sentiment = sentiment_t
fast_semantic = zero_vector
fast_sentiment = zero_vector
initialized = True
```

Từ ngày có tin tiếp theo:

\[
slow^{semantic}_t =
(1-\alpha)slow_{t-1}+\alpha Z_t,
\qquad
\alpha=\frac{2}{31}
\]

\[
fast^{semantic}_t = Z_t - slow^{semantic}_t
\]

\[
slow^{sentiment}_t =
(1-\alpha)slow^{sentiment}_{t-1}
+\alpha sentiment_t
\]

\[
fast^{sentiment}_t =
sentiment_t-slow^{sentiment}_t
\]

### 6.4. Quy tắc ngày không có tin

Ngày không có tin không có semantic observation hợp lệ. Do đó:

```python
slow_semantic_t = slow_semantic_previous
slow_sentiment_t = slow_sentiment_previous
fast_semantic_t = zero_vector
fast_sentiment_t = zero_vector
no_news_t = 1
```

Không transform zero vector thô qua scaler/PCA và không cho ngày không tin kéo trạng thái `slow` về zero.

Quy tắc no-news ở trên được khóa từ nguyên tắc point-in-time, không được chọn theo tỷ lệ của final test. Trước khi train phải xuất riêng audit development-only đến 2024-04-16. Con số toàn mẫu dưới đây chỉ là post-lock QC mô tả:

```text
Calendar days: 2.738
Days with at least one relevant article: 2.690
No-news days: 48
No-news rate: 1,75%
```

Con số này phải được tính lại sau khi khóa bộ lọc cuối cùng và không được dùng để sửa luật lọc, cách impute hoặc kiến trúc.

### 6.5. Tính chất và căn thời gian

- Nhân quả theo thứ tự thời gian.
- Trên ngày có tin sau khởi tạo, `slow_semantic + fast_semantic == Z` và `slow_sentiment + fast_sentiment == sentiment` tới độ chính xác máy.
- Trên ngày không tin, `slow` giữ trạng thái và `fast=0`; đẳng thức phân hoạch không áp dụng vì không có \(Z_t\) quan sát được.
- Transient bắt đầu tại cùng ngày có tin đầu tiên cho mọi fold, dù tọa độ PCA khác nhau.

Với dự báo ngày \(t\), chỉ dùng:

```text
semantic_slow[t-30:t-1]
semantic_fast[t-30:t-1]
sentiment_slow[t-30:t-1]
sentiment_fast[t-30:t-1]
```

Không đưa đồng thời `Z`, `slow` và `fast` vào mạng.

`no_news_dummy` luôn được giữ để phân biệt `fast=0` do không có tin với một surprise vector thực sự gần zero.

---

## 7. Đặc trưng intraday

Các kênh đầu vào dự kiến:

```text
log_return
log_squared_return
log_high_low_range
log1p_volume
log1p(number_of_trades)
taker_buy_ratio
taker_buy_imbalance
```

Ví dụ:

\[
log\_range_i = \log(High_i / Low_i)
\]

\[
log\_squared\_return_i =
\log(r_i^2+\epsilon_r),
\qquad \epsilon_r=10^{-12}
\]

```python
log1p_volume = np.log1p(volume)
zero_volume = volume <= 0

taker_buy_ratio = np.where(
    zero_volume,
    0.5,
    taker_buy_base_volume / volume,
)
taker_buy_imbalance = 2.0 * taker_buy_ratio - 1.0
```

Khi `Volume=0`, tỷ lệ mua chủ động được đặt về giá trị trung tính 0,5 và imbalance bằng 0.

Audit development period từ 2018-01-01 đến 2024-04-16 phải tách zero-volume quan sát thực và no-trade bar bảo trì tổng hợp. Với file hiện tại sau chính sách §2.1.1:

```text
Valid calendar days: 2.269
Bars in valid days: 653.472
Organic zero-volume bars: 0
Verified-maintenance synthetic bars: 78
Total zero-volume bars: 78
```

Không dùng `zero_volume_mask` hoặc `maintenance_synthetic` như channel riêng. Đây là quyết định dữ liệu được khóa cùng chính sách bảo trì: bảy channel hiện có đã biểu diễn no-trade bằng return/range/volume/trades bằng 0 và taker ratio trung tính; mask tổng hợp chỉ phục vụ audit, không tạo shortcut nhận diện hai ngày lịch sử. Quy tắc neutral ratio vẫn được giữ.

Final-test period chỉ được kiểm tra chất lượng dữ liệu sau khi quyết định này đã khóa; không được dùng để chọn lại channel hoặc mở rộng danh sách maintenance interval.

Các biến đuôi dày có thể được winsorize bằng ngưỡng chỉ tính trên train fold. Không dùng ngưỡng toàn bộ dữ liệu.

Ngoài bảy chuỗi theo nến, mỗi patch có một scalar biến động riêng:

\[
patch\_logRV_{s,j}
=
\log\left(
\sum_{i\in patch(s,j)}r_i^2+\epsilon_{RV}
\right)
\qquad
\epsilon_{RV}=10^{-12}
\]

Đây không phải là `mean(log_squared_return)`: log của tổng bình phương return giữ trực tiếp mức RV của patch. Scalar này được chuẩn hóa bằng statistic core-train của từng scale và concatenate vào vector của mỗi channel patch trước linear projection. Nó là patch metadata dùng chung, không được tính như channel thứ tám.

Scaler của từng kênh chỉ được fit trên train fold. `sklearn.StandardScaler` tự đặt `scale_=1` cho feature hằng; nếu dùng custom scaler thì bắt buộc:

```python
scale = max(train_standard_deviation, scaler_epsilon)
```

và phải assertion mọi output sau scaling là hữu hạn.

Không dùng instance normalization làm mất mức biến động, trừ khi các statistic bị loại bỏ được đưa trở lại forecast head.

---

## 8. Chia patch theo PatchTST

### 8.0. Quy tắc cửa sổ liên tục

`7 ngày gần nhất` và `60 ngày gần nhất` luôn là ngày lịch UTC liên tục, không phải 7 hoặc 60 ngày hợp lệ gần nhất.

Một sample target ngày \(t\) chỉ được giữ nếu:

- Toàn bộ bảy ngày từ \(t-7\) đến \(t-1\) đều có đủ 288 nến và return liên tục sau verified-maintenance fill §2.1.1.
- Toàn bộ 60 ngày từ \(t-60\) đến \(t-1\) đều có đủ 288 nến và return liên tục sau verified-maintenance fill §2.1.1.
- Có nến `23:55` của ngày \(t-61\) để tính return `00:00` cho ngày đầu coarse window.
- Không nối hai đoạn intraday nằm ở hai phía của một ngày bị thiếu ngoài hai khoảng bảo trì đã khóa.

Nếu bất kỳ ngày nào trong cửa sổ không hợp lệ thì loại toàn bộ sample \(t\).

Audit phải báo cáo riêng:

```text
candidate_target_days
removed_for_invalid_target_day
removed_for_incomplete_fine_window
removed_for_incomplete_coarse_window
final_sample_count
```

### 8.1. Fine patches

Dùng bảy ngày intraday gần nhất:

\[
X^{fine}_t = X_{t-7:t-1}
\]

Cấu hình:

```text
Patch length = 12 nến = 1 giờ
Stride       = 12 nến
Số patch     = 7 × 24 = 168 patch/kênh
```

### 8.2. Coarse patches

Dùng 60 ngày intraday:

\[
X^{coarse}_t = X_{t-60:t-1}
\]

Cấu hình:

```text
Patch length = 72 nến = 6 giờ
Stride       = 72 nến
Số patch     = 60 × 4 = 240 patch/kênh
```

Coarse patch vẫn được tạo trực tiếp từ intraday, không dùng `logRV_d`, `logRV_w` hoặc `logRV_m`. Khi tính cả `patch_logRV`, patch 6 giờ giảm projection từ \(289\times d_{model}\) xuống \(73\times d_{model}\), đồng thời vẫn giữ bốn trạng thái nội nhật cho mỗi ngày.

Việc tăng từ 60 lên 240 coarse token làm attention đắt hơn. Do đó:

- Fine và coarse được encode thành hai chuỗi riêng.
- Không ghép token trước các PatchTST self-attention layer.
- Chỉ ghép hai scale sau khi đã encode và channel-fuse.
- Nếu profiling cho thấy full self-attention coarse quá tốn bộ nhớ, dùng local/window self-attention như một ablation kỹ thuật, không thay đổi patch definition.

---

## 9. PatchTST encoder

Với kênh \(c\), scale \(s\), patch \(j\):

\[
p_{c,s,j} \in \mathbb R^{P_s}
\]

Tạo input projection:

\[
\tilde p_{c,s,j}
=
[p_{c,s,j};patch\_logRV_{s,j}]
\in\mathbb R^{P_s+1}
\]

Patch embedding:

\[
z_{c,s,j}
=
W^{patch}_{s} \tilde p_{c,s,j}
+ PE_j
+ CE_c
+ SE_s
+ TE_j
\]

Trong đó:

- `PE`: sinusoidal positional encoding cố định.
- `CE`: channel embedding.
- `SE`: fine/coarse scale embedding.
- `TE`: time-of-day và DOW embedding.

PatchTST encoder chạy channel-independent: cùng block Transformer hai tầng được chia sẻ cho cả bảy kênh và hai scale fine/coarse; mỗi chuỗi vẫn được forward riêng. Chỉ projection \(W^{patch}_{s}\) và scale embedding là riêng theo scale. Hai encoder fine/coarse tách trọng số chỉ được phép như ablation nếu tổng tham số vẫn nằm trong ngân sách đã khóa.

Không dùng learned positional table cho toàn bộ patch sequence. Sinusoidal position encoding tiết kiệm tham số và cho phép thay đổi nhẹ chiều dài chuỗi mà không cần resize embedding.

### 9.1. Time embedding

Với fine patch:

\[
hour\_sin = \sin(2\pi hour/24)
\]

\[
hour\_cos = \cos(2\pi hour/24)
\]

\[
dow\_sin = \sin(2\pi dow/7)
\]

\[
dow\_cos = \cos(2\pi dow/7)
\]

Coarse patch 6 giờ dùng cả time-of-day, DOW và relative-position embedding.

---

## 10. Transformer self-attention

Với chuỗi token \(H\):

\[
Q = HW^Q,\qquad
K = HW^K,\qquad
V = HW^V
\]

Mỗi head:

\[
head_i =
softmax\left(
\frac{Q_i K_i^\top}{\sqrt{d_k}} + M
\right)V_i
\]

Multi-head attention:

\[
MHA(H) =
Concat(head_1,\ldots,head_h)W^O
\]

Sử dụng pre-LayerNorm Transformer block:

\[
H_1 =
H + Dropout(MHA(LN(H)))
\]

\[
H_2 =
H_1 + Dropout(FFN(LN(H_1)))
\]

Feed-forward network:

\[
FFN(x) =
W_2 GELU(W_1x+b_1)+b_2
\]

Đây là Transformer encoder đầy đủ với:

- Multi-head attention.
- Residual connection.
- Layer normalization.
- Position-wise feed-forward network.
- Dropout.

Self-attention chỉ hoạt động trên patch token, không hoạt động trực tiếp trên từng nến 5 phút.

---

## 11. Channel fusion

Sau PatchTST encoder, hợp nhất các kênh tại mỗi patch:

\[
H^{market}_{s,j}
=
\sum_c a_{c,s,j} H_{c,s,j}
\]

Kết quả:

```text
168 fine market patch tokens
240 coarse market patch tokens
--------------------------------
408 market patch tokens
```

Channel fusion chính dùng learned gated weighted sum:

\[
a_{c,s,j}
=
softmax_c\left(
g(H_{c,s,j})
\right)
\]

Attention pooling trên channel dimension chỉ là ablation được gắn nhãn, không phải lựa chọn được đổi sau khi xem validation.

Không flatten toàn bộ channel × patch vào một vector lớn.

---

## 12. News Transformer encoder

Với 30 ngày lịch sử:

```text
30 slow tokens
30 fast tokens
```

Projection:

\[
N^s_j =
W_{sf}[
semantic\_slow_j,
sentiment\_slow_j
]
+ PE_j + Type_s
\]

\[
N^f_j =
W_{sf}[
semantic\_fast_j,
sentiment\_fast_j
]
+W_d\,daily\_scalar_j
+ PE_j + Type_f
\]

Dùng chung \(W_{sf}\) cho thành phần slow/fast và dùng type embedding khác nhau. `daily_scalar_j` chỉ được gắn vào fast token qua \(W_d\), không lặp lại ở cả hai token.

`daily_scalar_j` là vector theo đúng ngày \(j\), đã được xử lý theo §5, gồm `news_intensity`, `log1p(canonical_source_count)`, `negative_ratio`, `log1p(negative_count_070)`, `negative_probability_max`, `negative_probability_std`, `positive_probability_max`, `sentiment_entropy_mean`, `semantic_dispersion`, `mean_relevance` và `no_news_dummy`. Vì vậy News Transformer nhận được chuỗi 30 ngày của mức độ tin, bất đồng sentiment và độ phân tán semantic; các scalar này không chỉ xuất hiện ở forecast head.

Ghép token:

\[
H^{news}
=
[N^s_1,N^f_1,\ldots,N^s_{30},N^f_{30}]
\]

Sau đó chạy một Transformer self-attention encoder block chuẩn.

Kết quả:

```text
60 contextualized news tokens
1 learned null-news token
```

Learned null-news token luôn tồn tại và không bị mask. Nó cung cấp một key/value trung tính khi cửa sổ có rất ít tin.

---

## 13. Cross-attention Transformer

Mô hình chính cho market patch token truy vấn news token:

\[
Q = H^{market}W_Q
\]

\[
K = H^{news}W_K,\qquad
V = H^{news}W_V
\]

\[
CrossAttn(H^{market},H^{news})
=
softmax\left(
\frac{QK^\top}{\sqrt{d_k}} + M_{cross}
\right)V
\]

Block chuẩn:

\[
M_1
=
H^{market}
+
Dropout(
CrossAttn(
LN(H^{market}),
LN(H^{news})
))
\]

\[
M_2
=
M_1
+
Dropout(
FFN(LN(M_1))
)
\]

Cross-attention chỉ chạy giữa:

- 408 market patch tokens.
- 60 slow/fast news tokens và một learned null-news token.

Không có cross-attention ở cấp nến 5 phút.

### 13.1. Hiện thực attention và ngân sách bộ nhớ

Coarse branch có 240 token cho mỗi channel. Self-attention coarse phải dùng `torch.nn.functional.scaled_dot_product_attention` hoặc `nn.MultiheadAttention` với `need_weights=False`, AMP và backend SDPA/Flash Attention của PyTorch khi GPU, dtype và mask thực sự hỗ trợ. SDPA không bảo đảm tự chọn Flash/memory-efficient backend trên mọi cấu hình, nên phải log backend và profile peak memory. Không tự tạo và giữ ma trận attention \(240\times240\) cho mọi channel nếu không cần xuất attention map.

Cấu hình khởi đầu dùng batch size 32; nếu không đủ bộ nhớ thì dùng gradient accumulation. Chỉ tăng lên 64 sau khi profile peak GPU memory với forward, backward và mixed precision. Với batch 64, 7 channel, 4 head và 240 token, riêng attention logits float32 đã xấp xỉ 413 MB mỗi layer trước activation phục vụ backward; do đó batch 64 không phải mặc định an toàn. Local/window attention chỉ được chạy như ablation kiến trúc được gắn nhãn, không được dùng làm silent fallback.

### 13.2. Chính sách mask

Không dùng điều kiện:

```text
news_time <= patch_end_time
```

Điều kiện này không cần thiết cho bài toán một-output-mỗi-ngày và có thể tạo hàng attention bị mask hoàn toàn: coarse patch từ \(t-60\) đến \(t-31\) không có news token tương ứng trong lookback 30 ngày. Softmax trên một hàng toàn `-inf` sẽ sinh NaN.

Tính nhân quả được bảo đảm khi dựng sample:

```text
market input <= t-1
news input   <= t-1
target       = t
```

Vì toàn bộ context đã thuộc quá khứ tại prediction origin, self-attention và cross-attention được phép nhìn toàn bộ token trong cửa sổ.

Chỉ dùng:

- `key_padding_mask` cho padding thực sự.
- Learned null-news token luôn unmasked.
- Assertion kiểm tra mỗi query còn ít nhất một key hợp lệ trước khi gọi softmax.

Nếu sau này chuyển sang nowcasting hoặc sinh output theo từng patch thì phải thiết kế lại temporal mask cho bài toán đó.

---

## 14. Attention pooling và forecast head

Không flatten 408 token.

Tạo bốn learned forecast query:

\[
Q_f \in \mathbb R^{4\times d}
\]

Pooling bằng cross-attention:

\[
H_f =
MHA(Q=Q_f,K=M_2,V=M_2)
\]

\[
z_t =
W_{pool}
[H_{f,1};H_{f,2};H_{f,3};H_{f,4}]
+b_{pool},
\qquad z_t\in\mathbb R^d
\]

Không lấy trung bình bốn query trong cấu hình này: phép trung bình tạo đối xứng và có nguy cơ làm các query dư thừa, dù không bắt buộc chúng phải hội tụ giống nhau. Phép nối \(4d\rightarrow d\) cho phép từng query học vai trò khác nhau, nhưng thêm \(4.128\) tham số khi \(d=32\). Phải đếm tổng tham số thực tế; nếu vượt ngân sách 60k thì một learned query là cấu hình chính tiết kiệm hơn và concat bốn query chuyển thành ablation.

Chỉ ghép trực tiếp hai scalar của ngày gần nhất \(t-1\):

- `news_intensity_{t-1}`
- `no_news_dummy_{t-1}`

Đây là shortcut có chủ đích cho cường độ tin gần nhất và trạng thái không có tin. Các scalar sentiment/dispersion đầy đủ vẫn nằm trong chuỗi 30 ngày của news token tại §12; không lặp toàn bộ chúng ở head để tránh một đường tắt quá mạnh và khó quy kết giá trị của attention.

Forecast head:

```text
Linear(d_model + 2, 32)
GELU
Dropout
Linear(32, 1)
```

Đầu ra là `forecast_z`.

---

## 15. Chuẩn hóa target

Trong mỗi fold:

\[
\mu_y = mean(\log RV_{train})
\]

\[
\sigma_y = std(\log RV_{train})
\]

Mạng xuất:

\[
\widehat z_t
\]

Chuyển về thang logRV:

\[
\widehat{\log RV}_t
=
\mu_y + \sigma_y \widehat z_t
\]

QLIKE phải được tính sau khi đã chuyển về thang logRV gốc.

Với mô hình deep tối ưu QLIKE:

\[
\widehat{RV}_t
=
\exp(\widehat{\log RV}_t)
\]

Không áp dụng Jensen correction hoặc Duan smearing cho đầu ra này. Cấu hình chính dùng exact QLIKE, nên \(\exp(\widehat{\log RV}_t)\) nhắm trực tiếp \(E[RV_t\mid\mathcal F_{t-1}]\) và không có cơ sở cộng thêm smearing factor. Smearing chỉ áp dụng cho benchmark HAR-OLS tại §19.1.

---

## 16. Hàm mất mát

QLIKE:

\[
L_t =
\exp(\log RV_t-\widehat{\log RV}_t)
-
(\log RV_t-\widehat{\log RV}_t)
-1
\]

Exact QLIKE tối ưu dự báo \(E[RV_t\mid\mathcal F]\), không tối ưu \(E[\log RV_t\mid\mathcal F]\).

Do đó:

- QLIKE là metric chính.
- \(R^2\), RMSE và MAE trên logRV là metric phụ.

Cấu hình chính huấn luyện bằng exact QLIKE, giống objective của HAR-QLIKE. Ổn định số được xử lý mà không đổi cực tiểu của objective:

- Tính \(u\), `exp(u)` và mean loss trong float64 bên ngoài AMP autocast.
- Khởi tạo weight của `Linear(32, 1)` cuối bằng 0 và bias theo:

  \[
  b_0
  =
  \frac{
  \log(mean(RV_{core\ train}))-\mu_y
  }{\sigma_y}
  \]

  nên mọi deep model và Gamma baseline đều xuất phát từ cùng dự báo QLIKE vô điều kiện \(\widehat{RV}=mean(RV_{core\ train})\).
- Unscale gradient trước khi clip.
- Clip global gradient norm tại ngưỡng khóa trong §17.
- Nếu loss hoặc gradient không hữu hạn, đánh dấu run thất bại; không âm thầm thay loss hoặc clamp \(u\).

Robust QLIKE với linear upper tail chỉ là fallback/sensitivity model được gắn nhãn riêng:

\[
u = \log RV_t-\widehat{\log RV}_t > c,
\qquad c=3
\]

Định nghĩa:

\[
L(u)=
\begin{cases}
\exp(u)-u-1, & u\le c\\
L(c)+(\exp(c)-1)(u-c), & u>c
\end{cases}
\]

Phần nối tuyến tính liên tục cả giá trị và đạo hàm tại \(c\), đồng thời vẫn giữ gradient đúng hướng khi mô hình dự báo RV quá thấp.

Robust hóa thay đổi estimating equation thành:

\[
E\left[
\min\left(
\frac{RV_t}{\widehat{RV}_t},
\exp(c)
\right)
\middle|\mathcal F_{t-1}
\right]=1
\]

vì vậy không tuyên bố head robust-QLIKE ước lượng chính xác conditional mean RV. Robust variant khóa \(c=3\), dùng cùng optimizer/scheduler/seeds và không được thay thế một run exact bị lỗi trong cùng bảng. Nếu exact training không hoàn thành đủ protocol, báo riêng tỷ lệ numerical failure và kết quả robust fallback; không đổi nhãn objective. Mọi đánh giá cuối cùng luôn dùng exact QLIKE gốc.

---

## 17. Cấu hình kiến trúc và huấn luyện đã khóa

| Thành phần | Giá trị |
|---|---:|
| Fine lookback | 7 ngày |
| Fine patch length | 12 nến |
| Fine stride | 12 |
| Coarse lookback | 60 ngày |
| Coarse patch length | 72 nến = 6 giờ |
| Coarse stride | 72 |
| Patch-level scalar | `log(sum(r²)+epsilon_RV)` |
| News lookback | 30 ngày |
| Sentence encoder | `BAAI/bge-base-en-v1.5` |
| Sentence encoder max input | 512 token |
| Semantic embedding | 768 chiều |
| FinBERT role | Chỉ sentiment probabilities |
| Sentiment dimensions | 3 |
| PCA dimensions | 8 cho model chính; 16 là ablation định trước |
| `d_model` | 32 |
| Attention heads | 4 |
| PatchTST layers | 2 |
| PatchTST weight sharing | Chung giữa 7 channel và 2 scale |
| Channel fusion | Learned gated weighted sum |
| News self-attention layers | 1 |
| Cross-attention layers | 1 |
| FFN dimension | 64 |
| Forecast queries | 4 nếu tổng model ≤60k tham số; ngược lại 1 |
| Forecast-query pooling | Concatenate \(4d\), chiếu \(4d\rightarrow d\); identity với 1 query |
| Dropout | 0,1 |
| Positional encoding | Sinusoidal cố định |
| Attention backend | PyTorch SDPA/Flash nếu khả dụng |
| Effective batch size | 32; dùng gradient accumulation nếu physical batch nhỏ hơn |
| DataLoader order | Core train shuffle bằng seeded generator; validation/test không shuffle; `drop_last=False` |
| Training loss chính | Exact QLIKE |
| Robust fallback loss | Linear-tail QLIKE, \(c=3\), báo riêng |
| Optimizer | AdamW |
| Learning rate | \(3\times10^{-4}\) |
| Adam betas / epsilon | `(0.9, 0.999)` / \(10^{-8}\) |
| Weight decay | \(10^{-4}\); không decay bias và LayerNorm |
| Warmup | Linear \(0\rightarrow3\times10^{-4}\) trong 100 optimizer steps |
| Scheduler | Cosine decay sau warmup đến \(3\times10^{-6}\), sau đó giữ floor |
| Scheduler horizon | \(H_{cos}\) epoch-equivalent từ step 0 đến LR floor, gồm cả warmup; khóa bằng §17.1 |
| Max epochs | 200 |
| Early-stopping metric | Mean exact validation QLIKE |
| Early-stopping patience | 20 epoch, `min_delta=1e-5`, restore best |
| Gradient clipping | Global norm \(=1,0\), sau AMP unscale |
| Forecast-output initialization | Weight \(=0\), bias \(b_0=(\log mean(RV_{core})-\mu_y)/\sigma_y\) |
| Primary seeds | `[11, 22, 33, 44, 55]` |
| Extended seeds | Thêm `[66, 77, 88, 99, 111]` chỉ khi chạy đồng loạt mọi model |
| Mục tiêu trainable parameters | 20k–60k, phải kiểm tra thực tế |

Sentence encoder và FinBERT đều được freeze và không tính vào số tham số trainable của mô hình dự báo.

Không search learning rate, weight decay, scheduler, patience hoặc clipping threshold trên từng fold/ablation. Mọi model deep và mọi seed dùng đúng protocol trên; chỉ các trục ablation được liệt kê rõ trong §19.2 mới được phép thay đổi.

Validation chạy sau mỗi epoch. Patience tăng khi mean exact validation QLIKE không giảm ít nhất \(10^{-5}\) so với best checkpoint; dừng sau 20 epoch liên tiếp và khôi phục checkpoint tốt nhất. Sau khi \(H_{cos}\) đã khóa, step mà learning rate chạm floor trong fold \(f\) là:

\[
T_{floor,f}
=
H_{cos}
\left\lceil
\frac{n_{core,f}}{32}
\right\rceil
\]

optimizer steps tính từ step 0. 100 step đầu là linear warmup; cosine decay chạy từ step 101 đến \(T_{floor,f}\). Nếu training tiếp tục sau \(T_{floor,f}\), learning rate giữ tại \(3\times10^{-6}\).

### 17.1. Khóa cosine horizon một lần

Trước khi sinh hoặc xem bất kỳ prediction/loss nào của test block:

1. Chạy đúng một pilot của full primary model trên core train và validation của fold 1, PCA \(k=8\), seed 11, exact QLIKE.
2. Pilot dùng provisional cosine horizon 200 epoch và cùng mọi siêu tham số khác của §17.
3. Gọi epoch early-stop thực tế là \(E_{pilot}\), rồi khóa:

   \[
   H_{cos}
   =
   \max\left(
   10,\,
   5\left\lceil
   \frac{E_{pilot}}{5}
   \right\rceil
   \right)
   \]

4. Ghi `H_cos`, `E_pilot`, config hash và timestamp vào `scheduler_horizon.json`.
5. Hủy pilot checkpoint và chạy lại từ đầu toàn bộ fold/model/seed bằng \(H_{cos}\) đã khóa.

Không được mở fold-1 test prediction để quyết định \(H_{cos}\). Nếu pilot không early-stop trước epoch 200, khóa \(H_{cos}=200\). Pilot chạy với horizon 200 còn các run chính chạy với \(H_{cos}\), nên epoch dừng thực tế của run chính có thể khác \(E_{pilot}\), thường sớm hơn khi annealing nhanh hơn. Đây là hành vi dự kiến và không phải lý do để khóa lại horizon. Không hiệu chỉnh lại theo ablation, seed, fold hoặc COVID stress fold.

---

## 18. Walk-forward

Final test giữ tương đương với thực nghiệm HAR:

```text
Development:
2018-01 đến 2024-04-16

Final test:
2024-04-17 đến 2025-06-30
```

Trong development:

- Dùng đúng 5 expanding-window fold, test block 180 ngày lịch và origin tăng 180 ngày.
- Với mỗi outer train fold, lấy 90 ngày lịch liên tục cuối cùng làm validation.
- Không randomize thứ tự trước khi chia fold. Sau khi dựng core-train sample point-in-time, DataLoader được shuffle tái lập theo seed như §17; validation/test giữ nguyên thứ tự thời gian.
- Phần train dùng để tối ưu trọng số kết thúc trước validation.
- Fit scaler và PCA trên phần train trước validation, không fit trên validation.
- Validation chỉ dùng để early stopping và chọn checkpoint; criterion là mean exact QLIKE trên validation, không phải robust training loss.
- Các observation validation không được đưa vào optimizer của fold đó.
- Không chọn hyperparameter trên final test.
- Chạy đúng năm primary seed ở §17 cho bảng chính.
- Dự báo ensemble bằng trung bình trên thang RV.

Lịch target date được khóa trước khi xem kết quả:

| Fold | Core train dùng cho optimizer/preprocessor | Validation 90 ngày | Test 180 ngày |
|---:|---|---|---|
| 1 | 2018-01-01 – 2021-07-31 | 2021-08-01 – 2021-10-29 | 2021-10-30 – 2022-04-27 |
| 2 | 2018-01-01 – 2022-01-27 | 2022-01-28 – 2022-04-27 | 2022-04-28 – 2022-10-24 |
| 3 | 2018-01-01 – 2022-07-26 | 2022-07-27 – 2022-10-24 | 2022-10-25 – 2023-04-22 |
| 4 | 2018-01-01 – 2023-01-22 | 2023-01-23 – 2023-04-22 | 2023-04-23 – 2023-10-19 |
| 5 | 2018-01-01 – 2023-07-21 | 2023-07-22 – 2023-10-19 | 2023-10-20 – 2024-04-16 |

Các mốc trên là ngày lịch trước khi áp dụng quy tắc cửa sổ liên tục tại §8.0; target core-train đầu tiên thực tế chỉ xuất hiện sau khi có đủ 60 ngày lookback. Không dịch ranh giới fold sau khi nhìn kết quả. Sample không hợp lệ được loại nhưng phải báo cáo số target thực dùng trong từng core train, validation và test block. Validation phải còn ít nhất 60 target hợp lệ; nếu thấp hơn thì đánh dấu fold không hợp lệ, không tự chọn block thay thế.

Sau khi khóa kiến trúc và hyperparameter bằng 5 fold trên, final model dùng:

```text
Core train: 2018-01-01 – 2024-01-17
Validation: 2024-01-18 – 2024-04-16
Final test: 2024-04-17 – 2025-06-30
```

Checkpoint được chọn bằng validation 90 ngày cuối và không refit trên validation sau khi early stopping. Model weights, scaler, PCA và target normalization được giữ cố định trong final test. News/market state và cửa sổ đầu vào vẫn được cập nhật tuần tự bằng dữ liệu quan sát đến hết \(t-1\).

### 18.1. COVID crisis stress fold

Năm fold chính không có khủng hoảng tháng 3/2020 trong test. Sau khi đã khóa kiến trúc, feature, optimizer, loss và toàn bộ siêu tham số, chạy thêm đúng một fold chẩn đoán:

```text
Core train: 2018-01-01 – 2019-10-06
Validation: 2019-10-07 – 2020-01-04  (90 ngày lịch)
Stress test: 2020-01-05 – 2020-07-02 (180 ngày lịch)
```

Core train có tối đa 584 target sau burn-in 60 ngày và trước bộ lọc cửa sổ. Fold này:

- Dùng đúng primary seeds và training protocol §17.
- Validation phải còn ít nhất 60 target hợp lệ như năm fold chính; nếu không, stress fold bị đánh dấu không hợp lệ và không dịch ranh giới.
- Dùng P90 RV của chính stress-core-train để phân loại normal/spike.
- Không tham gia chọn kiến trúc, hyperparameter, loss, PCA dimension hoặc threshold.
- Không gộp vào 5-fold development mean, Model Confidence Set chính hay final-test inference.
- Báo riêng mean QLIKE, QLIKE spike/normal, số numerical failure và từng ngày loss lớn nhất.

Validation 2019-10-07 – 2020-01-04 nằm ngay trước cú sốc COVID và thuộc regime tương đối yên tĩnh hơn stress test. Do đó checkpoint được chọn trong điều kiện bình lặng rồi áp dụng vào một regime cực đoan chưa xuất hiện trong core/validation. Đây là distribution-shift scenario có chủ đích, gần với tình huống dự báo thực tế, không phải sơ suất khi chia mẫu.

Đây là parameter-estimation OOS nhưng không phải design-time OOS hoàn hảo, vì kiến trúc nghiên cứu được khóa sau khi toàn bộ giai đoạn lịch sử đã tồn tại. Nó chỉ là stress diagnostic cho một regime cực đoan với train rất nhỏ.

### 18.2. Data loader

Không materialize sẵn tensor:

```text
samples × 60 days × 288 bars × channels
```

Chỉ lưu mảng intraday đã làm sạch một lần. `Dataset.__getitem__` nhận target index và slice trực tiếp:

```python
fine_window = intraday[start_fine:end]
coarse_window = intraday[start_coarse:end]
```

Patch được tạo trong `__getitem__` hoặc trong collate function. Điều này tránh nhân bản cùng một nến hàng chục lần giữa các sample chồng lấn.

Ensemble:

\[
\widehat{RV}^{ensemble}_t
=
\frac{1}{S}
\sum_{s=1}^{S}
\exp(\widehat{\log RV}^{(s)}_t)
\]

Trong bảng chính, \(S=5\) và phải đúng năm primary seed `[11,22,33,44,55]`. Không thay seed exact bị lỗi bằng robust fallback, seed mới hoặc duplicate của seed khác. Model thiếu bất kỳ primary seed nào ở fold/phân tích tương ứng chỉ được báo trong bảng phụ với `S_actual`; ensemble biến thiên theo \(S\) đó không tham gia so sánh chính hoặc Model Confidence Set.

---

## 19. Baseline và ablation

### 19.1. Benchmark

Đặt:

\[
y_t=\log RV_t,
\qquad
v_t=RV_t
\]

và dùng đúng định nghĩa HAR hiện tại:

\[
logRV_{d,t}=y_{t-1}
\]

\[
logRV_{w,t}=\frac{1}{5}\sum_{i=1}^{5}y_{t-i},
\qquad
logRV_{m,t}=\frac{1}{22}\sum_{i=1}^{22}y_{t-i}
\]

Các benchmark:

Với mọi baseline QLIKE, linear predictor là \(\eta_i=X_i\beta\), dự báo dương là \(\mu_i=\exp(\eta_i)\), và:

\[
QLIKE_i
=
\frac{v_i}{\mu_i}
-
\log\left(\frac{v_i}{\mu_i}\right)
-1
\]

Unit deviance của Gamma GLM với log link là:

\[
D_{\Gamma}(v_i,\mu_i)
=
2\left[
\frac{v_i}{\mu_i}
-
\log\left(\frac{v_i}{\mu_i}\right)
-1
\right]
=2\,QLIKE_i
\]

Do đó exact QLIKE chính là half Gamma deviance. HAR-QLIKE và các Ridge-QLIKE được fit như Gamma GLM chuẩn, không dùng finite-difference optimization.

1. **Random walk**:

   \[
   \widehat{\log RV}_t=\log RV_{t-1}
   \]

2. **HAR-OLS với Duan smearing**:

   \[
   x^{HAR}_t=[1,logRV_{d,t},logRV_{w,t},logRV_{m,t}]
   \]

   Fit OLS trên \(y_t\) của core train. Với residual core-train \(\hat\epsilon_i=y_i-x_i^\top\hat\beta\), tính:

   \[
   S_f=\frac{1}{n_f}\sum_{i\in core\ train_f}\exp(\hat\epsilon_i)
   \]

   và dùng:

   \[
   \widehat{RV}_t=S_f\exp(x_t^\top\hat\beta)
   \]

   cho QLIKE. \(S_f\) chỉ được ước lượng từ core train của fold. Bản naive \(\exp(x_t^\top\hat\beta)\) chỉ được báo cáo như diagnostic không hiệu chỉnh. \(R^2\)/RMSE logRV của HAR-OLS dùng \(x_t^\top\hat\beta\), không dùng \(\log S_f\).

3. **HAR-QLIKE**: dùng \(x^{HAR}_t\), nhưng ước lượng trực tiếp bằng exact QLIKE/Gamma GLM trên core train:

   \[
   \hat\beta
   =
   \arg\min_\beta
   \frac{1}{n_f}\sum_i
   \left[
   \exp(y_i-x_i^\top\beta)
   -(y_i-x_i^\top\beta)-1
   \right]
   \]

   Đây là `GammaRegressor(alpha=0)`; lớp này cố định Gamma distribution với log link. Khi gọi API, không đưa cột constant vào \(X\); dùng `fit_intercept=True`.

4. **HAR+DOW-QLIKE**: nối thêm đúng sáu biến giả:

   ```python
   pd.get_dummies(index.dayofweek, drop_first=True)
   ```

   DOW của target date \(t\) là thông tin lịch đã biết; thứ Hai là nhóm chuẩn.

5. **HAR-News-QLIKE**: dùng exact QLIKE với:

   \[
   [
   x^{HAR}_t,\,
   news\_intensity_{t-1},\,
   negative\_ratio_{t-1},\,
   negative\_probability\_max_{t-1},\,
   no\_news\_dummy_{t-1}
   ]
   \]

   Mọi feature tin đều chốt ở \(t-1\), dùng cùng phép impute/scaling train-fold như deep model. Đây là baseline tuyến tính trực tiếp cho câu hỏi tin tức có thêm thông tin ngoài HAR hay không.

6. **HAR+DOW-News-QLIKE**: nối sáu DOW dummy vào benchmark số 5 để kiểm tra tin tức sau khi đã kiểm soát cả persistence và mùa vụ theo ngày trong tuần.

7. **Ridge-QLIKE**: mô hình tuyến tính exact-QLIKE có phạt \(L_2\), không dùng learned embedding từ mạng. Vector đầu vào được khóa như sau:

   - Mean và standard deviation của từng kênh trong từng patch: \(408\times7\times2=5.712\) feature.
   - Một `patch_logRV` cho mỗi patch: 408 feature.
   - 60 lag `logRV` ngày: 60 feature.
   - Với từng ngày trong 30 ngày news lookback: `semantic_slow`, `semantic_fast`, `sentiment_slow`, `sentiment_fast` và 11 `daily_scalar`; tổng \(30(2k+17)\) feature.

   Tổng dimensionality là 7.170 khi \(k=8\), hoặc 7.650 khi \(k=16\). Tất cả feature được chuẩn hóa theo core train; hệ số phạt chỉ chọn trên validation. Pipeline phải lưu ordered feature names, \(k\), dimensionality và preprocessor hash để bảo đảm baseline tái lập đúng bộ thông tin đầu vào.

8. **Ridge-Reduced-QLIKE**: baseline tuyến tính đúng tầm mẫu, gồm:

   - 60 lag `logRV`.
   - `semantic_slow_{t-1}` và `semantic_fast_{t-1}`: \(2k\) feature.
   - `sentiment_slow_{t-1}` và `sentiment_fast_{t-1}`: 6 feature.
   - 11 `daily_scalar_{t-1}`.

   Tổng dimensionality là 93 khi \(k=8\), hoặc 109 khi \(k=16\). Đây là linear comparator chính; Ridge-QLIKE 7.170/7.650 chiều chỉ trả lời câu hỏi information-matched và được diễn giải như high-dimensional diagnostic.

Cả hai Ridge giữ quy ước penalty theo **sum-loss**:

\[
\sum_i L^{exact}_{QLIKE,i}
+\lambda_{sum}\|\beta_{non\text{-}intercept}\|_2^2
\]

Gradient giải tích của quy ước này là:

\[
g_{sum}
=
X^\top(1-\exp(u))
+
2\lambda_{sum}\beta_{non\text{-}intercept}
\]

với grid khóa trước:

```text
lambda_sum ∈ {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
              1, 10, 100, 1000, 10000}
```

`sklearn.linear_model.GammaRegressor` dùng objective:

\[
\frac{1}{n_f}\sum_i QLIKE_i
+
\frac{\alpha_f}{2}
\|\beta_{non\text{-}intercept}\|_2^2
\]

Vì vậy trong fold \(f\) có \(n_f\) core-train observation, ánh xạ bắt buộc là:

\[
\alpha_f
=
\frac{2\lambda_{sum}}{n_f}
\]

Không truyền trực tiếp số \(\lambda_{sum}\) vào tham số `alpha`. Chọn \(\lambda_{sum}\) có mean exact validation QLIKE thấp nhất; nếu chênh lệch nhỏ hơn `1e-5`, chọn \(\lambda_{sum}\) lớn hơn. `GammaRegressor(fit_intercept=True)` không phạt intercept.

Implementation chuẩn dùng `scikit-learn==1.7.2`:

```python
GammaRegressor(
    alpha=alpha_f,       # 0 cho các HAR-QLIKE không phạt
    fit_intercept=True,
    solver="lbfgs",
    max_iter=5000,
    tol=1e-8,
    warm_start=False,
)
```

Solver này truyền loss và gradient giải tích cho L-BFGS-B. Với \(u=\log v-X\beta\), gradient dưới quy ước mean-loss là:

\[
\nabla J
=
\frac{1}{n_f}
X^\top(1-\exp(u))
+
\alpha_f\beta_{non\text{-}intercept}
\]

và Hessian:

\[
\nabla^2 J
=
\frac{1}{n_f}
X^\top diag(\exp(u))X
+
\alpha_f I_{non\text{-}intercept}
\]

Không bao giờ gọi `scipy.optimize.minimize` mà thiếu `jac`, và không dùng sai phân hữu hạn. Intercept được khởi tạo tại \(\log(mean(v_{core\ train}))\), coefficient còn lại bằng 0.

Guard số học bắt buộc:

- Tính \(u\) trong float64.
- Candidate có `max(u) > 700` hoặc linear predictor không hữu hạn phải cho objective \(+\infty\) để line search lùi; không trả NaN.
- Unit test trên solver version đã khóa phải xác nhận extreme candidate trả `inf`, không phải NaN.
- Sau fit, assert coefficient, prediction và exact QLIKE đều hữu hạn; báo `n_iter_` và mọi `ConvergenceWarning`.
- Nghiệm không hội tụ không được dùng trong bảng chính.

### 19.2. Deep-learning ablation

1. Fine PatchTST, không news.
2. Fine + coarse PatchTST, không news.
3. Fine + coarse PatchTST + scalar news/sentiment, không semantic embedding.
4. PatchTST + semantic news concatenate, chưa dùng slow/fast.
5. PatchTST + gated slow/fast.
6. PatchTST + slow/fast cross-attention.
7. Cross-attention chiều ngược lại: news query market patches.
8. Pure DL chính cộng trực tiếp `logRV_d`, `logRV_w`, `logRV_m` tại forecast head.

Mô hình đề xuất chính:

```text
Fine + coarse PatchTST
+ News Transformer
+ slow/fast cross-attention
+ direct QLIKE forecast
```

Mô hình số 8 là bản hybrid chẩn đoán, không phải mô hình chính. Nếu hybrid thắng rõ rệt, kết luận là nhánh coarse chưa tự học được persistence mà HAR cung cấp sẵn. Nếu khoảng cách nhỏ, đó là bằng chứng pure DL đã tự học được cấu trúc này.

HAR-News thắng mô hình HAR không chứng minh Transformer chỉ khai thác đúng các scalar đó. Giá trị của semantic embedding và cross-attention phải được suy ra từ các so sánh nested giữa ablation 2→3→4→5→6 trên cùng fold và seed protocol.

---

## 20. Đánh giá

Báo cáo:

- Mean QLIKE tổng thể.
- Sum QLIKE.
- QLIKE normal.
- QLIKE spike.
- \(R^2\) trên logRV, chỉ là metric phụ.
- RMSE và MAE trên logRV.
- QLIKE theo từng fold.
- Mean và standard deviation qua các seed.
- Ensemble QLIKE.
- Số tham số.
- Thời gian huấn luyện.
- Numerical-failure rate theo model/seed.
- PCA captured variance ratio và relative drop theo fold.

Mọi chênh lệch QLIKE giữa hai mô hình phải được tính trên đúng giao các target date mà cả hai có dự báo hợp lệ. Bảng chính dùng common evaluation support của toàn bộ mô hình đã định trước; coverage riêng của từng mô hình được báo cáo bổ sung nhưng không dùng để so loss trên hai tập ngày khác nhau.

Tương tự ở chiều seed, bảng chính và Model Confidence Set chỉ nhận model hoàn thành đủ cả năm primary seed. Model thiếu seed được chuyển nguyên trạng sang bảng phụ, báo `S_actual`, seed thất bại và nguyên nhân; không tính một ensemble \(S<5\) như thể tương đương ensemble chính.

Spike được xác định bằng P90 của RV chỉ trên core train của từng fold:

\[
Spike_t =
\mathbf 1
\{RV_t > Q_{90}(RV_{core\ train})\}
\]

Không dùng P90 của test để tạo ngưỡng.

Với nhiều mô hình:

- Dùng Model Confidence Set làm đánh giá tổng thể.
- Dùng kiểm định chênh lệch loss đã định trước với HAR-QLIKE và HAR+DOW-QLIKE.
- QLIKE spike được gộp từ các out-of-sample fold và cần block-bootstrap confidence interval do số quan sát mỗi fold nhỏ.
- Không kết luận mạnh từ QLIKE spike của một fold đơn lẻ.

Năm development fold dùng để chọn kiến trúc/hyperparameter và báo độ ổn định. Model Confidence Set cùng kiểm định loss trên final test là phân tích confirmatory chính; kết quả gộp development fold chỉ là diagnostic vì đã tham gia quá trình lựa chọn mô hình.

COVID stress fold được trình bày trong bảng riêng và không được cộng vào mean/sum QLIKE của năm fold chính hoặc final test.

---

## 21. Các kiểm tra chống leakage và tính đúng đắn

1. Target ngày \(t\) chỉ dùng intraday đến hết \(t-1\).
2. News ngày \(t\) không được dùng để dự báo \(RV_t\).
3. Market/news scaler chỉ fit trên core train của fold.
4. PCA chỉ fit trên semantic embedding của ngày có tin trong core train.
5. SHA-256 của toàn bộ semantic transform pipeline — scaler center/scale, PCA mean, components và `n_components` — phải khớp giữa feature cache và model checkpoint.
6. Mỗi fold transform lại lịch sử và chạy lại slow/fast từ `2018-01-01`; không tái sử dụng \(Z\) hoặc state của fold khác.
7. Slow state không reset tại ranh giới core train→validation→test; news ngày test trước được phép cập nhật state cho dự báo một bước trước của ngày test sau.
8. Trước first-news, slow bằng zero; tại first-news khóa đúng quy tắc `slow=observation, fast=0`.
9. Xáo trộn mọi dữ liệu sau một cutoff không được làm thay đổi bất kỳ market/news feature nào trước cutoff.
10. Trên ngày có tin sau khởi tạo, `semantic_slow + semantic_fast == Z` và `sentiment_slow + sentiment_fast == sentiment` tới độ chính xác máy.
11. Trên ngày không tin, cả hai slow state được giữ, cả hai fast vector bằng zero và `no_news_dummy == 1`.
12. Scalar sentiment/dispersion ngày no-news phải bằng zero sau fold transform; test phải chặn việc đưa raw zero qua scaler.
13. Scaler output, patch tensor, attention output, loss và gradient đều hữu hạn; custom scale luôn lớn hơn hoặc bằng `scaler_epsilon`.
14. Không có attention query nào bị toàn bộ key padding-mask; null-news token luôn unmasked.
15. Toàn bộ market và news input của target \(t\) có timestamp không muộn hơn \(t-1\).
16. Fine và coarse window gồm các ngày lịch liên tục, không nối qua ngày thiếu.
17. Semantic embedding và FinBERT sentiment được tạo từ cùng nội dung đã loại footer.
18. Sentiment probabilities không đi qua semantic PCA.
19. `news_intensity_j` dùng đúng rolling/expanding median nhân quả và target \(t\) chỉ nhận các ngày \(j\le t-1\).
20. Early stopping chỉ dùng validation; validation observations không tham gia optimizer, scaler hoặc PCA của fold.
21. Fold boundary phải khớp bảng §18; mỗi block phải xuất candidate/removed/final sample counts.
22. Final test không được dùng để chọn feature, hyperparameter, epoch, loss cap hoặc attention backend.
23. Audit phải tách organic zero-volume và verified-maintenance synthetic bars; cả `zero_volume_mask` và `maintenance_synthetic` đều không được đưa vào feature, final period chỉ là QC.
24. Xuất bắt buộc audit drift tin theo tháng/năm và audit timestamp bất thường. Không tuyên bố đã loại leakage publication-vs-crawl nếu schema không có ingestion/version timestamp.
25. Full Ridge-QLIKE phải assert dimensionality 7.170/7.650; Ridge-Reduced phải assert 93/109 tương ứng với \(k\), ordered feature schema và grid \(\lambda\) đã khóa.
26. Attention smoke test phải log backend thực dùng, `need_weights=False`, peak GPU memory và tổng số trainable parameters.
27. Feature schema phải cấm `source_count`, `member_count`, republishing offsets và mọi member list hồi cứu; thay đổi các trường này ở record tương lai không được làm đổi feature của canonical day.
28. Mỗi ngày hợp lệ phải có đúng 288 return, trong đó return `00:00` dùng close `23:55` ngày trước; test phải thất bại nếu chỉ có 287 sai phân nội ngày.
29. `patch_logRV` phải bằng `log(sum(raw_return**2) + epsilon_RV)` của đúng patch trước scaling, không bằng mean của `log_squared_return`.
30. Tất cả loss differential, MCS và bảng QLIKE chính phải dùng cùng common OOS target-date mask.
31. Unit test training config phải assert AdamW, learning rate, weight decay, 100 warmup steps, cosine scheduler, max 200 epoch, patience 20 và grad-norm cap 1,0; checkpoint metadata phải lưu toàn bộ giá trị này. Với mọi fold kể cả COVID, bắt buộc \(T_{floor,f}>100\) để cosine phase có độ dài dương sau warmup.
32. Exact QLIKE là loss chính; robust \(c=3\) phải có model label riêng và không được thay thế âm thầm run exact không hữu hạn.
33. `CVR_core`, `CVR_validation`, `CVR_test` và relative PCA drop phải được xuất bằng đúng fold scaler/PCA; test feature không được refit vì diagnostic này.
34. COVID stress fold phải khớp đúng §18.1, validation còn ít nhất 60 target hợp lệ và không xuất hiện trong bất kỳ artifact chọn model/hyperparameter nào.
35. `Taker Buy Quote Asset Volume` không được xuất hiện trong input feature schema; nó chỉ có thể xuất hiện trong data-QC artifact.
36. Unit test phải xác nhận `Gamma unit deviance / 2 == exact QLIKE` trên một batch RV dương tới sai số float64.
37. Với mỗi Ridge fold, metadata phải chứa `lambda_sum`, \(n_f\), `alpha_f` và assert `alpha_f == 2*lambda_sum/n_f`; intercept không được đưa vào penalty.
38. Gamma solver phải dùng analytic gradient. Extreme-\(u\) test phải trả objective `inf`, không NaN; cấm mọi lời gọi L-BFGS-B không có `jac`.
39. Ngay sau initialization, mọi deep sample phải cho cùng \(\widehat{RV}=mean(RV_{core\ train})\) tới sai số số học.
40. `scheduler_horizon.json` phải tồn tại trước mọi test-prediction artifact; hash của nó phải khớp mọi deep checkpoint.
41. Main-table eligibility phải assert đủ chính xác năm primary seed và `S==5`; model có `S_actual<5` chỉ được xuất sang supplemental table và bị loại khỏi MCS.
42. Maintenance fill chỉ được áp dụng cho đúng 78 timestamp đã khóa tại §2.1.1, dùng duy nhất close ở `start-5 phút`, không đọc giá mở cửa lại; unit test phải xác nhận return của bar tổng hợp bằng 0 và toàn bộ reopening jump nằm ở nến thực đầu tiên sau bảo trì. Gap ngoài whitelist vẫn phải làm ngày/cửa sổ không hợp lệ.

---

## 22. Quyết định đã khóa trước khi code

- [x] Loại `Bitcoin Cash` và `BCH` nếu không có tín hiệu BTC độc lập.
- [x] Không dùng raw `source_count/member_count` hoặc member metadata hồi cứu của news cluster; trọng số bài chỉ dựa trên relevance point-in-time.
- [x] Semantic embedding dùng `BAAI/bge-base-en-v1.5`; FinBERT chỉ tạo sentiment probabilities.
- [x] PCA chỉ chạy trên semantic embedding 768 chiều.
- [x] PCA 8 chiều cho model chính; 16 chiều là ablation định trước, kèm captured-variance drift diagnostic theo fold.
- [x] Scaler/PCA/slow-fast được fit hoặc tái tính độc lập theo fold từ `2018-01-01`; state không reset giữa core train, validation và test.
- [x] Sentiment ngày là weighted mean theo cùng trọng số với semantic embedding và được phân rã slow/fast riêng.
- [x] Ngày no-news giữ slow, đặt fast và scalar không xác định về zero trong không gian đã transform, đồng thời giữ `no_news_dummy`.
- [x] Raw `news_count` được thay bằng `news_intensity` dùng rolling median 365 ngày nhân quả.
- [x] Fine lookback cố định 7 ngày.
- [x] Coarse lookback 60 ngày, patch 6 giờ và stride 6 giờ.
- [x] RV ngày có đúng 288 return, bao gồm return `00:00` từ close `23:55` ngày trước.
- [x] Chỉ hai khoảng Binance spot maintenance tại §2.1.1 được dựng causal no-trade bars; mọi gap khác vẫn bị loại.
- [x] Mỗi patch thêm `patch_logRV = log(sum(r²)+epsilon)` trước projection.
- [x] Không dùng `zero_volume_mask` hoặc `maintenance_synthetic` làm channel; organic zero và 78 bar bảo trì được audit riêng.
- [x] Positional encoding dùng sinusoidal cố định.
- [x] PatchTST Transformer chia sẻ trọng số giữa channel/scale; channel fusion chính là learned gated weighted sum.
- [x] Cross-attention chính là market patch query news.
- [x] Không dùng temporal cross-mask bên trong cửa sổ; causality được bảo đảm khi dựng sample.
- [x] Daily news scalar chỉ gắn vào fast-news token; forecast head chỉ nhận shortcut `news_intensity_{t-1}` và `no_news_dummy_{t-1}`.
- [x] Bốn forecast query được concatenate rồi chiếu \(4d\rightarrow d\) nếu tổng tham số không vượt 60k; nếu vượt, dùng một query theo quy tắc khóa trước khi train.
- [x] Effective batch 32, `need_weights=False`, AMP và SDPA/Flash nếu backend thực sự hỗ trợ.
- [x] Training protocol khóa: AdamW, LR \(3\times10^{-4}\), weight decay \(10^{-4}\), warmup 100 step, cosine đến \(3\times10^{-6}\), max 200 epoch, patience 20 và grad-norm cap 1,0.
- [x] \(H_{cos}\) được khóa đúng một lần từ fold-1 core/validation pilot theo §17.1 trước khi sinh hoặc xem bất kỳ test prediction nào; không hiệu chỉnh theo model/fold/seed.
- [x] Huấn luyện chính bằng exact QLIKE; robust linear-tail \(c=3\) chỉ là fallback/sensitivity được báo riêng; đánh giá luôn bằng exact QLIKE.
- [x] Deep forecast head khởi tạo tại cùng unconditional QLIKE optimum với Gamma baseline: \(\widehat{RV}=mean(RV_{core\ train})\).
- [x] Không dùng Jensen/smearing cho deep head; HAR-OLS dùng Duan smearing tính riêng trên core train.
- [x] HAR/Ridge exact-QLIKE dùng Gamma GLM log-link với analytic gradient; Ridge ánh xạ \(\alpha_f=2\lambda_{sum}/n_f\), không dùng finite differences.
- [x] Có HAR-News-QLIKE, HAR+DOW-News-QLIKE, full Ridge, Ridge-Reduced và scalar-only PatchTST ablation.
- [x] Development dùng 5 fold × 180 ngày, validation là 90 ngày lịch và cần ít nhất 60 target hợp lệ.
- [x] Early stopping chọn checkpoint bằng exact validation QLIKE; mọi so sánh model dùng cùng common OOS date mask.
- [x] Final test bắt đầu ngày 2024-04-17 để so sánh với HAR hiện tại.
- [x] Final fit giữ validation 2024-01-18 – 2024-04-16 để chọn checkpoint và không refit trên validation.
- [x] Primary seeds là `[11,22,33,44,55]`; chỉ mở rộng lên 10 seed nếu chạy đồng loạt mọi model.
- [x] Bảng chính/MCS chỉ nhận model hoàn thành đủ năm primary seed và ensemble \(S=5\); model thiếu seed chỉ báo ở bảng phụ với `S_actual`.
- [x] COVID fold 2020 chỉ là stress diagnostic hậu kiểm, không tham gia chọn model hoặc inference chính.
- [x] Ensemble trung bình dự báo trên thang RV.
- [x] Pure DL là mô hình chính; hybrid thêm ba đặc trưng HAR là mô hình chẩn đoán.
- [x] Cửa sổ fine/coarse bắt buộc liên tục và đầy đủ.
- [x] Dữ liệu intraday chỉ lưu một lần; window được slice trong `Dataset.__getitem__`.
- [x] `Taker Buy Quote Asset Volume` chỉ dùng QC, không phải feature.

## 23. Framing của thực nghiệm

Development period có 2.298 ngày lịch, tương ứng tối đa 2.238 target sau burn-in 60 ngày và trước khi loại cửa sổ intraday không hợp lệ. Core train tăng từ tối đa 1.248 target ở fold 1 lên 1.968 ở fold 5; final core train có tối đa 2.148 target. Số thực tế phải lấy từ audit §8.0. Đây vẫn là cỡ dữ liệu nhỏ cho mô hình đa nhánh có Transformer và cross-attention.

Kỳ vọng khoa học không nên được đóng khung là deep model chắc chắn đánh bại HAR. Câu hỏi chính là:

> Sau khi kiểm soát cùng lịch sử biến động, tin tức Bitcoin có cung cấp thông tin dự báo RV ngoài mẫu hay không?

Model Confidence Set có thể chứa đồng thời HAR-QLIKE và mô hình deep learning. Đây vẫn là kết quả hợp lệ nếu:

- Pipeline chống leakage.
- Market-only branch được cố định trước khi thay đổi news branch.
- Ablation tách được giá trị của market patches, semantic news, sentiment, slow/fast và cross-attention.
- QLIKE tổng thể và spike được báo cáo kèm độ bất định.

Năm development OOS fold chính và final test không chứa một khủng hoảng có quy mô như tháng 3/2020. Vì vậy QLIKE spike trong bảng chính chỉ chứng minh hiệu năng trên phần đuôi của các regime 2021–2025, không có cơ sở ngoại suy thành tuyên bố tổng quát về khủng hoảng chưa từng thấy. COVID fold §18.1 bổ sung bằng chứng stress riêng nhưng có core train tối đa 584 target và không phải design-time OOS; kết luận từ nó phải được gắn nhãn thăm dò.

Nếu captured variance ratio của PCA giảm mạnh trên test, cần nêu rằng cơ sở semantic \(k=8\) đã suy giảm khả năng biểu diễn. Không được dùng diagnostic này để refit PCA hoặc đổi \(k\) trên final test.

Để quy trách nhiệm rõ ràng, thứ tự phát triển phải là:

1. Khóa market-only PatchTST.
2. Thêm scalar news và sentiment.
3. Thêm semantic embedding concatenate.
4. Thêm slow/fast.
5. Cuối cùng mới thêm cross-attention.

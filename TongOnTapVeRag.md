# TỔNG ÔN TẬP VẤN ĐÁP — SmartDoc AI RAG

> **Format:** Mỗi mục = 1 câu hỏi vấn đáp. Trả lời ngắn → chi tiết → code thực tế.
> **Cập nhật:** Embedding model LaBSE, RRF normalize tuyệt đối, Co-RAG BM25 fix.
> *Dựa trên source code thực tế: modules/ + app.py + config.py*

---

# NHÓM A: RAG CƠ BẢN

---

### A1. RAG là gì? Tại sao cần RAG?

**Trả lời ngắn:**
RAG (Retrieval-Augmented Generation) = kết hợp tìm kiếm tài liệu + LLM sinh câu trả lời. Cần RAG vì LLM thuần hay "bịa" (hallucination), không biết tài liệu mới, và không trích nguồn.

**Công thức:**
```
a = G(q, C)      # answer = Generator(query, context)
C = R(q)         # context = Retriever(query) → top-k chunks
```

**4 vấn đề LLM thuần + RAG giải quyết:**

| Vấn đề | Hậu quả | RAG giải quyết |
|---|---|---|
| Hallucination | Bịa thông tin tự tin | Ràng buộc LLM chỉ dùng context |
| Knowledge cutoff | Không biết TL mới | Inject tài liệu bất kỳ lúc nào |
| Không trích nguồn | Không kiểm chứng được | Chunk mang metadata nguồn+trang |
| Token limit | Không nhét cả tài liệu | Chỉ lấy đoạn liên quan nhất |

---

### A2. Hai giai đoạn của RAG là gì?

**Giai đoạn 1 — INDEXING** (chạy 1 lần khi upload):
```
PDF/DOCX → Parser → Chunking → Embedding → FAISS + BM25 index
```

**Giai đoạn 2 — QUERYING** (chạy mỗi lần hỏi):
```
Query → Embed → Retrieve → (Rerank) → Prompt → LLM → Answer
```

---

### A3. SmartDoc AI dùng LLM gì? Cấu hình thế nào?

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| Model | `qwen2.5:7b` via Ollama | Chạy local, không cần internet |
| URL | `http://localhost:11434` | Ollama local server |
| temperature | 0.7 | Cân bằng sáng tạo/chính xác |
| top_p | 0.9 | Nucleus sampling |
| repeat_penalty | 1.1 | Giảm lặp từ |

**Hardware:** Intel i5-12400F, 24GB DDR4, CPU only (không GPU)

---

# NHÓM B: EMBEDDING & CHUNKING

---

### B1. Embedding là gì? SmartDoc AI dùng model nào?

**Trả lời ngắn:**
Embedding chuyển văn bản → vector số. Văn bản giống nhau về nghĩa → vector gần nhau.

**Model đang dùng:** `sentence-transformers/LaBSE` *(Language-Agnostic BERT Sentence Embeddings)*

| Đặc điểm | Giá trị |
|---|---|
| Tên model | `sentence-transformers/LaBSE` |
| Max tokens | **512 tokens** |
| Output dims | **768 chiều** |
| Ngôn ngữ | 109 ngôn ngữ (multilingual mạnh) |
| Device | CPU (`EMBEDDING_DEVICE = "cpu"`) |
| Normalize | `True` → unit vector |

> ⚠️ **Thay đổi từ phiên bản cũ:** model cũ `paraphrase-multilingual-mpnet-base-v2` chỉ có **128 tokens**, đã bị thay bằng LaBSE 512 tokens để xử lý chunk dài hơn.

---

### B2. Cosine similarity tính thế nào?

$$\text{sim}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{|\vec{q}| \times |\vec{d}|}$$

**Kết quả nằm trong [-1, +1]:**
- +1 = hoàn toàn giống nhau
- 0 = không liên quan
- -1 = trái nghĩa

**Khi `normalize_embeddings=True`:** tất cả vector là unit vector (|v|=1) → `sim = q · d` (chỉ cần dot product, nhanh hơn).

---

### B3. Tại sao cần chunk? Overlap có tác dụng gì?

**Cần chunk vì:**
- LLM có giới hạn context window
- Tìm kiếm chunk nhỏ chính xác hơn (query match đoạn cụ thể, không match cả file)
- Trích dẫn nguồn đến trang cụ thể

**Cần overlap vì:**
```
KHÔNG overlap:    "...thông tin A." | "Kết luận: B..."
                    → Mất "Kết luận" trong chunk 1, mất ngữ cảnh trong chunk 2

CÓ overlap=200:   "...thông tin A. [200 ký tự đầu chunk 2]"
                    → Không bao giờ mất thông tin ở ranh giới
```

**Config:** `CHUNK_SIZE=1500`, `CHUNK_OVERLAP=200` (≈13% overlap)

**RecursiveCharacterTextSplitter chia theo thứ tự:** `\n\n` → `\n` → `. ` → ` ` → ký tự đơn

---

### B4. DOCX xử lý khác PDF thế nào?

| | PDF | DOCX |
|---|---|---|
| Parser | `pdfplumber` | `python-docx` |
| Số trang | Có sẵn trong file | **Không có** → tạo "trang ảo" |
| Trang ảo | - | Gộp paragraph+bảng đến ~1500 ký tự |
| Bảng | Có thể miss | Thu thập: `"col1 | col2 | col3"` |

---

### B5. Indexing pipeline: từ file đến vector lưu trong FAISS?

**Toàn bộ luồng indexing** (chạy 1 lần khi upload tài liệu):

```
 PDF/DOCX
    │
    ▼
[document_processor.py]
    │  pdfplumber (PDF) / python-docx (DOCX)
    │  → trích text + metadata (filename, page)
    ▼
[RecursiveCharacterTextSplitter]
    │  CHUNK_SIZE=1500, CHUNK_OVERLAP=200
    │  → List[Document(page_content, metadata)]
    ▼
[LaBSE model — sentence-transformers]
    │  encode(chunk_text) → np.array shape (768,)
    │  normalize=True → unit vector
    │  (batch encoding — nhiều chunk 1 lần)
    ▼
[FAISS IndexFlatIP]
    │  add(vectors)  → lưu ma trận float32 [N × 768]
    │  mỗi vector ↔ 1 chunk ID
    ▼
[BM25Retriever]
    │  tokenize(chunk_text) → term frequency
    │  → inverted index (không dùng vector)
    ▼
 Lưu ra disk:
   vectorstore/smartdoc_index/index.faiss
   vectorstore/smartdoc_index/index.pkl  (docstore + metadata)
```

**Code thực tế (`vector_store.py`):**
```python
# Embed + tạo FAISS index
vectorstore = FAISS.from_documents(
    documents=chunks,          # List[Document]
    embedding=HuggingFaceEmbeddings(
        model_name="sentence-transformers/LaBSE",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
)
vectorstore.save_local(VECTOR_STORE_PATH)

# BM25 index (từ cùng chunks)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = HYBRID_TOP_K  # top-5
```

**Kết quả:** N chunks → N vector 768 chiều trong FAISS + inverted index trong BM25.

---

### B6. Bên trong embedding: một đoạn text trở thành vector thế nào?

**Quy trình LaBSE encode một chunk:**

```
 Đầu vào: "Hợp đồng lao động có hiệu lực từ ngày..."
      │
      ▼
 [Tokenizer]
      │  WordPiece / SentencePiece tokenization
      │  "Hợp" → "H##ợ##p" (ví dụ subword)
      │  Thêm [CLS] đầu, [SEP] cuối
      │  Padding/truncate đến max 512 tokens
      ▼
 [BERT Encoder — 12 transformer layers]
      │  Mỗi token → vector 768 chiều
      │  Attention: mỗi từ "nhìn" toàn bộ câu
      │  → Ma trận [512 × 768]
      ▼
 [Mean Pooling]
      │  Lấy trung bình tất cả token vectors
      │  [512 × 768] → [1 × 768]
      ▼
 [L2 Normalize]
      │  chia cho độ dài → unit vector
      │  |v| = 1.0
      ▼
 Đầu ra: np.array([0.021, -0.043, 0.118, ..., 0.007])  # 768 số
```

**Tại sao mean pooling, không lấy [CLS] token?**
LaBSE và các Sentence-Transformer đều dùng mean pooling vì [CLS] của BERT gốc được huấn luyện cho classification, không phải sentence similarity. Mean pooling tổng hợp nghĩa **toàn câu** tốt hơn.

**Hai vector gần nhau = hai đoạn văn nghĩa tương đồng:**
```
vec("Hợp đồng lao động")  · vec("Hợp đồng việc làm")  ≈ 0.92  ✅ gần
vec("Hợp đồng lao động")  · vec("Công thức hoá học")   ≈ 0.11  ❌ xa
```
*(dot product = cosine similarity vì đã normalize)*

---

# NHÓM C: RETRIEVAL — FAISS, BM25, HYBRID

---

### C0. Khi user hỏi, query được embed + tìm top chunks thế nào?

**Toàn bộ luồng retrieval** (chạy mỗi lần có câu hỏi):

```
 Câu hỏi: "Điều kiện thanh lý hợp đồng là gì?"
      │
      ▼
 [LaBSE encode — cùng model lúc indexing]
      │  → query_vector: np.array shape (768,) normalized
      ▼
 ┌─────────────────────┬───────────────────────┐
 │   FAISS search      │    BM25 search        │
 │                     │                       │
 │ IndexFlatIP.search( │ tokenize(query)       │
 │   query_vector,     │ → TF-IDF score vs     │
 │   fetch_k=50        │   mỗi chunk           │
 │ )                   │ → rank by BM25 score  │
 │ → 50 (doc, cosine)  │ → top-5 docs          │
 │        ↓            │         ↓             │
 │ MMR filter          │                       │
 │ (lambda=0.7)        │                       │
 │ → top-5 diverse     │                       │
 └──────────┬──────────┴──────────┬────────────┘
            │                    │
            └─────────┬──────────┘
                      ▼
           [RRF Fusion — EnsembleRetriever]
            score = 0.6/(60+rank_faiss)
                  + 0.4/(60+rank_bm25)
            → sort by score desc
            → normalize / (1/61)
                      ▼
            Top-K chunks (RETRIEVAL_TOP_K=8)
            mỗi chunk có: page_content, metadata, score
```

**FAISS tìm kiếm nhanh thế nào?**
FAISS dùng cấu trúc **HNSW** (Hierarchical Navigable Small World graph): thay vì so sánh query với tất cả N vectors (O(n)), nó đi qua đồ thị nhiều lớp để "nhảy" tới vùng gần nhất → O(log n) xấp xỉ.

```
FAISS IndexFlatIP.search(query_vec, k=50):
   → trả về distances[], indices[]
   distances[i] = dot_product(query_vec, stored_vec[i])
               = cosine similarity (vì đã normalize)
   indices[i]   = ID của chunk trong docstore
```

**Chú ý quan trọng:**
- Query phải dùng **cùng model** lúc indexing (LaBSE) — khác model là vector ở không gian khác, kết quả vô nghĩa
- FAISS chỉ biết vector, không biết text → metadata lưu riêng trong `docstore` (`.pkl`)
- BM25 không dùng embedding — nó tokenize cả query lẫn docs, so khớp từ khoá trực tiếp

---

### C1. FAISS là gì? MMR là gì?

**FAISS** (Facebook AI Similarity Search): thư viện tìm vector gần nhất với O(log n) nhờ cấu trúc HNSW thay vì O(n) brute-force.

**Vấn đề của FAISS thuần:** top-8 kết quả có thể là 8 đoạn **gần giống nhau** (cùng trang) → LLM nhận 8 chunk nhưng chỉ có 1 "góc nhìn".

**MMR — Maximal Marginal Relevance:**

$$\text{MMR}(d_i) = \lambda \cdot \text{sim}(d_i, q) - (1-\lambda) \cdot \max_{d_j \in S}\, \text{sim}(d_i, d_j)$$

- `sim(dᵢ, q)` = liên quan với query (muốn cao)
- `sim(dᵢ, dⱼ)` = giống với chunk đã chọn (muốn thấp)
- `λ=0.7` → 70% ưu tiên liên quan, 30% ưu tiên đa dạng

**Luồng SmartDoc AI:**
```
fetch_k=50 (FAISS lấy 50 ứng viên)
    ↓ MMR
k=8 chunks đa dạng
    ↓ CrossEncoder (nếu bật)
top_k=3 chính xác nhất
```

---

### C2. BM25 là gì? Khác FAISS thế nào?

**BM25** (Best Match 25): tìm kiếm từ khóa dựa trên tần suất từ, phiên bản cải tiến TF-IDF.

$$\text{score}(q,d) = \sum_{t \in q} \text{IDF}(t) \times \frac{\text{tf}(t,d) \times (k_1+1)}{\text{tf}(t,d) + k_1(1-b+b\frac{|d|}{\text{avgdl}})}$$

| Tiêu chí | FAISS (Vector) | BM25 (Keyword) |
|---|---|---|
| Hiểu đồng nghĩa | ✅ "xe cộ" = "phương tiện" | ❌ |
| Khớp từ kỹ thuật/số | ❌ có thể miss "Điều 15" | ✅ chính xác |
| Đa ngôn ngữ | ✅ LaBSE 109 ngôn ngữ | ❌ phụ thuộc tokenizer |
| Tốc độ tìm | O(log n) HNSW | O(n) tuyến tính |
| Cập nhật index | ✅ merge_from() | ❌ rebuild toàn bộ |

**→ Dùng cả hai: Hybrid Search**

---

### C3. Hybrid Search + RRF hoạt động thế nào?

**Vấn đề:** Không thể cộng điểm FAISS và BM25 vì thang đo khác nhau:
- FAISS cosine: [-1, +1]
- BM25 score: [0, ∞) không có chặn trên

**Giải pháp: RRF (Reciprocal Rank Fusion)** — thay điểm số bằng **thứ hạng**:

$$\text{RRF}(d) = \sum_{i} \frac{w_i}{k + \text{rank}_i(d)}$$

- `wᵢ`: trọng số retriever i (FAISS=0.6, BM25=0.4)
- `rank_i(d)`: thứ hạng của document d trong retriever i (bắt đầu từ 1)
- `k=60`: hằng số làm trơn — xem giải thích bên dưới

**Tại sao mẫu số là `k + rank` chứ không phải `rank`?**

Nếu không có `k`, rank 1 được ưu tiên quá lớn so với rank 2:

| rank | không có k | có k=60 |
|---|---|---|
| 1 | 1/1 = 1.000 | 1/61 = 0.0164 |
| 2 | 1/2 = 0.500 | 1/62 = 0.0161 |
| 3 | 1/3 = 0.333 | 1/63 = 0.0159 |

Với `k=60` → khoảng cách giữa các rank được làm mượt, tránh một retriever "thắng áp đảo" chỉ vì rank 1.

**Ví dụ:**

| Doc | Rank FAISS | Rank BM25 | RRF = 0.6/(60+r₁) + 0.4/(60+r₂) |
|---|---|---|---|
| Doc A | 1 | 3 | 0.6/61 + 0.4/63 = **0.01619** |
| Doc B | 5 | 1 | 0.6/65 + 0.4/61 = **0.01573** |
| Doc C | 1 | — | 0.6/61 + 0 = **0.00984** |

Doc A thắng vì cả 2 retrievers đều tìm ra. Doc C chỉ có FAISS → điểm thấp hơn.

**Code:**
```python
EnsembleRetriever(
    retrievers=[faiss_retriever, bm25_retriever],
    weights=[0.6, 0.4]   # HYBRID_VECTOR_WEIGHT, HYBRID_BM25_WEIGHT
)
```

---

### C4. Điểm số trong Hybrid RAG có ý nghĩa gì?

Score RRF được normalize về [0, 1] bằng **max lý thuyết**:

$$\text{Max lý thuyết} = \frac{w_\text{FAISS} + w_\text{BM25}}{k+1} = \frac{0.6 + 0.4}{61} = \frac{1}{61}$$

Đây là điểm của document đứng **rank 1 ở TẤT CẢ** retriever — điểm tốt nhất có thể đạt được.

$$\text{score\_norm} = \frac{\text{RRF}(d)}{\text{Max lý thuyết}} = \text{RRF}(d) \times 61$$

| Score | Nghĩa |
|---|---|
| ~100% | Rank 1 ở cả FAISS + BM25 |
| ~98% | Rank 1 FAISS + rank 2 BM25 |
| ~60% | Chỉ có ở FAISS (rank 1, w=0.6) |
| ~40% | Chỉ có ở BM25 (rank 1, w=0.4) |

> ⚠️ **Tại sao không dùng max batch?** Max batch = doc tốt nhất luôn hiện 100% kể cả khi kết quả đều kém. Max lý thuyết cho phép so sánh **tuyệt đối** giữa các query khác nhau.

**Code (rag_chain.py `_compute_rrf_scores`):**
```python
rrf_theoretical_max = sum(weights) / (k + 1)   # = 1/61 ≈ 0.01639
return [(doc, round(min(1.0, score / rrf_theoretical_max), 4))
        for doc, score in sorted_pairs]
```

---

# NHÓM D: RERANKING — CROSSENCODER

---

### D1. CrossEncoder khác Bi-encoder thế nào?

| | Bi-encoder (FAISS) | CrossEncoder |
|---|---|---|
| Input | encode(query) riêng, encode(doc) riêng | [CLS] query [SEP] doc [SEP] cùng lúc |
| Attention | Query không "thấy" doc | Full self-attention: query ↔ doc |
| Tốc độ | ✅ Nhanh (pre-compute doc) | ❌ Chậm (mỗi cặp chạy 1 lần) |
| Độ chính xác | Khá tốt | ✅ Chính xác hơn |
| Pre-compute | ✅ Có thể pre-compute | ❌ Không thể |

**Kiến trúc 2 giai đoạn:**
```
Stage 1 (Recall):   Bi-encoder FAISS   → fetch_k=50 → MMR → k=8   (~10-50ms)
Stage 2 (Precision): CrossEncoder      → 8 cặp → top_k=3           (~200-500ms)
```

---

### D2. CrossEncoder normalize score thế nào?

**Raw logit** từ CrossEncoder không có giới hạn → normalize bằng sigmoid:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

| Logit x | sigmoid(x) | Ý nghĩa |
|---|---|---|
| -5 | 0.007 | Gần như không liên quan |
| 0 | 0.500 | Trung lập |
| +2 | 0.881 | Liên quan |
| +5 | 0.993 | Rất liên quan |

**Lưu ý:** Score CrossEncoder ≠ Score RRF/cosine — không so sánh trực tiếp được.

---

### D3. Reranker nằm ở đâu trong pipeline?

Reranker LUÔN chạy CUỐI CÙNG, sau tất cả các mode RAG:
```
[Self-RAG] hoặc [Co-RAG] hoặc [Standard RAG]
        ↓ (nếu reranker_enabled=True)
doc_score_pairs → rerank_with_cross_encoder(query, pairs, top_k=3)
        ↓
score hiển thị = ce_score (CrossEncoder), không phải RRF
bi_encoder_score lưu riêng để so sánh
```

---

# NHÓM E: SELF-RAG

---

### E1. Self-RAG có mấy tầng? Mỗi tầng làm gì?

**3 tầng kiểm soát chất lượng:**

```
RAG thường: Query → Retrieve → Generate → Answer
Self-RAG:   Query → [Rewrite] → Retrieve → [Filter] → Generate → [Grade]
                    Tầng 1              Tầng 2                   Tầng 3
```

**Tầng 1 — Query Expansion** (TRƯỚC retrieve):
- LLM viết lại câu hỏi thành 3 phiên bản
- Retrieve với cả 4 queries (gốc + 3 bản)
- Hợp nhất + dedup bằng `page_content[:100]`
- Overhead: ~0.8–1.5s (1 LLM call)

**Tầng 2 — Relevance Filtering** (SAU retrieve):
- Với mỗi chunk: LLM trả lời "CÓ"/"KHÔNG" có liên quan không
- Safety: nếu tất cả bị loại → giữ `docs[:2]`
- Overhead: ~0.2–1.0s × số chunks

**Tầng 3 — Answer Grading** (SAU generate):
- LLM tự đánh giá câu trả lời vừa sinh
- Trả về JSON: `{score, is_grounded, has_hallucination, feedback}`
- Parse bằng regex `r'\{.*\}'` (DOTALL — xử lý LLM thêm text thừa)
- Fallback: `{score: 0.5, is_grounded: True, has_hallucination: False}`
- Overhead: ~1.3–1.7s (1 LLM call)

**Score hiển thị trong Self-RAG:** FAISS cosine similarity từ `similarity_search_with_scores()` → đã [0,1], hiển thị trực tiếp × 100.

---

### E2. Tổng overhead Self-RAG?

```
Tầng 1 (rewrite):   1 LLM call  ≈ 0.8–1.5s
Tầng 2 (filter):    k LLM calls ≈ 0.2–1.0s × k chunks
Tầng 3 (grade):     1 LLM call  ≈ 1.3–1.7s
──────────────────────────────────────────────
Tổng thêm vào:      +1.6s (câu đơn giản) → +3.8s (câu phức tạp)
```

---

# NHÓM F: CO-RAG

---

### F1. Co-RAG có mấy agents? Mỗi agent làm gì?

**3 agents chạy tuần tự** (không song song — FAISS không thread-safe trên CPU):

| Agent | Phương pháp | Score |
|---|---|---|
| **Agent 1** Semantic | `similarity_search_with_relevance_scores()` | FAISS cosine [0,1] thật |
| **Agent 2** Keyword (BM25) | `BM25Retriever.invoke(query)` | RRF formula: `61/(60+rank)` |
| **Agent 3** Conceptual | LLM phân rã → sub-questions → FAISS | FAISS cosine [0,1] thật |

**Agent 3 (Conceptual) pipeline:**
```
LLM(question) → sub_questions[:3]
for sub_q: similarity_search_with_relevance_scores(sub_q, k=5)
dedup bằng fingerprint = page_content[:120]
```

---

### F2. Score Co-RAG Agent 2 (BM25) tính thế nào?

**BM25 LangChain không expose score thật** → dùng RRF formula chuẩn hóa:

```python
_rrf_k = 60
_rrf_max = 1.0 / (_rrf_k + 1)   # = 1/61 (score lý thuyết tại rank 1)
approx_score = (1.0 / (_rrf_k + rank)) / _rrf_max  # = 61/(60+rank)
```

| Rank | Score |
|---|---|
| 1 | 61/61 = **1.000** |
| 2 | 61/62 = **0.984** |
| 3 | 61/63 = **0.968** |
| 5 | 61/65 = **0.938** |

> ⚠️ **Thay đổi từ phiên bản cũ:** trước đây dùng `1.0 - (i/n)*0.5` (tuyến tính tùy tiện). Giờ dùng RRF k=60 (có cơ sở lý thuyết, nhất quán với Hybrid RAG).

---

### F3. Consensus Merger tính merged_score thế nào?

**Bước 1:** Xây bảng vote theo fingerprint:
```python
doc_registry[page_content[:120]] = [(agent_name, doc, score), ...]
```

**Bước 2:** Tính merged score:
```python
vote_count = len(entries)                        # số agents tìm ra doc này
avg_score  = sum(s for _,_,s in entries) / vote_count
vote_boost = 1.0 + (vote_count - 1) * 0.15     # +15% per thêm 1 agent
merged_score = min(1.0, avg_score * vote_boost) # cap tại 1.0
```

**Bảng vote boost:**
| Số agents đồng thuận | vote_boost | Ý nghĩa |
|---|---|---|
| 1 | 1.00 | Không boost |
| 2 | 1.15 | +15% — 2 agents đồng ý |
| 3 | 1.30 | +30% — tất cả đồng ý |

**Bước 3:** Lọc `vote_count >= CO_RAG_MIN_VOTES=2` → loại noise 1 agent

---

### F4. Co-RAG vs Self-RAG: chọn khi nào?

| | Self-RAG | Co-RAG |
|---|---|---|
| Phù hợp với | Câu hỏi phức tạp, cần độ tin cậy cao | Câu hỏi đa góc độ, cần nhiều nguồn |
| Điểm nổi bật | Tự kiểm tra hallucination | Đồng thuận đa agent |
| Overhead | +1.6–3.8s | +1 LLM call (Agent 3) |
| Kết hợp cả hai? | ❌ MUTUALLY EXCLUSIVE (if/elif trong app.py) |

Lý do: 5-9 LLM calls/query trên CPU i5 = 30-45s. Không thực tế.

---

# NHÓM G: KIẾN TRÚC & LUỒNG APP

---

### G1. SmartDoc AI có kiến trúc mấy tầng?

```
┌──────────────────────────────────────┐
│  PRESENTATION: app.py (Streamlit)    │  Upload, Chat, Sidebar
├──────────────────────────────────────┤
│  APPLICATION:                        │  rag_chain.py, self_rag.py,
│  RAG Pipelines                       │  co_rag.py, reranker.py,
│                                      │  language_detector.py
├──────────────────────────────────────┤
│  DATA:                               │  vector_store.py (FAISS+BM25)
│  Storage & Processing                │  document_processor.py
│                                      │  vectorstore/ + data/uploads/
├──────────────────────────────────────┤
│  MODEL:                              │  qwen2.5:7b (Ollama)
│  AI Models                           │  LaBSE (embedding, 512 tok)
│                                      │  cross-encoder/mmarco-mMiniLMv2
└──────────────────────────────────────┘
```

---

### G2. Luồng xử lý trong app.py thế nào?

```
┌─────────────────────────────────────────────────────────┐
│                    Câu hỏi user                         │
└──────────────────────────┬──────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
     [if self_rag]   [elif co_rag]      [else]
           │               │               │
           ▼               ▼               ▼
  self_rag_pipeline() co_rag_pipeline() ask_question()
  rewrite → retrieve  3 agents          hybrid / FAISS
  → filter → grade    → merger          → LLM
           │               │               │
           └───────────────┴───────────────┘
                           │
              (3 nhánh loại trừ nhau, đổ vào đây)
                           │
           ┌───────────────┴───────────────┐
           │                               │
     [if reranker]                    [else — NO]
           │                               │
           ▼                               ▼
  rerank_with_cross_encoder()     giữ nguyên sources
  score = ce_score (sigmoid)      score từ nhánh trên
           │                               │
           └───────────────┬───────────────┘
                           │
                           ▼
           Hiển thị: answer + sources
           score_pct = min(100, int(score × 100))
```

**Ghi chú:**
- `self_rag` và `co_rag` dùng `if/elif` → **loại trừ nhau**, không chạy song song.
- Reranker chạy **sau** bất kỳ nhánh nào, là bước độc lập cuối cùng.

---

### G3. Score hiển thị trong từng mode?

| Mode | Score gốc | Normalize | Hiển thị |
|---|---|---|---|
| **Standard Hybrid** | RRF thô ~0.01x | ÷ max lý thuyết (1/61) | % tuyệt đối |
| **Standard FAISS** | Cosine [0,1] | × 100 | % tuyệt đối |
| **Self-RAG** | Cosine [0,1] | × 100 | % tuyệt đối |
| **Co-RAG** | Merged [0,1] | × 100 | % tuyệt đối |
| **+Reranking** | CrossEncoder logit | sigmoid → [0,1] → × 100 | % tuyệt đối |

**Code `render_sources()` (app.py):**
```python
score_pct = min(100, int(score * 100))   # không còn max_score normalization
```

---

# QUICK REFERENCE

---

## Bảng tham số đầy đủ

| Tham số | Giá trị | File | Ý nghĩa |
|---|---|---|---|
| OLLAMA_MODEL | `qwen2.5:7b` | config.py | LLM local |
| EMBEDDING_MODEL | `sentence-transformers/LaBSE` | config.py | 512 tokens, 768 dims |
| CHUNK_SIZE | 1500 | config.py | Ký tự tối đa/chunk |
| CHUNK_OVERLAP | 200 | config.py | Ký tự trùng lặp |
| RETRIEVAL_TOP_K | 8 | config.py | Chunks vào CrossEncoder |
| RETRIEVAL_FETCH_K | 50 | config.py | FAISS fetch trước MMR |
| RETRIEVAL_LAMBDA_MULT | 0.7 | config.py | λ trong MMR |
| HYBRID_VECTOR_WEIGHT | 0.6 | config.py | Trọng số FAISS trong RRF |
| HYBRID_BM25_WEIGHT | 0.4 | config.py | Trọng số BM25 trong RRF |
| HYBRID_TOP_K | 5 | config.py | Per-retriever trước ensemble |
| CO_RAG_MIN_VOTES | 2 | config.py | Ngưỡng đồng thuận |
| top_k CrossEncoder | 3 | app.py | Chunks cuối cùng |
| vote_boost coeff | 0.15 | co_rag.py | Boost per thêm 1 agent |
| k trong RRF | 60 | rag_chain.py | Hằng số làm trơn |

---

## Bảng công thức

| Công thức | Dùng cho | Key |
|---|---|---|
| $\frac{\vec{q}\cdot\vec{d}}{\|\vec{q}\|\|\vec{d}\|}$ | Cosine similarity | [-1,+1], 1=hoàn toàn giống |
| $\lambda \cdot \text{sim}(d,q) - (1-\lambda)\max\text{sim}(d,S)$ | MMR | λ=0.7, S=đã chọn |
| $\sum_i \frac{w_i}{60+\text{rank}_i}$ | RRF score | max=Σwᵢ/61≈0.01639 |
| $\frac{1}{1+e^{-x}}$ | Sigmoid (CrossEncoder) | 0→0.5, 5→0.993 |
| $\min(1.0,\, \bar{s} \times (1+(v-1)\times0.15))$ | Co-RAG merger | v=vote_count |

---

## Câu hỏi bẫy thường gặp

**Q: LaBSE khác mpnet cũ thế nào?**
A: LaBSE 512 tokens (gấp 4×), xử lý chunk dài mà không bị truncate. mpnet cũ 128 tokens.

**Q: weights=[0.6, 0.4] trong EnsembleRetriever có nghĩa là cộng 60% + 40% không?**
A: Không. Đây là trọng số điều chỉnh contribution trong RRF: `wᵢ/(k+rank)`. Không phải cộng điểm trực tiếp.

**Q: BM25 trong Co-RAG có score thật không?**
A: Không. BM25Retriever LangChain không expose score thật. Dùng RRF formula: `61/(60+rank)` — rank-based nhưng có cơ sở lý thuyết.

**Q: Tại sao 3 agents Co-RAG không chạy song song?**
A: FAISS không thread-safe trên CPU. ThreadPoolExecutor có thể gây lỗi.

**Q: Self-RAG và Co-RAG có thể bật cùng lúc không?**
A: Không — `if/elif` trong app.py đảm bảo mutually exclusive. Kết hợp = 5-9 LLM calls = 30-45s.

**Q: Reranker chạy trước hay sau Self-RAG/Co-RAG?**
A: LUÔN SAU (app.py ~line 2033, sau khi tất cả 3 nhánh if/elif/else xử lý xong).

**Q: Score 97% hybrid nghĩa là gì?**
A: Doc được tìm thấy bởi cả FAISS lẫn BM25, rank cao ở cả hai. Score 60% = chỉ 1 retriever tìm được.

**Q: fetch_k=50 rồi k=8 — bước gì ở giữa?**
A: MMR (`max_marginal_relevance_search`) — chọn 8 chunk đa dạng nhất từ 50 ứng viên.

---

*SmartDoc AI — Cập nhật: May 2026*
*Nguồn: modules/ + app.py + config.py (source code thực tế)*

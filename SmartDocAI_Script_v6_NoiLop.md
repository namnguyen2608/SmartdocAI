# SCRIPT THUYẾT TRÌNH TRƯỚC LỚP — SmartDoc AI (v6)
**Môn:** Phát Triển Phần Mềm Mã Nguồn Mở | **Lớp:** DCT1231 | **GVHD:** Từ Lãng Phiêu  
**Nhóm:** Lê Nguyễn Nhất Tâm · Lưu Hồng Phúc · Đặng Quang Phong · Nguyễn Phước Nam

---

> **Cách dùng file này:**
> - Đây là lời thoại thực tế khi đứng trước lớp — ngắn gọn, tự nhiên
> - Không cần học thuộc từng chữ, chỉ cần nắm ý chính và diễn đạt bằng lời của mình
> - **[NÓI]** — lời thoại gợi ý
> - **[CHUYỂN]** — câu dẫn sang slide tiếp theo
> - **⏱** — thời gian mục tiêu
> - Tổng: **~12–15 phút** (chưa tính Q&A)
> - → Xem **SmartDocAI_Script_v5.md** để đọc giải thích chi tiết phục vụ ôn luyện

---

## SLIDE 1 — Trang bìa
**⏱ 20 giây**

**[NÓI]**
Kính chào thầy và các bạn. Nhóm em xin trình bày đề tài **SmartDoc AI — hệ thống hỏi đáp tài liệu thông minh**, dưới sự hướng dẫn của thầy Từ Lãng Phiêu.

---

## SLIDE 2 — Thành viên nhóm
**⏱ 20 giây**

**[NÓI]**
Nhóm gồm bốn thành viên: Lê Nguyễn Nhất Tâm, Lưu Hồng Phúc, Đặng Quang Phong và Nguyễn Phước Nam.

**[CHUYỂN]**
*Trước tiên, tại sao nhóm em làm đề tài này?*

---

## SLIDE 3 — Vấn đề đặt ra
**⏱ 1 phút**

**[NÓI]**
Có ba vấn đề thực tế thúc đẩy dự án.

Một là **mất thời gian** — mỗi ngày phải đọc hàng trăm trang tài liệu để tìm một thông tin nhỏ.

Hai là **bảo mật** — dùng ChatGPT phải upload tài liệu nội bộ lên server nước ngoài, rủi ro rò rỉ dữ liệu.

Ba là **hallucination** — AI trả lời tự tin nhưng sai hoàn toàn, không có nguồn trích dẫn.

Từ đó nhóm đặt câu hỏi: *Có thể xây dựng AI đọc tài liệu — chính xác, an toàn, chạy offline trên máy cá nhân không?*

**[CHUYỂN]**
*Câu trả lời là có — và đây là SmartDoc AI.*

---

## SLIDE 4 — Giải pháp & 5 đóng góp
**⏱ 50 giây**

**[NÓI]**
SmartDoc AI chạy **hoàn toàn offline** trên nền kiến trúc RAG, với 5 đóng góp kỹ thuật:

**Một — Hybrid Search:** kết hợp FAISS và BM25, dưới 50ms.  
**Hai — CrossEncoder Reranking:** xếp hạng lại kết quả chính xác hơn.  
**Ba — Self-RAG 3 tầng:** tự kiểm soát chất lượng, phát hiện hallucination.  
**Bốn — Co-RAG 3 agent:** tăng recall +16% cho câu hỏi đa bước.  
**Năm — Citation Tracking:** trích dẫn tên file và số trang chính xác.

**[CHUYỂN]**
*Hãy xem từng đóng góp này giải quyết vấn đề gì.*

---

## SLIDE 5 — Nền tảng RAG & công nghệ
**⏱ 1 phút 30 giây**

**[NÓI]**
Nền tảng là **RAG — Retrieval-Augmented Generation**. Ý tưởng đơn giản: thay vì để LLM trả lời từ kiến thức nội tại và có thể bịa, ta **buộc nó phải đọc tài liệu trước rồi mới trả lời** — đây là cơ chế chống hallucination cơ bản nhất.

Luồng xử lý gồm 5 bước: tài liệu PDF/DOCX được chia nhỏ thành chunk 1500 ký tự, chuyển thành vector 768 chiều, lưu vào FAISS và BM25 index. Khi người dùng hỏi, hệ thống truy xuất chunk liên quan nhất rồi đưa vào Qwen2.5:7b để sinh câu trả lời.

Bốn công nghệ cốt lõi:
- **Qwen2.5:7b qua Ollama** — 7B tham số, context 128K, xử lý tiếng Việt tốt, chạy local hoàn toàn
- **FAISS** — tìm kiếm vector trong mili giây, MMR đảm bảo kết quả đa dạng
- **Multilingual MPNet** — embedding 768 chiều, hỗ trợ tiếng Việt
- **LangChain + Streamlit** — điều phối pipeline và giao diện

**[CHUYỂN]**
*RAG cơ bản còn 4 hạn chế — nhóm giải quyết từng cái.*

---

## SLIDE 6 — Thách thức 1 → Hybrid Search
**⏱ 1 phút 30 giây**

**[NÓI]**
**Hạn chế:** FAISS thuần túy có thể bỏ sót tài liệu chứa từ khóa kỹ thuật đặc thù như "Điều 15 Nghị định 100" — vì vector không đủ tương đồng. Ngược lại BM25 thuần chỉ khớp từ giống hệt — hỏi "phương tiện giao thông" không tìm được "xe cộ".

**Giải pháp — Hybrid Search:** kết hợp cả hai qua **Reciprocal Rank Fusion**. Thay vì cộng điểm trực tiếp — vốn không được vì hai thang điểm khác đơn vị — RRF chỉ dùng **thứ hạng**: tài liệu nào được cả FAISS lẫn BM25 xếp cao đều sẽ nổi lên trên. Trọng số: FAISS 0.6, BM25 0.4 — xác định qua thực nghiệm.

Kết quả: dưới 50ms với 5.000 chunks, vừa hiểu ngữ nghĩa vừa không bỏ sót từ khóa.

**[CHUYỂN]**
*Retrieve đúng rồi — nhưng trong 8 kết quả đó, thứ tự xếp hạng có thực sự đúng không?*

---

## SLIDE 7 — Thách thức 2 → CrossEncoder Reranking
**⏱ 1 phút**

**[NÓI]**
**Hạn chế:** Bi-encoder của FAISS mã hóa câu hỏi và tài liệu **độc lập** — không có cross-attention giữa hai văn bản, đôi khi xếp sai thứ tự.

**Giải pháp — kiến trúc 2 giai đoạn:**
- Giai đoạn 1: FAISS fetch nhanh **top-8** (RETRIEVAL_TOP_K=8)
- Giai đoạn 2: CrossEncoder nhận cả câu hỏi **lẫn** tài liệu cùng lúc, tính full attention giữa từng từ — chính xác hơn nhiều, nhưng chỉ áp dụng cho 8 candidates đã lọc.

Kết quả: **top-3** tốt nhất (top_k=3) được đưa vào Prompt cho LLM. Điểm chuẩn hóa về [0, 1] — người dùng thấy "độ liên quan 87%" thay vì số vô nghĩa.

*Ví dụ dễ nhớ: Bi-encoder giống tìm người quen bằng ảnh chụp riêng lẻ. Cross-encoder giống đứng trước mặt nói chuyện trực tiếp — tốn hơn nhưng nhận ra chính xác hơn.*

**[CHUYỂN]**
*Retrieve đúng, xếp hạng đúng — nhưng LLM vẫn có thể hallucinate. Làm sao kiểm soát?*

---

## SLIDE 8 — Thách thức 3 → Self-RAG
**⏱ 1 phút 30 giây**

**[NÓI]**
**Hạn chế:** RAG chuẩn không có cơ chế tự kiểm soát — chunk không liên quan lọt vào Prompt, LLM vẫn cố tổng hợp và hallucinate.

**Giải pháp — Self-RAG 3 tầng**, mỗi tầng can thiệp tại một điểm khác nhau:

**Tầng 1 — Query Expansion** *(trước khi retrieve)*: LLM viết lại câu hỏi thành 3 phiên bản từ góc độ khác nhau — tổng 4 queries được retrieve lần lượt, hợp nhất và dedup. Tăng khả năng tìm đúng tài liệu. Tốn 0.8–1.5 giây.

**Tầng 2 — Relevance Grading** *(sau retrieve)*: LLM chấm từng document "CÓ liên quan" hay "KHÔNG" — loại nhiễu trước khi vào Prompt. Tốn 0.2–1.0 giây.

**Tầng 3 — Answer Grading** *(sau generation)*: LLM tự đánh giá câu trả lời vừa sinh, trả về JSON với điểm tin cậy, is_grounded, has_hallucination và feedback. Người dùng thấy điểm này trên giao diện. Tốn 1.3–1.7 giây.

Overhead tổng: +1.6 đến +3.8 giây — chấp nhận được trên CPU không GPU.

**[CHUYỂN]**
*Self-RAG tốt cho câu đơn. Còn câu hỏi đa bước cần tổng hợp nhiều phần tài liệu?*

---

## SLIDE 9 — Thách thức 4 → Co-RAG
**⏱ 1 phút 15 giây**

**[NÓI]**
**Hạn chế:** Câu hỏi phức tạp như *"So sánh cách xử lý lỗi phần A với phương pháp phần B"* — một vector duy nhất phải biểu diễn nhiều khía cạnh cùng lúc, không khía cạnh nào đủ sắc nét.

**Giải pháp — Co-RAG 3 agent hoạt động song song:**
- **Agent 1 (FAISS):** tìm theo ngữ nghĩa tổng thể
- **Agent 2 (BM25):** khớp từ khóa kỹ thuật chính xác
- **Agent 3 (LLM):** phân rã câu hỏi thành tối đa 3 sub-questions nguyên tử, retrieve riêng từng cái — vector sắc nét hơn nhiều

Ba kết quả hợp nhất qua **Vote Boost**: `Score_final = avg_score × (1 + (v−1) × 0.15)`. Tài liệu được 2 agent đồng thuận: +15%. Cả 3 đồng thuận: +30%. Ngưỡng MIN_VOTES=2 loại bỏ false positive.

Kết quả: recall **72% → 88%** (+16%) trên 20 câu hỏi đa bước. Đánh đổi 1.5–4 giây.

**[CHUYỂN]**
*Bốn giải pháp ghép lại thành một kiến trúc hoàn chỉnh.*

---

## SLIDE 10 — Kiến trúc hệ thống
**⏱ 45 giây**

**[NÓI]**
Pipeline xử lý một câu hỏi qua 7 bước: Self-RAG T1 mở rộng query → Co-RAG 3 agent gather candidates → Hybrid Search RRF tổng hợp → CrossEncoder lọc top-8 xuống top-3 → Self-RAG T2&T3 lọc nhiễu và kiểm hallucination → Qwen2.5:7b sinh câu trả lời kèm citation.

Kiến trúc 4 tầng: Presentation (Streamlit), Application (LangChain), Data (FAISS + BM25), Model (Qwen2.5 + CrossEncoder) — mỗi tầng có thể nâng cấp độc lập.

**[CHUYỂN]**
*Kiến trúc đã rõ — nó hoạt động nhanh như thế nào trong thực tế?*

---

## SLIDE 11 — Thực nghiệm — Hiệu năng retrieval
**⏱ 1 phút 15 giây**

**[NÓI]**
Đo latency 3 phương pháp trên 1.000–10.000 chunks, trung bình **25 lượt đo** (5 queries × 5 runs), phần cứng i5-12400F không GPU.

Ba điểm nổi bật:

**FAISS hoàn toàn ổn định** — 35–37ms bất kể index tăng 10 lần. Lý do: bước encode câu hỏi chiếm 95%+ tổng latency, thời gian search vector chỉ 0.1–1.5ms.

**BM25 tăng tuyến tính** — 1.48ms lên 21.89ms. Cần lưu ý khi scale lớn.

**Hybrid** — 38–61ms, tăng đều, luôn dưới 50ms ở 5.000 chunks — đây là lựa chọn mặc định vì chất lượng context tốt nhất.

Ngoài ra: Co-RAG cải thiện recall +16%, đánh đổi 1.5–4 giây.

**[CHUYỂN]**
*Self-RAG overhead và chiến lược chunking?*

---

## SLIDE 12 — Self-RAG Overhead & Chunking
**⏱ 1 phút**

**[NÓI]**
**Overhead Self-RAG:** câu đơn giản +2.7 giây, câu trung bình +1.6 giây, câu phức tạp +3.8 giây. Câu trung bình ít overhead hơn vì Tầng 2 lọc được nhiều document, context ngắn lại, LLM generate nhanh hơn.

**Chunking:** Thử 4 kích thước trên cùng tập tài liệu. Chunk 1.500 ký tự overlap 200 là tối ưu — tương đương 300–400 tokens tiếng Việt, đủ giữ một ý hoàn chỉnh, đủ nhỏ để vector đặc trưng cho một chủ đề cụ thể.

**[CHUYỂN]**
*Tính đúng đắn được bảo đảm bởi bộ kiểm thử tự động.*

---

## SLIDE 13 — Kiểm thử 141 test cases
**⏱ 45 giây**

**[NÓI]**
**141 test cases — tất cả passed** — gồm 114 unit tests và 27 integration tests trên 7 module.

Unit tests chạy offline dưới 1 giây bằng cách mock LLM — phù hợp CI/CD. Integration tests tự động skip nếu Ollama chưa chạy — không làm gián đoạn pipeline.

Coverage đầy đủ 11 yêu cầu kỹ thuật Q1–Q11.

**[CHUYỂN]**
*Tổng kết và hướng tiếp theo.*

---

## SLIDE 14 — Kết luận & Hướng phát triển
**⏱ 45 giây**

**[NÓI]**
**Đạt được:** RAG offline hoàn toàn, Hybrid Search <50ms, CrossEncoder top-8→top-3, Self-RAG 3 tầng, Co-RAG +16% recall, 141 tests passed.

**Hạn chế:** LLM 2–8 giây/turn trên CPU, chỉ hỗ trợ PDF và DOCX, BM25 phải rebuild toàn bộ khi thêm file.

**Hướng phát triển:** mở rộng sang Excel/PowerPoint, tối ưu batch embedding, cải thiện UI quản lý tài liệu.

---

## SLIDE 15 — Cảm ơn
**⏱ 15 giây**

**[NÓI]**
Đó là toàn bộ nội dung trình bày của nhóm em về SmartDoc AI.

Cảm ơn thầy và các bạn đã lắng nghe. Nhóm em rất mong nhận được câu hỏi và nhận xét.

---

## GỢI Ý TRẢ LỜI CÂU HỎI THƯỜNG GẶP

**Q: Tại sao chọn Qwen2.5:7b?**  
Hiệu năng tiếng Việt tốt nhất ở kích thước 7B trong benchmark cộng đồng. Context 128K tokens. Chạy ổn định trên CPU qua Ollama — thử Llama 3 8B cho thấy inference chậm hơn trên cùng phần cứng.

---

**Q: RRF ưu điểm gì so với lấy trung bình điểm FAISS và BM25?**  
Điểm FAISS nằm trong [-1, 1], điểm BM25 không bị chặn trên — không thể cộng trực tiếp. RRF chỉ dùng **thứ hạng** nên không cần chuẩn hóa, tránh bias hoàn toàn.

---

**Q: FAISS latency tại sao không tăng khi data tăng 10 lần?**  
Bước encode câu hỏi bằng MPNet chiếm 95%+ tổng latency (~35–37ms cố định). Thời gian search HNSW chỉ 0.1–1.5ms — quá nhỏ để thấy trên biểu đồ.

---

**Q: Khi nào bật Self-RAG, khi nào tắt?**  
Bật khi cần độ chính xác cao: tài liệu pháp lý, báo cáo quan trọng, câu hỏi phức tạp. Tắt khi cần phản hồi nhanh hoặc câu hỏi tra cứu đơn giản.

---

**Q: Tại sao Co-RAG và Self-RAG không dùng cùng lúc?**  
Kết hợp cả hai tạo 5–9 LLM calls mỗi query. Trên CPU không GPU, tổng có thể 30–45 giây — không thực tế. Hai chế độ mutually exclusive.

---

**Q: Vote Boost ý nghĩa gì?**  
`Score_final = avg_score × (1 + (v−1) × 0.15)`. Tài liệu được 2 agent độc lập xác nhận tăng 15%, 3 agent tăng 30%. Hệ số 0.15 đủ có ý nghĩa nhưng không lấn át chất lượng — document v=1 điểm 0.8 vẫn cao hơn document v=2 điểm 0.6 (→ 0.69).

---

**Q: Tiếng Việt xử lý thế nào?**  
Hai cơ chế: MPNet-multilingual tạo embedding phản ánh ngữ nghĩa tiếng Việt. Module `language_detector` đếm tỷ lệ ký tự có dấu để chọn prompt template đúng ngôn ngữ.

---

**Q: BM25 có cập nhật khi thêm file mới không?**  
Phải rebuild toàn bộ vì IDF thay đổi khi corpus thay đổi. FAISS hỗ trợ incremental update. Giải pháp dài hạn: Elasticsearch/OpenSearch — đây là một trong hướng phát triển tiếp theo.

---

*SmartDoc AI  •  DCT1231  •  2025–2026*

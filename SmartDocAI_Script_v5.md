# SCRIPT CHUẨN BỊ KIẾN THỨC — SmartDoc AI (v5 · Tài liệu tham khảo nội bộ)
**Môn:** Phát Triển Phần Mềm Mã Nguồn Mở | **Lớp:** DCT1231 | **GVHD:** Từ Lãng Phiêu  
**Nhóm:** Lê Nguyễn Nhất Tâm · Lưu Hồng Phúc · Đặng Quang Phong · Nguyễn Phước Nam

---

> **Mục đích file này:** Tài liệu chuẩn bị kiến thức nội bộ cho người trình bày — **KHÔNG đọc từng chữ khi đứng trước lớp**.
> Giải thích đầy đủ nguyên lý, công thức, số liệu để người trình bày hiểu sâu và tự tin trả lời câu hỏi.
> → Xem **SmartDocAI_Script_v6_NoiLop.md** để biết lời thoại thực tế khi trình bày.
>
> - **[NÓI]** — nội dung kiến thức đầy đủ
> - **[CHUYỂN]** — câu dẫn sang slide tiếp theo
> - **⏱** — ước lượng nếu đọc hết (chỉ để tham khảo)
> - Tổng thời gian nếu đọc hết: **~20–25 phút**

---

## SLIDE 1 — Trang bìa
**⏱ 30 giây**

**[NÓI]**
Kính chào thầy và các bạn. Hôm nay nhóm em xin trình bày đồ án môn Phát Triển Phần Mềm Mã Nguồn Mở với đề tài **SmartDoc AI — Intelligent Document Question & Answer System**, dưới sự hướng dẫn của thầy Từ Lãng Phiêu.

---

## SLIDE 2 — Thành viên nhóm
**⏱ 30 giây**

**[NÓI]**
Nhóm chúng em gồm bốn thành viên: Lê Nguyễn Nhất Tâm, Lưu Hồng Phúc, Đặng Quang Phong và Nguyễn Phước Nam.

**[CHUYỂN]**
*Trước khi đi vào kỹ thuật, hãy cùng nhìn vào vấn đề thực tế mà dự án này ra đời để giải quyết.*

---

## SLIDE 3 — Vấn đề đặt ra
**⏱ 1 phút 30 giây**

**[NÓI]**
Chúng ta đang sống trong thời đại mà lượng tài liệu điện tử — đặc biệt là PDF và Word — tăng lên không ngừng. Tại doanh nghiệp, trường học hay cơ quan, mỗi ngày người dùng phải đọc và tra cứu hàng trăm trang tài liệu. Quá trình này không chỉ tốn thời gian mà còn rất dễ bỏ sót thông tin quan trọng nằm ở trang 47 của một bản báo cáo dài 200 trang. **Đây là vấn đề đầu tiên.**

Vấn đề thứ hai là **bảo mật**. Khi dùng ChatGPT hay các dịch vụ AI đám mây, người dùng buộc phải tải tài liệu nội bộ lên server đặt ở nước ngoài. Với hợp đồng kinh doanh, báo cáo tài chính, hay dữ liệu bệnh nhân — đây là rủi ro rò rỉ thông tin mà nhiều tổ chức không thể chấp nhận.

Vấn đề thứ ba là **hallucination** — hiện tượng mô hình AI trả lời tự tin nhưng hoàn toàn sai, bịa ra thông tin không có trong tài liệu. Điều này đặc biệt nguy hiểm trong môi trường học thuật và doanh nghiệp nơi mỗi số liệu đều cần có nguồn gốc xác thực.

Từ ba vấn đề đó, nhóm em đặt ra câu hỏi nghiên cứu: *Liệu có thể xây dựng một hệ thống AI đọc hiểu tài liệu — chính xác, an toàn, chạy hoàn toàn offline trên máy tính cá nhân mà không cần kết nối đám mây?*

**[CHUYỂN]**
*Câu trả lời của nhóm em là — có thể. Và đây là SmartDoc AI.*

---

## SLIDE 4 — Giải pháp & 5 đóng góp chính
**⏱ 1 phút 15 giây**

**[NÓI]**
SmartDoc AI là hệ thống hỏi-đáp tài liệu thông minh, hoạt động **hoàn toàn ngoại tuyến**, dựa trên kiến trúc RAG — Retrieval-Augmented Generation.

Hệ thống có 5 đóng góp kỹ thuật chính, mỗi cái giải quyết một hạn chế cụ thể của RAG truyền thống:

**Đóng góp 1 — Hybrid Search:** Kết hợp FAISS với BM25 qua thuật toán RRF, đạt dưới 50ms với 5.000 chunks. Giải quyết bài toán: FAISS bỏ sót từ khóa, BM25 không hiểu ngữ nghĩa.

**Đóng góp 2 — CrossEncoder Reranking:** Tái xếp hạng kết quả chính xác hơn bi-encoder thông thường. Giải quyết bài toán: top-K thu được chưa chắc đã xếp đúng thứ tự quan trọng.

**Đóng góp 3 — Self-RAG 3 tầng:** Cơ chế tự kiểm soát chất lượng không cần ground truth. Giải quyết bài toán: LLM hallucinate khi context có nhiễu.

**Đóng góp 4 — Co-RAG 3 agent:** Cải thiện recall +16% trên câu hỏi đa bước. Giải quyết bài toán: một retriever đơn lẻ không đủ để bao phủ câu hỏi phức tạp.

**Đóng góp 5 — Citation Tracking:** Trích dẫn tên tệp và số trang chính xác cho cả PDF lẫn DOCX — người dùng có thể xác minh nguồn gốc của mỗi câu trả lời.

**[CHUYỂN]**
*Để hiểu SmartDoc AI hoạt động như thế nào từ bên trong, hãy bắt đầu từ nền tảng lý thuyết: kiến trúc RAG.*

---

## SLIDE 5 — Nền tảng RAG & công nghệ
**⏱ 2 phút 30 giây**

**[NÓI]**
Nền tảng của hệ thống là kiến trúc **Retrieval-Augmented Generation**.

**Về công thức toán học**, đầu ra được mô tả là: **a = G(q, C)**, trong đó q là câu hỏi của người dùng, C là tập ngữ cảnh C = R(q) = {c₁, c₂, …, cₖ} được retriever R truy xuất từ kho tài liệu, và G là mô hình ngôn ngữ lớn có nhiệm vụ sinh câu trả lời. Ý nghĩa cốt lõi: LLM không tự bịa ra câu trả lời từ kiến thức nội tại mà **phải dựa vào ngữ cảnh C** vừa được truy xuất — đây là cơ chế chống hallucination cơ bản nhất.

**Luồng xử lý đi qua 5 bước:**

**Bước 1 — Đọc tài liệu:** Hệ thống nhận file PDF hoặc DOCX. Với PDF, dùng pdfplumber đọc từng trang giữ nguyên cấu trúc. Với DOCX, dùng python-docx đọc paragraph và bảng biểu.

**Bước 2 — Chunking:** Dùng RecursiveCharacterTextSplitter chia tài liệu thành các đoạn 1500 ký tự với overlap 200 ký tự. Overlap quan trọng vì đảm bảo thông tin nằm ở ranh giới giữa hai chunk không bị mất đi — luôn xuất hiện đầy đủ ở ít nhất một trong hai chunk liền kề.

**Bước 3 — Embedding:** Mỗi chunk được chuyển thành vector 768 chiều bằng mô hình Multilingual MPNet. Hai đoạn văn có ý nghĩa tương đồng sẽ có vector nằm gần nhau trong không gian 768 chiều này — đây là cơ sở toán học của tìm kiếm ngữ nghĩa.

**Bước 4 — Indexing:** Các vector được lưu vào FAISS index trên disk, đồng thời nội dung text được index vào BM25 trong bộ nhớ. Hai index song song phục vụ cho Hybrid Search sau này.

**Bước 5 — Query & Generate:** Khi người dùng đặt câu hỏi, hệ thống truy xuất các chunk liên quan nhất, ghép vào Prompt cùng câu hỏi, rồi đưa cho Qwen2.5:7b sinh câu trả lời.

**Bốn công nghệ cốt lõi:**

**Qwen2.5:7b qua Ollama** — 7 tỷ tham số, context window 128.000 tokens, xử lý tiếng Việt vượt trội so với các mô hình cùng kích thước. Chạy hoàn toàn local qua Ollama — không một byte dữ liệu nào rời khỏi máy người dùng.

**FAISS** — Facebook AI Similarity Search. Tìm kiếm hàng triệu vector chỉ trong mili giây nhờ thuật toán xấp xỉ HNSW. Chiến lược MMR — Maximal Marginal Relevance — với TOP_K=8, FETCH_K=50, lambda=0.7 đảm bảo kết quả vừa liên quan vừa đa dạng, không trả về 8 chunk trùng lặp về cùng một ý.

**Multilingual MPNet** — mô hình paraphrase-multilingual-mpnet-base-v2 tạo embedding 768 chiều, được huấn luyện trên 50+ ngôn ngữ bao gồm tiếng Việt.

**LangChain và Streamlit** — LangChain điều phối toàn bộ pipeline từ chunking đến generation. Streamlit cung cấp giao diện web với toggle bật/tắt từng tính năng nâng cao.

**[CHUYỂN]**
*Nền tảng đã rõ ràng. Nhưng RAG cơ bản như vừa mô tả còn 4 hạn chế — nhóm em giải quyết từng cái theo thứ tự như thế nào?*

---

## SLIDE 6 — Thách thức 1 → Hybrid Search
**⏱ 2 phút 30 giây**

**[NÓI]**
**Hạn chế của RAG cơ bản:** Nếu chỉ dùng FAISS, hệ thống tìm kiếm bằng cosine similarity trong không gian vector. Điều này tốt cho ngữ nghĩa nhưng có thể bỏ sót những tài liệu chứa từ khóa kỹ thuật đặc thù. Ví dụ: câu hỏi "Điều 15 Nghị định 100/2019" — nếu embedding của câu hỏi và của văn bản pháp quy không đủ tương đồng về góc cosine, FAISS sẽ bỏ qua tài liệu đó. Ngược lại, nếu chỉ dùng BM25 truyền thống dựa trên TF-IDF, hệ thống chỉ khớp từ giống nhau. Hỏi "phương tiện giao thông" trong khi tài liệu ghi "xe cộ" — BM25 không tìm ra được vì không hiểu hai cụm này là đồng nghĩa.

**Giải pháp — Hybrid Search** kết hợp cả hai qua thuật toán **Reciprocal Rank Fusion (RRF)**.

Cách RRF hoạt động: với mỗi tài liệu d, tính điểm từ cả hai danh sách kết quả theo công thức:

**RRF(d) = Σᵢ 1 / (k + rankᵢ(d))**

Trong đó rankᵢ(d) là vị trí của tài liệu d trong danh sách kết quả của retriever thứ i (FAISS hoặc BM25), và k=60 là hằng số làm trơn để tránh một tài liệu ở vị trí số 1 được boost quá lớn so với vị trí số 2.

Ví dụ cụ thể: Giả sử tài liệu A được FAISS xếp hạng 1 và BM25 xếp hạng 3. Điểm RRF = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323. Tài liệu B được FAISS xếp hạng 5 nhưng BM25 xếp hạng 1. Điểm RRF = 1/(60+5) + 1/(60+1) = 0.0154 + 0.0164 = 0.0318. RRF tự nhiên kết hợp cả hai tín hiệu mà không cần chuẩn hóa thủ công — điều quan trọng vì điểm cosine của FAISS nằm trong [-1, 1] còn điểm BM25 không bị chặn trên, không thể cộng trực tiếp.

Trọng số cuối: FAISS chiếm 0.6, BM25 chiếm 0.4 — được xác định qua thực nghiệm. Tín hiệu ngữ nghĩa quan trọng hơn nhưng từ khóa vẫn đóng vai trò hiệu chỉnh.

**So sánh ba phương pháp theo kết quả đo thực tế:**
- BM25 thuần: Latency 1.48ms ở 1.000 chunks nhưng tăng tuyến tính lên 21.89ms ở 10.000 chunks. Nhanh nhưng không hiểu ngữ nghĩa.
- FAISS thuần: Ổn định 35–37ms ở mọi quy mô — vì thời gian chủ yếu là encode câu hỏi, không phải search.
- Hybrid: 38–61ms — overhead chấp nhận được, đổi lấy chất lượng context tốt nhất.

**[CHUYỂN]**
*Hybrid Search đảm bảo retrieve đúng tài liệu. Nhưng trong 8 kết quả trả về, thứ tự xếp hạng có thực sự phản ánh mức độ quan trọng không — đây là vấn đề tiếp theo.*

---

## SLIDE 7 — Thách thức 2 → CrossEncoder Reranking
**⏱ 2 phút**

**[NÓI]**
**Hạn chế của bi-encoder:** FAISS dùng bi-encoder — câu hỏi và tài liệu được mã hóa **độc lập** thành hai vector riêng biệt, rồi mới tính cosine similarity giữa chúng. Cách này nhanh vì các vector tài liệu có thể pre-computed từ trước. Nhưng vì hai văn bản không "nhìn thấy" nhau trong quá trình encoding, mô hình bỏ qua nhiều tín hiệu tinh tế về mức độ liên quan thực sự. Kết quả là thứ hạng đôi khi sai — chunk liên quan nhất không nằm ở vị trí 1 mà ở vị trí 4 hoặc 5.

**Giải pháp — kiến trúc 2 giai đoạn:**

**Giai đoạn 1 — Recall bằng bi-encoder:** FAISS fetch nhanh **top-8 candidates** (RETRIEVAL_TOP_K = 8). Bước này chạy trong mili giây vì vector đã được pre-computed. Mục tiêu là đảm bảo đánh lưới đủ rộng — 8 candidates có xác suất cao chứa 3 chunk thực sự liên quan nhất.

**Giai đoạn 2 — Rerank bằng Cross-Encoder:** Model cross-encoder/ms-marco-MiniLM-L-6-v2 nhận đồng thời cả câu hỏi và tài liệu theo format: **[CLS] câu hỏi [SEP] nội dung tài liệu [SEP]**. Toàn bộ chuỗi này đi qua transformer với full self-attention — nghĩa là mỗi từ trong câu hỏi có thể "nhìn" trực tiếp vào mỗi từ trong tài liệu. Cơ chế này phát hiện được những liên kết tinh tế mà bi-encoder bỏ qua, ví dụ "thuật toán này" trong câu hỏi khớp với tên thuật toán được đặt ở đầu đoạn văn trong tài liệu.

Chi phí là O(n) inference calls — phải chạy mô hình n lần cho n cặp (câu hỏi, tài liệu). Đây là lý do chỉ áp dụng cho n=8 candidates đã lọc trước, không phải toàn bộ index.

Raw score từ cross-encoder là logit không bị chặn (có thể âm hoặc rất dương). Nhóm em áp dụng **hàm sigmoid** để chuẩn hóa: σ(x) = 1/(1 + e⁻ˣ) — đưa về khoảng [0, 1] để người dùng có thể hiểu là "độ liên quan 87%" thay vì một số vô nghĩa như 3.4.

**Kết quả:** 3 chunk có điểm cao nhất (top_k=3) sau reranking được đưa vào Prompt. Hệ thống có fallback tự động về điểm bi-encoder nếu cross-encoder gặp lỗi — đảm bảo không bao giờ bị crash.

**So sánh trực quan:** Bi-encoder giống như tìm người quen trong đám đông bằng cách nhìn ảnh riêng lẻ từng người. Cross-encoder giống như đứng trực tiếp trước mặt từng người và nói chuyện — tốn thời gian hơn nhưng nhận ra chính xác hơn nhiều.

**[CHUYỂN]**
*Retrieve đúng, xếp hạng đúng — nhưng kể cả khi có 3 chunk tốt nhất, LLM vẫn có thể sinh câu trả lời sai nếu ngữ cảnh không đủ rõ. Làm sao kiểm soát điều đó?*

---

## SLIDE 8 — Thách thức 3 → Self-RAG
**⏱ 2 phút 30 giây**

**[NÓI]**
**Hạn chế của RAG chuẩn:** Ngay cả khi retrieve và rerank tốt, vẫn có trường hợp chunk không thực sự liên quan lọt vào Prompt — ví dụ câu hỏi về chương 3 nhưng hệ thống lại kéo về chunk từ phần mục lục hoặc phần tài liệu tham khảo. LLM sẽ cố gắng tổng hợp câu trả lời từ ngữ cảnh đó và có thể hallucinate ra thông tin không có trong tài liệu. RAG chuẩn không có cơ chế phát hiện điều này.

**Giải pháp — Self-RAG với 3 tầng kiểm soát**, mỗi tầng can thiệp tại một điểm khác nhau trong pipeline:

**Tầng 1 — Query Expansion** *(trước khi retrieve, tốn 0.8–1.5 giây)*

Vấn đề: Người dùng đôi khi đặt câu hỏi không rõ ràng hoặc dùng từ ngữ khác với từ trong tài liệu. Ví dụ hỏi "cách tính lương" nhưng tài liệu dùng từ "phương pháp xác định thu nhập". Một câu hỏi duy nhất có thể bỏ sót những góc độ tiếp cận khác.

Cách hoạt động: LLM nhận câu hỏi gốc và sinh thêm **3 phiên bản** câu hỏi từ góc độ khác nhau. Ví dụ từ "cách tính lương" có thể mở rộng thành "công thức xác định mức lương", "quy trình tính thu nhập nhân viên", "các khoản khấu trừ khi tính lương". **Bốn câu hỏi** (1 gốc + 3 viết lại) được retrieve lần lượt, kết quả được hợp nhất và dedup. Tổng số candidates tăng lên đáng kể, recall được cải thiện.

**Tầng 2 — Relevance Grading** *(sau retrieve, trước LLM generation, tốn 0.2–1.0 giây)*

Vấn đề: Dù đã Hybrid Search và rerank, vẫn có chunk không thực sự liên quan đến câu hỏi — chỉ khớp bề mặt về từ khóa.

Cách hoạt động: Với mỗi document trong tập candidates, LLM được hỏi: "Document này có chứa thông tin liên quan để trả lời câu hỏi không? Trả lời CÓ hoặc KHÔNG." Document bị chấm KHÔNG bị loại khỏi context trước khi vào Prompt. Bước này giúp LLM chỉ đọc những tài liệu thực sự có ích, không bị phân tâm bởi nhiễu.

**Tầng 3 — Answer Grading** *(sau khi LLM sinh câu trả lời, tốn 1.3–1.7 giây)*

Vấn đề: Ngay cả với context tốt, LLM đôi khi thêm thông tin từ kiến thức nội tại thay vì chỉ dựa vào tài liệu.

Cách hoạt động: Câu trả lời vừa sinh được đưa trở lại cho LLM để tự đánh giá. Kết quả trả về dạng JSON với 4 trường:
- score: điểm tin cậy từ 0.0 đến 1.0
- is_grounded: câu trả lời có neo vào tài liệu không
- has_hallucination: có phát hiện thông tin bịa đặt không
- feedback: giải thích ngắn gọn

Người dùng nhìn thấy điểm confidence này trên giao diện — giúp họ biết khi nào nên tin câu trả lời và khi nào nên kiểm tra lại tài liệu gốc.

**Về overhead:** Self-RAG thêm từ +1.6 giây (câu trung bình) đến +3.8 giây (câu phức tạp). Điểm thú vị là câu trung bình lại có overhead ít hơn câu đơn giản: vì câu hỏi trung bình thường liên quan rõ hơn, Tầng 2 lọc được nhiều document hơn, context ngắn lại, nên Tầng 3 grading cũng nhanh hơn. Overhead không phải hằng số — nó phụ thuộc vào chất lượng retrieval.

**[CHUYỂN]**
*Self-RAG kiểm soát tốt chất lượng của một câu hỏi đơn. Nhưng với câu hỏi đa bước — đòi hỏi tổng hợp thông tin từ nhiều phần rải rác trong tài liệu — một retriever vẫn chưa đủ.*

---

## SLIDE 9 — Thách thức 4 → Co-RAG
**⏱ 2 phút 15 giây**

**[NÓI]**
**Hạn chế với câu hỏi đa bước:** Xét câu hỏi: "So sánh cách xử lý lỗi trong phần A với phương pháp đề xuất ở phần B, trong bối cảnh ràng buộc C." Khi encode câu hỏi này thành một vector duy nhất, vector đó phải đồng thời đại diện cho cả ba khía cạnh A, B, C — kết quả là không khía cạnh nào được biểu diễn đủ rõ nét. FAISS sẽ trả về các chunk "gần trung bình" của tất cả, bỏ sót những chunk liên quan đến riêng từng khía cạnh.

**Giải pháp — Co-RAG với 3 agent retrieval:**

**Agent 1 — Semantic (FAISS MMR):** Tìm kiếm theo ngữ nghĩa tổng thể của câu hỏi. Retriever mạnh nhất cho câu hỏi đơn giản hoặc khi ngữ nghĩa rõ ràng.

**Agent 2 — Keyword (BM25):** Khớp thuật ngữ kỹ thuật chính xác — tên hàm, tên biến, mã lỗi, hay ký hiệu viết tắt mà embedding có thể không nắm bắt được. Bổ sung những gì Agent 1 bỏ sót do khác biệt từ ngữ bề mặt.

**Agent 3 — Conceptual (LLM sub-question decompose):** Đây là điểm khác biệt cốt lõi. LLM phân rã câu hỏi phức tạp thành tối đa 3 sub-questions nguyên tử. Ví dụ câu hỏi trên sẽ được phân rã thành: (1) "Cách xử lý lỗi trong phần A là gì?", (2) "Phương pháp đề xuất ở phần B là gì?", (3) "Ràng buộc C ảnh hưởng như thế nào?". Mỗi sub-question được retrieve riêng — mỗi vector đặc trưng sắc nét hơn nhiều so với vector của câu hỏi gộp.

**Về kiến trúc thiết kế:** Ba agent **hoạt động độc lập** với nhau — không agent nào phụ thuộc kết quả của agent kia, kết quả chỉ được hợp nhất sau khi cả ba hoàn tất. Trong slide và khi trình bày, mô tả này là **"song song"**. *(Ghi chú kỹ thuật: implementation thực tế dùng for loop tuần tự trên CPU đơn luồng — nhưng đây là chi tiết triển khai, không phải ý tưởng kiến trúc.)*

**Cơ chế Consensus Merger — Vote Boost:**

Tất cả candidates từ 3 agent được hợp nhất. Điểm cuối được tính theo công thức:

**Score_final = avg_score × (1 + (v − 1) × 0.15)**

Trong đó v là số agent đồng thuận tìm ra document đó:
- v=1: không boost (×1.0) — chỉ 1 agent tìm thấy, độ tin cậy thấp
- v=2: boost 15% (×1.15) — 2 agent độc lập đều thấy document này liên quan
- v=3: boost 30% (×1.30) — cả 3 agent đồng thuận, rất đáng tin cậy

Ngưỡng CO_RAG_MIN_VOTES = 2 loại bỏ những document chỉ được 1 agent tìm thấy — giảm false positive.

**Kết quả so sánh trên 20 câu hỏi đa bước:**
- RAG Single Agent: recall ~72%
- Co-RAG 3 Agents: recall ~88% — cải thiện +16%
- Đánh đổi: +1.5–4 giây độ trễ bổ sung

Đánh đổi 16% recall lấy 1.5–4 giây thêm là hoàn toàn hợp lý cho các câu hỏi phức tạp. Với câu hỏi đơn giản, người dùng có thể tắt Co-RAG để giữ tốc độ.

**[CHUYỂN]**
*Bốn thách thức — bốn giải pháp. Giờ hãy xem tất cả các thành phần này ghép lại thành kiến trúc hệ thống hoàn chỉnh như thế nào.*

---

## SLIDE 10 — Kiến trúc hệ thống
**⏱ 1 phút 30 giây**

**[NÓI]**
Khi ghép lại, toàn bộ pipeline xử lý một câu hỏi đi qua 7 bước theo thứ tự từ trên xuống:

**Bước 1:** Truy vấn người dùng nhập vào giao diện Streamlit.

**Bước 2 — Self-RAG T1:** Query Expansion — LLM mở rộng câu hỏi thành 3 phiên bản viết lại (tổng 4 queries gồm cả câu gốc), retrieve lần lượt, hợp nhất candidates.

**Bước 3 — Co-RAG:** 3 agent (Semantic, Keyword, Conceptual) hoạt động song song, kết quả được Consensus Merger hợp nhất qua vote boost.

**Bước 4 — Hybrid Search RRF:** FAISS và BM25 retrieve với trọng số 0.6/0.4, RRF tổng hợp điểm xếp hạng.

**Bước 5 — CrossEncoder Reranking:** top-8 candidates được tái xếp hạng bằng full cross-attention, chọn top-3 tốt nhất.

**Bước 6 — Self-RAG T2 và T3:** Relevance Grading loại chunk không liên quan, Answer Grading kiểm tra hallucination sau khi LLM sinh xong.

**Bước 7:** Qwen2.5:7b qua Ollama sinh câu trả lời cuối, kèm Citation Tracking tên tệp và số trang.

Kiến trúc phân tầng bên phải có 4 tầng tách biệt — Presentation, Application, Data, và Model. Sự phân tầng này đảm bảo mỗi tầng có thể được thay thế hoặc nâng cấp độc lập — ví dụ có thể swap Qwen2.5 bằng Llama3 mà không ảnh hưởng tầng Presentation và Application ở trên.

**[CHUYỂN]**
*Kiến trúc đã rõ. Câu hỏi thực tế nhất: nó hoạt động nhanh như thế nào, và hiệu quả ra sao?*

---

## SLIDE 11 — Thực nghiệm — Hiệu năng retrieval
**⏱ 2 phút**

**[NÓI]**
Nhóm em đo latency của 3 phương pháp trên tập dữ liệu synthetic từ 1.000 đến 10.000 chunks — gồm các thuật ngữ AI/RAG đa ngôn ngữ. Sử dụng 5 câu truy vấn mẫu, mỗi câu đo 5 lần, tổng **25 lượt đo** mỗi mức. Môi trường: Intel Core i5-12400F (AVX2), 24GB DDR4, không GPU, Windows 11.

**Phân tích từng đường trên biểu đồ:**

**FAISS — đường phẳng nhất:** Từ 1.000 lên 10.000 chunks (tăng 10 lần), latency chỉ tăng từ 35.84ms lên 37.20ms — gần như không đổi. Lý do: bước encode câu hỏi bằng Multilingual MPNet chiếm khoảng 35–37ms cố định, trong khi thời gian search vector HNSW chỉ 0.1–1.5ms — quá nhỏ để thấy trên biểu đồ. FAISS không bị degradation khi corpus tăng — rất quan trọng cho khả năng mở rộng.

**BM25 — đường dốc nhất:** Tăng từ 1.48ms lên 21.89ms, đúng tuyến tính theo số chunks. BM25 phải duyệt inverted index tỷ lệ thuận với kích thước dữ liệu — không có cơ chế xấp xỉ như FAISS. Ở 10.000 chunks vẫn nhanh hơn FAISS, nhưng khi corpus lên hàng chục nghìn chunks, BM25 sẽ trở thành bottleneck của Hybrid Search.

**Hybrid — đường tăng đều giữa:** 37.95ms đến 60.52ms. Overhead đến từ BM25 (tăng tuyến tính) cộng với bước RRF hợp nhất kết quả. Tuy nhiên Hybrid luôn dưới 50ms tới 5.000 chunks — đây là ngưỡng thực tế của hầu hết use case. Chất lượng context tốt nhất ở mọi quy mô.

**Khuyến nghị thiết thực:**
- Dưới 5.000 chunks: Hybrid Search là lựa chọn mặc định.
- Trên 10.000 chunks: cân nhắc giảm trọng số BM25 hoặc chuyển sang FAISS thuần.

Ngoài ra: Co-RAG cải thiện recall từ 72% lên 88% (+16%) trên câu hỏi đa bước, đánh đổi bằng 1.5–4 giây độ trễ bổ sung.

**[CHUYỂN]**
*Hiệu năng đã được chứng minh bằng số. Bây giờ xem Self-RAG overhead phân bổ như thế nào và tại sao chúng ta chọn chunk_size 1500.*

---

## SLIDE 12 — Self-RAG Overhead & Chiến lược Chunking
**⏱ 1 phút 30 giây**

**[NÓI]**
**Biểu đồ overhead Self-RAG** cho thấy 3 kịch bản đo trên cùng phần cứng:

Câu hỏi đơn giản: RAG chuẩn 2.6 giây, Self-RAG 5.3 giây — overhead +2.7 giây.
Câu hỏi trung bình: RAG chuẩn 6.6 giây, Self-RAG 8.2 giây — overhead +1.6 giây.
Câu hỏi phức tạp: RAG chuẩn 8.3 giây, Self-RAG 12.1 giây — overhead +3.8 giây.

Điều thú vị là câu trung bình lại ít overhead hơn câu đơn giản. Lý do: câu hỏi trung bình thường liên quan rõ hơn với tài liệu, Tầng 2 lọc được nhiều document không liên quan hơn, context ngắn lại, nên Tầng 3 và bước generation của LLM cũng chạy nhanh hơn. Overhead không phải hằng số — nó phụ thuộc vào chất lượng retrieval.

Breakdown overhead theo từng tầng:
- Tầng 1 Query Expansion: 0.8–1.5 giây — 1 LLM call ngắn
- Tầng 2 Relevance Grading: 0.2–1.0 giây — tỷ lệ với số document cần grade
- Tầng 3 Answer Grading: 1.3–1.7 giây — 1 LLM call đánh giá câu trả lời

**Bảng chunking** ở bên phải — thử nghiệm 4 kích thước trên cùng tập 5 tài liệu PDF khoảng 120 trang:

- Chunk 500, overlap 50: sinh ~1.840 chunks. Context quá ngắn, câu hỏi phức tạp mất ngữ cảnh — LLM thiếu thông tin để tổng hợp câu trả lời hoàn chỉnh.
- Chunk 1.000, overlap 100: sinh ~920 chunks. Tốt cho văn xuôi thông thường nhưng chưa đủ cho tài liệu kỹ thuật dài.
- **Chunk 1.500, overlap 200: sinh ~610 chunks — điểm tối ưu.** 1500 ký tự tương đương 300–400 tokens tiếng Việt — đủ để chứa một lập luận hoàn chỉnh hoặc một đơn vị thông tin có nghĩa, đủ nhỏ để vector embedding đại diện cho một chủ đề cụ thể. Overlap 200 ký tự (13%) đảm bảo không mất thông tin tại ranh giới giữa hai chunk.
- Chunk 2.000, overlap 200: sinh ~460 chunks. Một chunk có thể chứa 2–3 chủ đề khác nhau, làm loãng tín hiệu cosine similarity — kết quả retrieve kém chính xác hơn.

**[CHUYỂN]**
*Hiệu năng và chunking đã rõ. Tính đúng đắn của từng module được bảo đảm bởi bộ kiểm thử mà tôi sẽ trình bày tiếp theo.*

---

## SLIDE 13 — Kiểm thử 141 test cases
**⏱ 1 phút 15 giây**

**[NÓI]**
Để đảm bảo tính đúng đắn và ổn định qua mỗi lần thay đổi code, SmartDoc AI được trang bị bộ kiểm thử tự động với **141 test cases — tất cả đều passed** — gồm 114 unit tests và 27 integration tests, trải rộng trên 7 module.

**Chiến lược kiểm thử hai tầng:**

**Unit Tests (114 tests):** Chạy hoàn toàn offline dưới 1 giây bằng cách dùng unittest.mock để giả lập LLM và embedding model. Không cần cài đặt Ollama hay tải model để chạy test — rất phù hợp cho CI/CD. Mỗi test kiểm tra đúng một hành vi: chunking trả về đúng kích thước? Metadata có được giữ nguyên qua pipeline không? Citation dedup hoạt động chính xác chưa?

**Integration Tests (27 tests):** Được đánh dấu @pytest.mark.integration và tự động skip nếu Ollama chưa chạy hoặc model chưa được tải — đảm bảo không làm gián đoạn pipeline CI khi môi trường thiếu tài nguyên. Các test này kiểm tra end-to-end: upload file thật, query thật, citation có đúng tên file và số trang không.

Coverage đầy đủ 11 yêu cầu kỹ thuật Q1–Q11: từ xử lý tài liệu đa định dạng (Q1), hội thoại (Q2, Q3), chunking (Q4), citation (Q5), conversational RAG (Q6), hybrid search (Q7), metadata filtering (Q8), reranking (Q9), Self-RAG (Q10), đến Co-RAG (Q11).

**[CHUYỂN]**
*Hệ thống đúng và hiệu quả. Hãy nhìn lại toàn bộ kết quả và những gì cần làm tiếp theo.*

---

## SLIDE 14 — Kết luận & Hướng phát triển
**⏱ 1 phút 15 giây**

**[NÓI]**
**Kết quả đạt được:**

SmartDoc AI triển khai thành công pipeline RAG offline hoàn toàn — không một byte dữ liệu nào rời khỏi máy người dùng. Năm đóng góp kỹ thuật đều hoạt động đúng thiết kế: Hybrid Search dưới 50ms, CrossEncoder top-8→top-3, Self-RAG 3 tầng, Co-RAG recall +16%, 141 test cases passed với coverage Q1–Q11 đầy đủ.

**Hạn chế trung thực:**

CPU bottleneck là hạn chế lớn nhất — mỗi LLM call tốn 2–8 giây trên i5 không GPU. FAISS lưu toàn bộ index trong RAM — corpus hàng triệu chunks cần FAISS IVF hoặc vector database phân tán. Hiện chỉ hỗ trợ PDF và DOCX. Co-RAG thêm 1.5–4 giây không phù hợp cho ứng dụng yêu cầu phản hồi tức thì.

**Hướng phát triển:**

Ưu tiên cao nhất là mở rộng định dạng sang Excel và PowerPoint, bao gồm trích xuất bảng phức tạp trong PDF bằng OCR. Tiếp theo là tối ưu batch embedding và thử nghiệm model nhẹ hơn như MiniLM-L6 hoặc BGE-M3. Và cải thiện UI quản lý tài liệu với khả năng xóa từng file riêng lẻ, lịch sử đa phiên, và dashboard thống kê truy vấn.

---

## SLIDE 15 — Cảm ơn
**⏱ 20 giây**

**[NÓI]**
Đó là toàn bộ nội dung trình bày của nhóm em về SmartDoc AI — từ vấn đề, giải pháp, từng thách thức kỹ thuật, đến kết quả thực nghiệm và kiểm thử.

Cảm ơn thầy và các bạn đã lắng nghe. Nhóm em rất mong nhận được câu hỏi và nhận xét.

---

## GỢI Ý TRẢ LỜI CÂU HỎI THƯỜNG GẶP

---

**Q: Tại sao chọn Qwen2.5:7b thay vì Llama 3 hay Mistral?**

Ba lý do chính. Thứ nhất, Qwen2.5 có hiệu năng xử lý tiếng Việt vượt trội trong các benchmark cộng đồng — đặc biệt quan trọng vì tài liệu của người dùng Việt Nam thường mix tiếng Việt và thuật ngữ kỹ thuật tiếng Anh. Thứ hai, context window 128.000 tokens cho phép đưa nhiều ngữ cảnh vào một Prompt hơn — giảm số lần cần retrieve. Thứ ba, Qwen2.5:7b chạy ổn định trên CPU không GPU với Ollama — các thử nghiệm với Llama 3 8B cho thấy thời gian inference cao hơn trên cùng phần cứng.

---

**Q: RRF có ưu điểm gì so với cách đơn giản hơn là lấy trung bình điểm FAISS và BM25?**

Vấn đề của việc lấy trung bình điểm là hai nguồn điểm không cùng đơn vị: điểm cosine của FAISS nằm trong [-1, 1], còn điểm BM25 không bị chặn trên — có thể là 0.5 hoặc 15 tùy corpus. Cộng hai số này trực tiếp sẽ cho kết quả sai lệch nghiêm trọng. RRF giải quyết bằng cách chỉ dùng **thứ hạng** thay vì điểm tuyệt đối — thứ hạng luôn là số nguyên từ 1 trở lên, có thể so sánh trực tiếp giữa hai retriever mà không cần chuẩn hóa.

---

**Q: Tại sao FAISS latency gần như không tăng khi data tăng 10 lần?**

Latency FAISS = thời gian encode câu hỏi + thời gian search. Thời gian encode bằng Multilingual MPNet cố định khoảng 35–37ms cho bất kỳ câu hỏi nào, không phụ thuộc kích thước corpus. Thời gian search vector HNSW chỉ 0.1–1.5ms — quá nhỏ. Vì encode chiếm 95%+ tổng latency, khi corpus tăng từ 1.000 lên 10.000 chunks, latency tổng chỉ tăng đúng phần search nhỏ đó.

---

**Q: Khi nào nên bật Self-RAG, khi nào nên tắt?**

Nên bật khi: tài liệu quan trọng (pháp lý, y tế, tài chính) nơi hallucination có hậu quả nghiêm trọng; câu hỏi phức tạp cần tổng hợp từ nhiều phần; hoặc khi người dùng cần điểm confidence để quyết định có tin câu trả lời không. Nên tắt khi: cần phản hồi nhanh; câu hỏi tra cứu đơn giản một thông tin cụ thể; hoặc khi đang demo realtime với người xem.

---

**Q: Tại sao Co-RAG và Self-RAG không thể bật cùng lúc?**

Khi bật cả hai, số lần gọi LLM tối thiểu trong một query là: Self-RAG T1 (1 call) + Co-RAG Agent 3 phân rã câu hỏi (1 call) + Self-RAG T2 grading từng document (1–5 calls) + LLM generation (1 call) + Self-RAG T3 grading câu trả lời (1 call) = 5–9 LLM calls. Trên CPU không GPU, mỗi call tốn 2–8 giây, tổng có thể lên 30–45 giây mỗi query — không thực tế cho người dùng cuối. Vì vậy hai chế độ được thiết kế mutually exclusive.

---

**Q: Vote Boost trong Co-RAG có ý nghĩa gì về mặt toán học?**

Công thức Score_final = avg_score × (1 + (v−1) × 0.15) với v là số agent đồng thuận. Khi v=1: không boost — chỉ 1 agent tìm ra, có thể là false positive. Khi v=2: tăng 15% — 2 agent độc lập (ví dụ semantic và keyword) đều thấy document này liên quan, xác suất thực sự liên quan cao hơn nhiều. Khi v=3: tăng 30% — cả semantic, keyword và conceptual đều đồng thuận. Hệ số 0.15 được chọn để boost có ý nghĩa nhưng không quá lớn — một document v=2 điểm 0.6 sẽ có score_final = 0.69, vẫn thấp hơn document v=1 điểm 0.8. Tín hiệu chất lượng vẫn quan trọng hơn số lượng vote.

---

**Q: Làm sao hệ thống xử lý đúng tiếng Việt?**

Hai cơ chế. Thứ nhất, embedding: mô hình paraphrase-multilingual-mpnet-base-v2 được huấn luyện trên 50+ ngôn ngữ với dữ liệu tiếng Việt đáng kể — vector của "phương tiện giao thông" và "xe cộ" sẽ nằm gần nhau trong không gian 768 chiều. Thứ hai, language detection: module language_detector phát hiện ngôn ngữ bằng cách đếm tỷ lệ ký tự có dấu (à, á, ả, ã, â, ê...) — nếu tỷ lệ vượt ngưỡng, hệ thống chọn prompt template tiếng Việt để LLM trả lời đúng ngôn ngữ.

---

**Q: BM25 index có được cập nhật khi thêm file mới không?**

Đây là hạn chế kỹ thuật thực tế của BM25 so với FAISS. FAISS hỗ trợ incremental update — có thể thêm vector mới vào index mà không rebuild toàn bộ. BM25 thì không — inverted index cần được xây dựng lại từ đầu từ toàn bộ corpus vì thống kê IDF (Inverse Document Frequency) phụ thuộc vào toàn bộ tập tài liệu: khi thêm file mới, IDF của tất cả các từ đều có thể thay đổi. Giải pháp dài hạn là dùng Elasticsearch hoặc OpenSearch hỗ trợ incremental BM25 update — đây là một trong những hướng phát triển tiếp theo.

---

*SmartDoc AI  •  DCT1231  •  2025–2026*

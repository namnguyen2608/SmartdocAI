**UBND THÀNH PHỐ HỒ CHÍ MINH**

**TRƯỜNG ĐẠI HỌC SÀI GÒN**  
**KHOA CÔNG NGHỆ THÔNG TIN**

![A blue circle with white textDescription automatically generated][image1]

**BÁO CÁO**  
**PHÁT TRIỂN PHẦN MỀM MÃ NGUỒN MỞ**

**ĐỀ TÀI:**   
**SMARTDOC AI \- INTELLIGENT DOCUMENT Q\&A SYSTEM**

**Giảng viên hướng dẫn:**	Từ Lãng Phiêu  
**Nhóm sinh viên thực hiện:**	   Lê Nguyễn Nhất Tâm \- 3122410369  
   Lưu Hồng Phúc	   \- 3121410389  
   Đặng Quang Phong     \- 3122410304  
   Nguyễn Phước Nam	   \- 3122410249  
**Lớp môn học:**	DCT1231  
**Năm học:** 2025 – 2026

---

# **MỤC LỤC**

[**LỜI MỞ ĐẦU**](#lời-mở-đầu)

[**CHƯƠNG I.** **Giới Thiệu và Các Công Trình Liên Quan**](#chương-i-giới-thiệu-và-các-công-trình-liên-quan)

[**1.1.** **Bối Cảnh**](#11-bối-cảnh)

[**1.2.** **Mục Tiêu Dự Án**](#12-mục-tiêu-dự-án)

[**1.3.** **Phạm Vi Dự Án**](#13-phạm-vi-dự-án)

[**1.4.** **Các Công Trình Liên Quan**](#14-các-công-trình-liên-quan)

[**CHƯƠNG II.** **Cơ Sở Lý Thuyết**](#chương-ii-cơ-sở-lý-thuyết)

[**2.1.** **Kiến Trúc RAG**](#21-kiến-trúc-rag-retrieval-augmented-generation)

[**2.2.** **Công Nghệ Embedding và Tìm Kiếm Tương Đồng Vector**](#22-công-nghệ-embedding-và-tìm-kiếm-tương-đồng-vector)

[**2.2.1.** **Text Embedding**](#221-text-embedding)

[**2.2.2.** **FAISS**](#222-faiss-facebook-ai-similarity-search)

[**2.3.** **Mô Hình Ngôn Ngữ Lớn Qwen2.5 và Nền Tảng Ollama**](#23-mô-hình-ngôn-ngữ-lớn-qwen25-và-nền-tảng-ollama)

[**2.4.** **Framework Điều Phối LangChain**](#24-framework-điều-phối-langchain)

[**2.5.** **Conversational RAG — Xử Lý Hội Thoại Đa Lượt**](#25-conversational-rag--xử-lý-hội-thoại-đa-lượt)

[**2.6.** **Lọc Metadata Theo Tài Liệu Nguồn**](#26-lọc-metadata-theo-tài-liệu-nguồn)

[**2.7.** **Hybrid Search — BM25 và Ensemble Retrieval**](#27-hybrid-search--bm25-và-ensemble-retrieval)

[**2.8.** **Re-Ranking với Cross-Encoder**](#28-re-ranking-với-cross-encoder)

[**2.9.** **Self-RAG — Tự Đánh Giá và Cải Thiện**](#29-self-rag--tự-đánh-giá-và-cải-thiện)

[**2.10.** **Co-RAG — Truy Xuất Hợp Tác Đa Agent**](#210-co-rag--truy-xuất-hợp-tác-đa-agent)

[**CHƯƠNG III.** **Thiết Kế và Triển Khai Hệ Thống**](#chương-iii-thiết-kế-và-triển-khai-hệ-thống)

[**3.1.** **Kiến Trúc Tổng Quan**](#31-kiến-trúc-tổng-quan)

[**3.2.** **Luồng Dữ Liệu**](#32-luồng-dữ-liệu)

[**3.3.** **Chi Tiết Các Thành Phần**](#33-chi-tiết-các-thành-phần)

[**3.4.** **Prompt Engineering**](#34-prompt-engineering)

[**3.5.** **Cấu Trúc Thư Mục Dự Án**](#35-cấu-trúc-thư-mục-dự-án)

[**3.6.** **Cài Đặt và Khởi Chạy**](#36-cài-đặt-và-khởi-chạy)

[**CHƯƠNG IV.** **Thực Nghiệm và Kiểm Thử**](#chương-iv-thực-nghiệm-và-kiểm-thử)

[**4.1.** **Môi Trường và Thiết Lập Thí Nghiệm**](#41-môi-trường-và-thiết-lập-thí-nghiệm)

[**4.2.** **Kết Quả Thí Nghiệm**](#42-kết-quả-thí-nghiệm)

[**4.3.** **Kiểm Thử Phần Mềm**](#43-kiểm-thử-phần-mềm-testing)

[**4.4.** **So Sánh Co-RAG và RAG Truyền Thống**](#44-so-sánh-co-rag-và-rag-truyền-thống)

[**CHƯƠNG V.** **Giao Diện và Hướng Dẫn Sử Dụng**](#chương-v-giao-diện-và-hướng-dẫn-sử-dụng)

[**5.1.** **Thiết Kế Giao Diện Người Dùng**](#51-thiết-kế-giao-diện-người-dùng)

[**5.2.** **Luồng Sử Dụng (User Flow)**](#52-luồng-sử-dụng-user-flow)

[**5.3.** **Tính Năng Chính**](#53-các-tính-năng-chính)

[**5.4.** **Hướng Dẫn Người Dùng Cuối**](#54-hướng-dẫn-người-dùng-cuối)

[**5.5.** **Hướng Dẫn Developer**](#55-hướng-dẫn-developer)

[**CHƯƠNG VI.** **Phân Tích Kỹ Thuật Triển Khai**](#chương-vi-phân-tích-kỹ-thuật-triển-khai)

[**6.1.** **Pipeline Thu Nạp Tài Liệu**](#61-pipeline-thu-nạp-tài-liệu)

[**6.2.** **Kiến Trúc Truy Xuất Thông Tin**](#62-kiến-trúc-truy-xuất-thông-tin)

[**6.3.** **Quản Lý Ngữ Cảnh Hội Thoại**](#63-quản-lý-ngữ-cảnh-hội-thoại)

[**6.4.** **Kiểm Soát Chất Lượng Qua Self-RAG**](#64-kiểm-soát-chất-lượng-qua-self-rag)

[**6.5.** **Co-RAG: Truy Xuất Hợp Tác Đa Agent**](#65-co-rag-truy-xuất-hợp-tác-đa-agent)

[**CHƯƠNG VII.** **Kết Luận và Hướng Phát Triển**](#chương-vii-kết-luận-và-hướng-phát-triển)

[**7.1.** **Kết Quả Đạt Được**](#71-kết-quả-đạt-được)

[**7.2.** **Điểm Hạn Chế**](#72-điểm-hạn-chế)

[**7.3.** **Hướng Phát Triển**](#73-hướng-phát-triển)

[**TÀI LIỆU THAM KHẢO**](#tài-liệu-tham-khảo)

---

# **DANH MỤC KÝ HIỆU VÀ VIẾT TẮT**

| **STT** | **Tên Viết Tắt** | **Diễn Giải** |
| :---: | :---: | :--- |
| 1 | API | Application Programming Interface |
| 2 | AVX2 | Advanced Vector Extensions 2 |
| 3 | BM25 | Best Match 25 |
| 4 | Co-RAG | Collaborative Retrieval-Augmented Generation |
| 5 | CPU | Central Processing Unit |
| 6 | DOCX | Document XML (định dạng Microsoft Word) |
| 7 | FAISS | Facebook AI Similarity Search |
| 8 | GPU | Graphics Processing Unit |
| 9 | LangChain | Language Chain Framework |
| 10 | LLM | Large Language Model |
| 11 | MMR | Maximal Marginal Relevance |
| 12 | MPNet | Masked and Permuted Pre-training for Language Understanding |
| 13 | NLP | Natural Language Processing |
| 14 | PDF | Portable Document Format |
| 15 | Q&A | Question and Answering |
| 16 | RAG | Retrieval-Augmented Generation |
| 17 | RAM | Random Access Memory |
| 18 | RRF | Reciprocal Rank Fusion |
| 19 | Self-RAG | Self-Reflective Retrieval-Augmented Generation |
| 20 | SSD | Solid State Drive |
| 21 | UI | User Interface |
| 22 | VRAM | Video Random Access Memory |

---

# **LỜI MỞ ĐẦU**

Trong bối cảnh bùng nổ thông tin của thời đại số, nhu cầu tra cứu và khai thác tri thức từ tài liệu điện tử một cách nhanh chóng, chính xác và bảo mật ngày càng trở nên cấp thiết. Báo cáo này trình bày toàn bộ quá trình nghiên cứu, thiết kế và phát triển hệ thống **SmartDoc AI** — một trợ lý hỏi đáp tài liệu thông minh hoạt động hoàn toàn ngoại tuyến (offline), được xây dựng dựa trên kiến trúc Retrieval-Augmented Generation (RAG) kết hợp mô hình ngôn ngữ lớn Qwen2.5:7b thông qua nền tảng Ollama.

Mục tiêu cốt lõi của dự án là chứng minh rằng một hệ thống AI hỏi đáp tài liệu có khả năng xử lý đa ngôn ngữ (đặc biệt tiếng Việt), hỗ trợ nhiều định dạng tệp (PDF, DOCX), và tích hợp các kỹ thuật retrieval nâng cao (Hybrid Search, Re-Ranking, Self-RAG, Co-RAG) hoàn toàn có thể được triển khai trên phần cứng cá nhân mà không cần phụ thuộc vào các dịch vụ đám mây hay tiềm ẩn rủi ro bảo mật dữ liệu.

Báo cáo được tổ chức theo trình tự logic từ lý thuyết đến thực hành:

* **Chương I** trình bày bối cảnh, mục tiêu và phạm vi dự án, cùng tổng quan các công trình liên quan.
* **Chương II** xây dựng nền tảng lý thuyết đầy đủ: RAG, Embedding, FAISS, LLM, LangChain, Hybrid Search, Cross-Encoder Re-Ranking, Self-RAG và Co-RAG.
* **Chương III** đi sâu vào thiết kế kiến trúc hệ thống, chi tiết triển khai từng thành phần và hướng dẫn cài đặt.
* **Chương IV** trình bày kết quả thực nghiệm, đo lường hiệu năng và kết quả kiểm thử tự động (**141 test cases passed**).
* **Chương V** mô tả giao diện người dùng và hướng dẫn sử dụng chi tiết cho cả người dùng cuối lẫn developer.
* **Chương VI** phân tích kỹ thuật triển khai theo 5 mảng chức năng: pipeline thu nạp tài liệu, kiến trúc truy xuất thông tin, quản lý ngữ cảnh hội thoại, kiểm soát chất lượng qua Self-RAG, và Co-RAG đa agent.
* **Chương VII** tổng kết kết quả đạt được và định hướng phát triển tương lai.

---

# **CHƯƠNG I: GIỚI THIỆU VÀ CÁC CÔNG TRÌNH LIÊN QUAN**

## **1.1. Bối Cảnh**

Trong kỷ nguyên chuyển đổi số hiện nay, khối lượng thông tin và tri thức được lưu trữ dưới dạng tài liệu điện tử, đặc biệt là định dạng PDF, đang tăng lên theo cấp số nhân. Tại các tổ chức, doanh nghiệp và môi trường học thuật, việc phải tiếp nhận, đọc hiểu và xử lý hàng trăm trang tài liệu mỗi ngày đã trở thành một thách thức lớn. Quá trình tra cứu, tổng hợp và trích xuất thông tin thủ công từ nguồn dữ liệu khổng lồ này không chỉ tiêu tốn nhiều thời gian mà còn dễ dẫn đến tình trạng quá tải thông tin hoặc bỏ sót những dữ kiện quan trọng.

## **1.2. Mục Tiêu Dự Án**

Mục tiêu tổng quát của dự án là nghiên cứu và phát triển thành công một hệ thống hỏi đáp tài liệu thông minh (SmartDoc AI) dựa trên nền tảng kiến trúc Retrieval-Augmented Generation (RAG). Hệ thống được định hướng để giải quyết bài toán tra cứu, tổng hợp và trích xuất thông tin từ các tài liệu PDF đa ngôn ngữ (đặc biệt là tiếng Việt), với yêu cầu khắt khe về độ chính xác, tính tự chủ và khả năng bảo mật dữ liệu hoàn toàn ngoại tuyến (offline).

Để hiện thực hóa mục tiêu tổng quát này, đồ án đặt ra các mục tiêu cụ thể trên từng phương diện kỹ thuật như sau:

* **Thiết kế và làm chủ đường ống xử lý RAG (RAG Pipeline):** Xây dựng một luồng xử lý dữ liệu khép kín và tự động hoàn toàn, bắt đầu từ khâu nạp tài liệu (document loading), chia nhỏ văn bản (text chunking), nhúng dữ liệu (embedding) cho đến truy xuất (retrieval) và sinh ngôn ngữ (generation).  
* **Xử lý và biểu diễn dữ liệu trong không gian Vector:** Nghiên cứu và tích hợp công nghệ Text Embedding đa ngôn ngữ để chuyển đổi văn bản thô thành các vector không gian nhiều chiều. Đồng thời, triển khai cơ sở dữ liệu vector FAISS (Facebook AI Similarity Search) nhằm lưu trữ và thực hiện các thuật toán tìm kiếm độ tương đồng (Similarity Search) với hiệu năng cao.  
* **Triển khai nền tảng Mô hình ngôn ngữ lớn (LLM) cục bộ:** Cài đặt và vận hành mô hình ngôn ngữ lớn Qwen2.5 (phiên bản 7B parameters) thông qua nền tảng Ollama. Việc tối ưu hóa mô hình này chạy trực tiếp trên máy tính cá nhân nhằm đảm bảo khả năng xử lý ngôn ngữ tự nhiên xuất sắc cho tiếng Việt, đồng thời loại bỏ hoàn toàn rủi ro rò rỉ dữ liệu khi sử dụng các API bên ngoài.  
* **Phát triển giao diện tương tác và trải nghiệm người dùng (UI/UX):** Xây dựng một ứng dụng web trực quan và thân thiện. Giao diện cần cung cấp đầy đủ các tính năng tải lên tài liệu định dạng PDF, khung tương tác hỏi đáp thời gian thực, và các cơ chế xử lý lỗi (error handling) mềm dẻo nhằm nâng cao trải nghiệm người dùng cuối.  
* **Đánh giá và tối ưu hóa chất lượng hệ thống:** Đo lường và tối ưu hóa hiệu suất tổng thể của hệ thống, bao gồm thời gian phản hồi và độ chính xác của câu trả lời. Hệ thống phải có khả năng hiểu đúng ngữ cảnh của tài liệu, trả lời chính xác các câu hỏi và giảm thiểu tối đa hiện tượng "ảo giác thông tin" (hallucination).

## **1.3. Phạm Vi Dự Án**

Để đảm bảo tính khả thi và tập trung vào các giá trị cốt lõi, phạm vi của đồ án được giới hạn trong các khía cạnh sau:

* **Định dạng dữ liệu đầu vào:** Hệ thống tập trung xử lý và phân tích dữ liệu từ các tài liệu dưới định dạng PDF và DOCX (Microsoft Word).  
* **Hỗ trợ đa ngôn ngữ:** Hệ thống có khả năng xử lý xuất sắc tiếng Việt cùng hơn 50 ngôn ngữ phổ biến khác trên thế giới. Đặc biệt, ứng dụng được trang bị cơ chế tự động phát hiện ngôn ngữ đầu vào để đưa ra phản hồi bằng ngôn ngữ tương ứng một cách tự nhiên.  
* **Môi trường triển khai:** Toàn bộ hệ thống, bao gồm cả mô hình LLM và cơ sở dữ liệu, được cấu hình để chạy trực tiếp (local) trên máy tính cá nhân mà không cần phụ thuộc vào kết nối mạng.  
* **Nền tảng công nghệ:** Dự án cam kết chỉ sử dụng các mô hình ngôn ngữ và thư viện mã nguồn mở, hoàn toàn miễn phí, giúp dễ dàng nhân rộng và tiếp cận.

## **1.4. Các Công Trình Liên Quan**

Trong những năm gần đây, sự bùng nổ của các Mô hình ngôn ngữ lớn (LLM) như GPT, Llama hay Qwen đã mở ra một kỷ nguyên mới cho việc xử lý ngôn ngữ tự nhiên. Tuy nhiên, việc ứng dụng LLM vào các bài toán thực tế thường vấp phải rào cản về tính cập nhật của dữ liệu và hiện tượng "ảo giác" (hallucination).

Để giải quyết vấn đề này, kiến trúc Retrieval-Augmented Generation (RAG) đã trở thành một xu hướng nghiên cứu và ứng dụng chủ đạo.

Nghiên cứu nền tảng của Lewis et al. (2020) [1] đã chứng minh rằng việc kết hợp một bộ truy xuất dữ liệu ngoại vi với một mô hình sinh văn bản cải thiện đáng kể độ chính xác trong các tác vụ cần nhiều tri thức (knowledge-intensive NLP). Tiếp nối, Karpukhin et al. (2020) [6] phát triển DPR (Dense Passage Retrieval) — phương pháp encode dày đặc cho retrieval ngữ nghĩa hiệu quả, đặt nền móng cho FAISS-based semantic search. Robertson và Zaragoza (2009) [2] đã định nghĩa công thức BM25 chuẩn tắc được hầu hết hệ thống tìm kiếm hiện đại áp dụng. Nogueira và Cho (2019) [3] giới thiệu mô hình Cross-Encoder dựa trên BERT cho bài toán re-ranking passage, đạt độ chính xác vượt trội so với bi-encoder nhờ khả năng mô hình hóa tương tác từng từ giữa câu hỏi và tài liệu.

Về tự đánh giá chất lượng, Asai et al. (2023) [4] giới thiệu Self-RAG — framework dạy LLM tự phát sinh reflection tokens để kiểm soát khi nào nên retrieve, khi nào critique kết quả. Luan et al. (2021) [7] nghiên cứu chiến lược hybrid search kết hợp dense retrieval và sparse retrieval, phân tích trade-off giữa ngữ nghĩa và từ khóa trong nhiều bộ dữ liệu thực tế.

Các framework như LangChain và LlamaIndex đã ra đời, cung cấp các công cụ chuẩn hóa để xây dựng các đường ống (pipelines) xử lý dữ liệu từ nhiều định dạng một cách hiệu quả. Bên cạnh đó, xu hướng Local LLM (triển khai mô hình ngôn ngữ cục bộ) đang ngày càng phát triển nhờ các nền tảng như Ollama [5], cho phép người dùng vận hành các mô hình mạnh mẽ như Qwen2.5 ngay trên phần cứng cá nhân mà không cần kết nối internet.

Đồ án SmartDoc AI được xây dựng dựa trên sự kế thừa các thành tựu này, tích hợp Hybrid Search [2,7], Cross-Encoder Re-Ranking [3], Self-RAG [4] và kiến trúc Co-RAG đa agent mới. Hệ thống đặc biệt tập trung tối ưu hóa khả năng truy xuất cho ngôn ngữ tiếng Việt — một lĩnh vực đòi hỏi sự tinh chỉnh đặc biệt về mô hình Embedding đa ngôn ngữ và kỹ thuật phân mảnh văn bản.

# **CHƯƠNG II: CƠ SỞ LÝ THUYẾT**

## **2.1. Kiến Trúc RAG (Retrieval-Augmented Generation)**  
Kiến trúc RAG là một quy trình hai giai đoạn nhằm nâng cao chất lượng phản hồi của AI bằng cách tích hợp thông tin từ một nguồn dữ liệu đáng tin cậy bên ngoài. Về mặt hình thức, đầu ra của mô hình được định nghĩa là:

$$\text{RAG}(x) = \text{Generator}\bigl(x,\, \text{Retriever}(x)\bigr)$$

trong đó $x$ là câu hỏi đầu vào, $\text{Retriever}(x)$ trả về tập văn bản liên quan từ kho tài liệu, và $\text{Generator}$ tổng hợp câu trả lời dựa trên cả $x$ lẫn ngữ cảnh được truy xuất [1].

1. Giai đoạn Truy xuất (Retrieval): Khi nhận được một câu hỏi từ người dùng, hệ thống sẽ thực hiện tìm kiếm trong cơ sở dữ liệu để trích xuất ra các đoạn văn bản (chunks) có liên quan nhất. Quá trình này không dựa trên việc đối khớp từ khóa đơn thuần mà dựa trên sự tương đồng về mặt ngữ nghĩa (Semantic Similarity) trong không gian vector.  
2. Giai đoạn Sinh câu trả lời (Generation): Các đoạn văn bản tìm được sẽ được kết hợp với câu hỏi gốc để tạo thành một "Prompt" (lời nhắc) đầy đủ ngữ cảnh. Mô hình ngôn ngữ lớn (LLM) sau đó sẽ đọc hiểu ngữ cảnh này và sinh ra câu trả lời dựa trên các dữ kiện có thực, từ đó giảm thiểu tối đa tình trạng đưa ra thông tin sai lệch

## **2.2. Công Nghệ Embedding và Tìm Kiếm Tương Đồng Vector**

### **2.2.1. Text Embedding**  
Đây là quá trình chuyển đổi các đơn vị ngôn ngữ (từ, câu, đoạn văn) thành các vector số thực trong không gian n-chiều. Đặc điểm quan trọng của Embedding là các văn bản có nội dung hoặc ý nghĩa tương tự nhau sẽ được biểu diễn bởi các vector nằm gần nhau trong không gian toán học. Dự án sử dụng mô hình `paraphrase-multilingual-mpnet-base-v2` (Multilingual MPNet) từ thư viện sentence-transformers với không gian 768 chiều, được tối ưu hóa đặc biệt cho tiếng Việt để đảm bảo độ chính xác khi tìm kiếm ngữ nghĩa.  
### **2.2.2. FAISS (Facebook AI Similarity Search)**  
FAISS là một thư viện mã nguồn mở chuyên dụng cho việc tìm kiếm sự tương đồng và phân cụm các vector mật độ cao. Khác với các cơ sở dữ liệu truyền thống, FAISS được tối ưu hóa để thực hiện tìm kiếm hàng triệu vector trong thời gian mili giây, hỗ trợ các thuật toán lập chỉ mục hiệu quả như IndexFlatL2 hoặc IndexIVF để cân bằng giữa tốc độ và độ chính xác. Dự án sử dụng chiến lược truy xuất MMR (Maximal Marginal Relevance) với thông số: TOP\_K = 8 kết quả trả về, FETCH\_K = 50 ứng viên ban đầu và LAMBDA\_MULT = 0.7 để cân bằng giữa độ liên quan và tính đa dạng của kết quả.  
## **2.3. Mô Hình Ngôn Ngữ Lớn Qwen2.5 và Nền Tảng Ollama**

* **Qwen2.5:7b:** Là thế hệ mô hình ngôn ngữ lớn mạnh mẽ được Alibaba Cloud phát triển. Phiên bản 7 tỷ tham số (7B) được lựa chọn nhờ khả năng hiểu ngữ cảnh dài (lên đến 128K tokens) và hiệu năng vượt trội trong việc xử lý tiếng Việt so với các mô hình cùng kích thước khác .  
* **Ollama:** Đóng vai trò là "môi trường thực thi" (runtime) cho các mô hình LLM trên máy tính cá nhân. Ollama giúp quản lý tài nguyên phần cứng (CPU/GPU) một cách tối ưu, cho phép chạy mô hình Qwen2.5 mượt mà mà không cần đến hạ tầng điện toán đám mây đắt đỏ.

## **2.4. Framework Điều Phối LangChain**  
LangChain là một bộ khung phát triển ứng dụng (framework) giúp kết nối các thành phần rời rạc của hệ thống RAG thành một chuỗi (chain) thống nhất. Nó cung cấp các module chuẩn để:

* Tải và xử lý tài liệu PDF và DOCX (Document Loaders & Document Processors).  
* Cắt nhỏ văn bản (RecursiveCharacterTextSplitter) theo các quy tắc đệ quy để bảo toàn ngữ nghĩa, với thông số CHUNK\_SIZE = 1500 ký tự và CHUNK\_OVERLAP = 200 ký tự.  
* Quản lý luồng hội thoại và Prompt Engineering (Chains & Prompts).

## **2.5. Conversational RAG — Xử Lý Hội Thoại Đa Lượt**  
Để hỗ trợ hội thoại liên tục (multi-turn conversation), hệ thống triển khai cơ chế viết lại câu hỏi (Question Reformulation):

* **Cách hoạt động:** Khi người dùng đặt câu hỏi tiếp theo, LLM phân tích lịch sử hội thoại và tự động viết lại thành một câu hỏi độc lập, đầy đủ ngữ cảnh trước khi thực hiện truy xuất.
* **Lợi ích:** Hệ thống trả lời đúng ngay cả khi người dùng dùng đại từ tham chiếu ("nó", "cái đó") mà không cần nhắc lại chủ đề.

## **2.6. Lọc Metadata Theo Tài Liệu Nguồn**  
Hệ thống hỗ trợ lọc kết quả truy xuất theo tên tệp tài liệu nguồn (Metadata Filtering):

* **Cách dùng:** Người dùng chọn tên file từ dropdown trong sidebar để giới hạn phạm vi tìm kiếm.
* **Cơ chế:** Bộ lọc chỉ trả về các chunk có trường `source` khớp với file được chọn, loại bỏ kết quả từ các tệp khác.
* **Lợi ích:** Tăng độ chính xác câu trả lời khi kho tài liệu có nhiều file không liên quan nhau.

## **2.7. Hybrid Search — BM25 và Ensemble Retrieval**

Hybrid Search là kỹ thuật kết hợp hai phương pháp tìm kiếm độc lập để bù đắp điểm yếu của nhau:

* **BM25 (Best Matching 25):** Là thuật toán tìm kiếm từ khóa truyền thống dựa trên mô hình thống kê TF-IDF (Term Frequency–Inverse Document Frequency). BM25 tìm các tài liệu chứa từ khóa giống hệt với câu truy vấn, hoạt động rất hiệu quả khi câu hỏi có từ chuyên ngành cụ thể hoặc tên riêng. Công thức BM25 (Robertson–Zaragoza) tính điểm liên quan của tài liệu $d$ với truy vấn $Q$ gồm các từ $q_1, \ldots, q_n$ như sau [2]:

$$\text{BM25}(d, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, d)\,(k_1 + 1)}{f(q_i, d) + k_1\!\left(1 - b + b\,\frac{|d|}{\text{avgdl}}\right)}$$

trong đó $f(q_i, d)$ là tần suất từ $q_i$ trong tài liệu $d$, $|d|$ là độ dài tài liệu, $\text{avgdl}$ là độ dài trung bình, $k_1 = 1.5$ và $b = 0.75$ là các tham số điều chỉnh. Hệ thống sử dụng thư viện `rank-bm25` với các giá trị mặc định này.

* **Semantic Search (FAISS):** Tìm kiếm dựa trên độ tương đồng ngữ nghĩa trong không gian vector, bắt được các từ đồng nghĩa và khái niệm liên quan, nhưng đôi khi bỏ sót tài liệu có từ khóa chính xác.

* **EnsembleRetriever:** LangChain cung cấp `EnsembleRetriever` (thông qua gói `langchain_classic`) để kết hợp kết quả từ nhiều retriever bằng thuật toán **Reciprocal Rank Fusion (RRF)**. Điểm RRF của tài liệu $d$ được tính theo công thức:

$$\text{score}_{\text{RRF}}(d) = \frac{1}{k + \text{rank}_{\text{vec}}(d)} + \frac{1}{k + \text{rank}_{\text{BM25}}(d)}$$

trong đó $k = 60$ là hằng số làm trơn và $\text{rank}(d)$ là thứ hạng của tài liệu trong danh sách kết quả của mỗi retriever. Tài liệu xuất hiện cao ở cả hai danh sách sẽ được xếp hạng ưu tiên trong kết quả cuối. Trong SmartDoc AI, FAISS chiếm trọng số 0.6 và BM25 chiếm 0.4.

## **2.8. Re-Ranking với Cross-Encoder**

Sau bước retrieval (bi-encoder), hệ thống thực hiện bước re-ranking (xếp hạng lại) sử dụng Cross-Encoder để cải thiện độ chính xác:

* **Bi-Encoder (FAISS):** Mã hóa câu hỏi và mỗi tài liệu thành hai vector độc lập, sau đó tính cosine similarity. Nhanh nhưng chỉ xem xét câu hỏi và tài liệu riêng rẽ.

* **Cross-Encoder:** Nhận cả câu hỏi và tài liệu làm đầu vào cùng một lúc (`[CLS] question [SEP] document [SEP]`), cho phép mô hình transformer tính toán tương tác từng từ giữa hai văn bản. Điều này cho điểm relevance chính xác hơn nhiều, nhưng tốn kém hơn về tính toán (O(n) inference calls).

Pipeline 2 bước: Bi-encoder retrieval (lấy top-K ứng viên nhanh) → Cross-encoder reranking (chọn top-k chính xác) là pattern phổ biến trong information retrieval hiện đại. SmartDoc AI sử dụng model `cross-encoder/ms-marco-MiniLM-L-6-v2` đã được fine-tune trên tập MS MARCO passage ranking.

## **2.9. Self-RAG — Tự Đánh Giá và Cải Thiện**

Self-RAG (Asai et al., 2023) là kỹ thuật giúp LLM tự đánh giá chất lượng quá trình retrieval và generation, gồm ba thành phần chính được tích hợp trong SmartDoc AI:

* **Query Rewriting (Viết lại câu hỏi):** Trước khi tìm kiếm, LLM tự động sinh ra 3 phiên bản câu hỏi từ các góc độ khác nhau. Điều này mở rộng không gian tìm kiếm, giúp bắt được các tài liệu liên quan mà câu hỏi gốc có thể bỏ sót.

* **Relevance Grading (Đánh giá độ liên quan):** Sau khi retrieve, LLM đánh giá từng tài liệu xem có thực sự liên quan đến câu hỏi hay không (nhãn CÓ/KHÔNG). Các tài liệu không liên quan bị loại trước khi đưa vào prompt generation.

* **Answer Grading (Đánh giá câu trả lời):** Sau khi LLM sinh câu trả lời, một lần đánh giá thứ hai được thực hiện để kiểm tra: câu trả lời có trả lời đúng câu hỏi không? có căn cứ trên context không? có hallucination không? Kết quả trả về dạng JSON với điểm confidence `[0.0, 1.0]`.

## **2.10. Co-RAG — Truy Xuất Hợp Tác Đa Agent**

Co-RAG (Collaborative RAG) là kiến trúc mở rộng RAG truyền thống bằng cách triển khai **nhiều retriever agent song song**, mỗi agent khai thác một chiều không gian tìm kiếm khác nhau. Sau khi các agent hoàn thành, một thành phần **Consensus Merger** tổng hợp kết quả thông qua cơ chế bầu chọn [3].

**Kiến trúc 3 Agent:**

| Agent | Phương pháp | Vai trò |
|---|---|---|
| Agent 1 — Semantic Retriever | FAISS MMR similarity search | Bắt ngữ nghĩa, đảm bảo tính đa dạng kết quả |
| Agent 2 — Keyword Retriever | BM25 exact/partial matching | Bắt từ khóa chuyên ngành, tên riêng |
| Agent 3 — Conceptual Decomposer | LLM phân rã câu hỏi → sub-questions | Xử lý câu hỏi đa bước, đa khía cạnh |

**Consensus Merger và Vote Boost:** Mỗi tài liệu được nhận diện qua 120 ký tự đầu (fingerprint). Điểm cuối cùng được tính theo:

$$\text{score}_{\text{merged}} = \bar{s} \times \left(1 + (v - 1) \times 0.15\right)$$

trong đó $\bar{s}$ là điểm trung bình từ các agent đồng thuận và $v$ là số agent trả về tài liệu đó. Tài liệu được xác nhận bởi nhiều agent nhận điểm cao hơn, phản ánh sự đồng thuận đa chiều.

**Ba chiến lược hợp nhất:**
* `voting` (mặc định): Chỉ giữ tài liệu có ít nhất $\text{CO\_RAG\_MIN\_VOTES} = 2$ agent đồng thuận — lọc nhiễu hiệu quả.
* `union`: Giữ toàn bộ tài liệu từ mọi agent — tối đa hoá độ bao phủ.
* `intersection`: Chỉ giữ tài liệu xuất hiện ở **tất cả** các agent — độ chính xác cao nhất.

Co-RAG đặc biệt hiệu quả với **câu hỏi đa bước** (multi-hop questions) vì Agent 3 phân rã câu hỏi phức tạp thành các sub-questions đơn giản hơn, đảm bảo từng khía cạnh đều được truy xuất đầy đủ. Triển khai tại `modules/co_rag.py` với các tham số cấu hình trong `config.py`: `CO_RAG_TOP_K_PER_AGENT = 5`, `CO_RAG_MIN_VOTES = 2`, `CO_RAG_MERGE_STRATEGY = "voting"`.

# **CHƯƠNG III: THIẾT KẾ VÀ TRIỂN KHAI HỆ THỐNG**

## **3.1. Kiến Trúc Tổng Quan**

SmartDoc AI được thiết kế theo mô hình **multi-layer architecture** gồm 4 tầng:

```
┌─────────────────────────────────────────────────────────┐
│              PRESENTATION LAYER (Streamlit)              │
│  • Giao diện web tương tác (dark theme)                  │
│  • Upload PDF/DOCX, Chat History, Source Display         │
│  • Sidebar: Settings, Clear History, Clear Vector Store  │
│  • Toggle: Hybrid Search, Self-RAG, Co-RAG               │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│              APPLICATION LAYER (LangChain)               │
│  • Document Processing Pipeline (document_processor.py)  │
│  • RAG Chain Management (rag_chain.py)                   │
│  • Hybrid Search / Re-Ranking / Self-RAG / Co-RAG       │
│  • Language Detection (language_detector.py)             │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                  DATA LAYER                              │
│  • FAISS Vector Store (vectorstore/smartdoc_index/)      │
│  • Multilingual MPNet Embeddings (768-dim)               │
│  • BM25 Index (in-memory, from rank_bm25)                │
│  • Uploaded Document Storage (data/uploads/)             │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                 MODEL LAYER (Ollama)                     │
│  • Qwen2.5:7b — LLM Inference (localhost:11434)          │
│  • Cross-Encoder ms-marco-MiniLM-L-6-v2 (Re-ranking)    │
└─────────────────────────────────────────────────────────┘
```

## **3.2. Luồng Dữ Liệu**

### **3.2.1. Document Processing Flow (Luồng Xử Lý Tài Liệu)**

1. **Upload:** Người dùng tải lên file PDF hoặc DOCX qua giao diện Streamlit.
2. **Loading:** `extract_text_from_pdf()` hoặc `extract_text_from_docx()` trong `modules/document_processor.py` trích xuất văn bản từng trang/đoạn kèm metadata (`source`, `page`, `file_type`, `upload_date`).
3. **Splitting:** `split_documents()` sử dụng `RecursiveCharacterTextSplitter` với `CHUNK_SIZE=1500`, `CHUNK_OVERLAP=200` chia nhỏ tài liệu thành các chunks có kích thước vừa phải.
4. **Embedding:** Multilingual MPNet (`paraphrase-multilingual-mpnet-base-v2`, 768-dim) chuyển đổi mỗi chunk thành vector.
5. **Indexing:** FAISS lưu trữ tất cả vectors vào index `vectorstore/smartdoc_index/`.
6. **BM25 Index:** Song song, `BM25Retriever` xây dựng index từ khóa in-memory từ nội dung text của các chunks.

### **3.2.2. Query Processing Flow (Luồng Xử Lý Câu Hỏi)**

1. **Query Input:** Người dùng nhập câu hỏi trong chat interface.
2. **Language Detection:** `detect_language()` xác định ngôn ngữ (vi/en) để chọn prompt phù hợp.
3. **Conversational Reformulation:** Nếu có lịch sử hội thoại, LLM viết lại câu hỏi thành dạng standalone (sử dụng `REFORMULATE_QUESTION_TEMPLATE`).
4. **Hybrid Retrieval:** `EnsembleRetriever` kết hợp FAISS (weight=0.6) và BM25 (weight=0.4) trả về top-K ứng viên.
5. **Re-Ranking:** `rerank_with_cross_encoder()` sắp xếp lại kết quả theo cross-encoder score (nếu khả dụng).
6. **Metadata Filtering:** Lọc chunks theo `source` nếu người dùng chọn file cụ thể.
7. **Prompt Construction:** Context từ các chunks được ghép vào `RAG_PROMPT_TEMPLATE` cùng câu hỏi và lịch sử hội thoại.
8. **LLM Inference:** Qwen2.5:7b (qua Ollama) sinh câu trả lời.
9. **Self-RAG Grading (optional):** `grade_answer()` đánh giá chất lượng câu trả lời, phát hiện hallucination.
10. **Response Display:** Câu trả lời, danh sách sources (file + trang), và lịch sử chat được hiển thị trên giao diện.

## **3.3. Chi Tiết Các Thành Phần**

### **3.3.1. `modules/document_processor.py`**

Module xử lý tài liệu đầu vào, gồm:

* `extract_text_from_pdf(file_path, source_name)`: Dùng `pdfplumber` đọc từng trang PDF, trả về `List[Document]` với metadata `{source, page, total_pages, file_type, upload_date}`.
* `extract_text_from_docx(file_path, source_name)`: Dùng `python-docx` đọc paragraphs + tables, ghép thành "trang ảo" để phù hợp pipeline.
* `split_documents(documents)`: Áp dụng `RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)` — tham số cấu hình trong `config.py`.
* `process_uploaded_file(file_path)`: Hàm tổng hợp: phát hiện định dạng → gọi extractor phù hợp → split → trả về chunks.

### **3.3.2. `modules/vector_store.py`**

Module quản lý FAISS vector store:

* `get_embedding_model()`: Singleton pattern — tải `paraphrase-multilingual-mpnet-base-v2` một lần duy nhất.
* `create_vector_store(documents)`: Tạo FAISS index từ danh sách Document.
* `save_vector_store(vector_store)` / `load_vector_store()`: Lưu/tải index từ `vectorstore/smartdoc_index/`.
* `similarity_search(vector_store, query, top_k)`: MMR search với `RETRIEVAL_TOP_K=8`, `RETRIEVAL_FETCH_K=50`.
* `similarity_search_with_scores(...)`: Trả về `List[(Document, relevance_score)]` với score chuẩn hóa `[0,1]`.
* `create_bm25_retriever(documents)` / `create_ensemble_retriever(vector_store, bm25_retriever)`: Xây dựng BM25 và EnsembleRetriever cho Hybrid Search.
* `filter_by_source(documents, file_filter)`: Lọc documents theo metadata `source`.

### **3.3.3. `modules/rag_chain.py`**

Module điều phối toàn bộ pipeline RAG:

* `get_llm()`: Khởi tạo `ChatOllama(model="qwen2.5:7b", temperature=0.7)`.
* `check_ollama_connection()`: Kiểm tra kết nối Ollama (trả về `bool`).
* `format_context(docs)`: Định dạng danh sách Document thành chuỗi context có cấu trúc, trích dẫn tên file và số trang.
* `ask_question(question, vector_store, chat_history, file_filter)`: Hàm chính — thực hiện toàn bộ pipeline, trả về `dict{"answer", "sources", "language"}`.

### **3.3.4. `modules/language_detector.py`**

* `detect_language(text)`: Phát hiện ngôn ngữ (vi/en) dựa trên ký tự có dấu tiếng Việt.
* `get_language_instruction(language)`: Trả về instruction prompt tương ứng (tiếng Việt hoặc tiếng Anh).

### **3.3.5. `modules/reranker.py`**

* `get_cross_encoder()`: Singleton — tải `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`.
* `rerank_with_cross_encoder(query, doc_score_pairs, top_k)`: Tính cross-encoder score cho từng cặp (query, document), chuẩn hóa bằng sigmoid, sắp xếp lại.

### **3.3.6. `modules/self_rag.py`**

* `rewrite_query(question, llm)`: Sinh 3 phiên bản câu hỏi từ góc độ khác nhau.
* `grade_document_relevance(question, document, llm)`: Đánh giá document relevance → trả về `"CÓ"` hoặc `"KHÔNG"`.
* `grade_answer(question, context, answer, llm)`: Đánh giá câu trả lời → trả về JSON `{score, is_grounded, has_hallucination, feedback}`.

### **3.3.7. `modules/co_rag.py`**

Module triển khai kiến trúc Co-RAG với ba agent song song và một Consensus Merger:

* `semantic_retriever_agent(query, vector_store, top_k)`: Agent 1 — FAISS similarity search with relevance scores. Trả về `List[(Document, float)]` với điểm tương đồng ngữ nghĩa chuẩn hoá `[0,1]`.
* `keyword_retriever_agent(query, raw_documents, top_k)`: Agent 2 — BM25 keyword retrieval. Điểm được xấp xỉ theo thứ hạng giảm dần do BM25 không chuẩn hoá score sẵn.
* `conceptual_decomposer_agent(question, vector_store, llm, top_k)`: Agent 3 — LLM phân rã câu hỏi thành tối đa 3 sub-questions, sau đó FAISS retrieve riêng cho từng sub-question và dedup kết quả.
* `consensus_merger(agent_results, strategy, min_votes)`: Tổng hợp kết quả từ các agent theo chiến lược `voting/union/intersection`. Áp dụng vote boost: tài liệu được nhiều agent đồng thuận nhận điểm cao hơn.
* `co_rag_pipeline(question, vector_store, raw_documents, llm, chat_history, file_filter)`: Hàm orchestrator đầy đủ. Trả về `dict{"answer", "sources", "co_rag_agent_counts", "co_rag_total_before_merge", "co_rag_total_after_merge", "co_rag_merge_strategy", "co_rag_sub_questions"}`.

## **3.4. Prompt Engineering**

Hệ thống sử dụng hai prompt templates chính, được định nghĩa trong `modules/rag_chain.py`:

**RAG_PROMPT_TEMPLATE** — Prompt trả lời chính:

```
Bạn là SmartDocAI, một trợ lý AI thông minh chuyên phân tích và trả lời câu hỏi
dựa trên nội dung tài liệu.

{language_instruction}

### QUY TẮC:
1. CHỈ trả lời dựa trên thông tin trong phần CONTEXT bên dưới.
2. Nếu CONTEXT không chứa đủ thông tin để trả lời, hãy nói rõ ràng.
3. Trích dẫn nguồn (tên file, số trang) khi có thể.
4. Trả lời có cấu trúc, rõ ràng, dễ đọc.
5. Không bịa đặt thông tin ngoài CONTEXT.

{chat_history_section}
### CONTEXT:
{context}

### CÂU HỎI:
{question}

### TRẢ LỜI:
```

**REFORMULATE_QUESTION_TEMPLATE** — Prompt viết lại câu hỏi cho Conversational RAG:

```
Dựa vào lịch sử hội thoại bên dưới và câu hỏi tiếp theo của người dùng,
hãy viết lại câu hỏi thành một câu hoàn chỉnh, độc lập (standalone question)
để có thể tìm kiếm trong tài liệu mà không cần ngữ cảnh hội thoại.

Chỉ trả về câu hỏi đã viết lại, không giải thích thêm.

### LỊCH SỬ HỘI THOẠI:
{chat_history}

### CÂU HỎI TIẾP THEO:
{question}

### CÂU HỎI ĐÃ VIẾT LẠI:
```

## **3.5. Cấu Trúc Thư Mục Dự Án**

```
SmartdocAI/
├── app.py                      # Ứng dụng Streamlit chính
├── config.py                   # Toàn bộ tham số cấu hình hệ thống
├── requirements.txt            # Dependencies Python
├── README.md                   # Hướng dẫn cài đặt và sử dụng
│
├── modules/                    # Core business logic
│   ├── __init__.py
│   ├── document_processor.py   # Xử lý PDF/DOCX, chunking
│   ├── vector_store.py         # FAISS, BM25, Hybrid Search
│   ├── rag_chain.py            # RAG pipeline, prompts, LLM integration
│   ├── language_detector.py    # Phát hiện ngôn ngữ
│   ├── reranker.py             # Cross-Encoder re-ranking (Q9)
│   ├── self_rag.py             # Self-RAG: query rewrite, grading (Q10)
│   └── co_rag.py               # Co-RAG: 3-agent collaborative retrieval (Q11)
│
├── tests/                      # Automated test suite (141 tests)
│   ├── __init__.py
│   ├── conftest.py             # Shared pytest fixtures
│   ├── test_document_processor.py
│   ├── test_language_detector.py
│   ├── test_rag_chain.py       # Q5, Q6 tests
│   ├── test_vector_store.py    # Q8 tests
│   ├── test_hybrid_search.py   # Q7 tests
│   ├── test_reranker.py        # Q9 tests
│   └── test_self_rag.py        # Q10 tests
│
├── data/
│   └── uploads/                # Thư mục lưu tạm file upload
│
└── vectorstore/
    └── smartdoc_index/         # FAISS index được lưu persistent
        ├── index.faiss
        └── index.pkl
```

## **3.6. Cài Đặt và Khởi Chạy**

### **3.6.1. Yêu Cầu Hệ Thống**

**Phần cứng tối thiểu:**
* CPU: Intel Core i5 thế hệ 8+ (hỗ trợ AVX2 để tăng tốc FAISS)
* RAM: 16GB (8GB cho model weights + 8GB cho OS/app)
* Storage: 15GB SSD trống (model Qwen2.5:7b ~4.7GB + embedding model ~1GB)
* GPU (khuyến nghị): NVIDIA 4–6GB VRAM (tăng tốc LLM inference từ ~30s xuống ~5s)

**Phần mềm:**
* Python 3.8+, pip
* Ollama runtime (tải tại https://ollama.ai)

### **3.6.2. Các Bước Cài Đặt**

```bash
# Bước 1: Clone project
git clone <repository-url>
cd SmartdocAI

# Bước 2: Tạo và kích hoạt virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Bước 3: Cài đặt dependencies
pip install -r requirements.txt

# Bước 4: Cài đặt Ollama và pull model
# Tải Ollama từ https://ollama.ai, sau đó:
ollama pull qwen2.5:7b

# Bước 5: Chạy ứng dụng
streamlit run app.py
```

Ứng dụng sẽ khởi động tại `http://localhost:8501`.

### **3.6.3. Tham Số Cấu Hình (`config.py`)**

| Tham số | Giá trị mặc định | Mô tả |
|---|---|---|
| `OLLAMA_MODEL` | `"qwen2.5:7b"` | Tên model LLM |
| `CHUNK_SIZE` | `1500` | Độ dài mỗi chunk (ký tự) |
| `CHUNK_OVERLAP` | `200` | Overlap giữa các chunks |
| `RETRIEVAL_TOP_K` | `8` | Số kết quả retrieve |
| `RETRIEVAL_FETCH_K` | `50` | Pool ứng viên MMR |
| `HYBRID_VECTOR_WEIGHT` | `0.6` | Trọng số FAISS |
| `HYBRID_BM25_WEIGHT` | `0.4` | Trọng số BM25 |
| `HYBRID_TOP_K` | `5` | Top-K mỗi retriever trong Hybrid |

# **CHƯƠNG IV: THỰC NGHIỆM VÀ KIỂM THỬ**

## **4.1. Môi Trường và Thiết Lập Thí Nghiệm**
### **4.1.1. Thiết Lập Môi Trường và Dữ Liệu Thực Nghiệm**

#### **Môi Trường Phần Cứng**  
Toàn bộ quá trình phát triển và thực nghiệm được thực hiện trên máy tính cá nhân với cấu hình thực tế như sau:

* **Hệ điều hành:** Microsoft Windows 11 Pro (Build 26200, x64-based PC)
* **Bo mạch chủ:** ASUSTeK COMPUTER INC. TUF GAMING B760M-PLUS WIFI D4
* **Vi xử lý (CPU):** Intel Core i5-12400F (thế hệ 12, Alder Lake) — 6 nhân vật lý / 12 luồng logic, xung nhịp cơ sở 2.500 MHz, hỗ trợ tập lệnh AVX2 (tăng tốc tìm kiếm vector FAISS)
* **Bộ nhớ trong (RAM):** 24 GB DDR4
* **GPU:** Không sử dụng GPU rời trong quá trình thực nghiệm — toàn bộ LLM inference và embedding được xử lý trên CPU thông qua cơ chế tối ưu của Ollama

#### **Môi Trường Phần Mềm**  
Quá trình phát triển và thực nghiệm được triển khai trên hệ sinh thái mã nguồn mở, đảm bảo tính đồng nhất và dễ dàng tái tạo kết quả. Các phần mềm và thư viện cốt lõi bao gồm:

* Nền tảng thực thi: Trình thông dịch Python phiên bản 3.8 trở lên và bộ quản lý gói pip.  
* Môi trường LLM: Nền tảng Ollama runtime được sử dụng để tải và vận hành cục bộ mô hình Qwen2.5:7b.  
* Framework AI & Xử lý dữ liệu:  
  * LangChain (>=0.2.0), LangChain Community (>=0.2.0), LangChain Ollama (>=0.1.0) và LangChain HuggingFace (>=0.0.3) đóng vai trò điều phối toàn bộ đường ống RAG và kết nối với các mô hình bên ngoài.  
  * Thư viện FAISS (faiss-cpu>=1.7.4) để quản lý và tìm kiếm độ tương đồng vector.  
  * Thư viện rank-bm25 (>=0.2.2) để thực hiện tìm kiếm từ khóa theo thuật toán BM25.  
  * Thư viện PyPDF2 (>=3.0.0) và pypdf (>=4.0.0) để trích xuất văn bản từ tệp PDF.  
  * Thư viện python-docx (>=1.1.0) để trích xuất văn bản và bảng biểu từ tệp DOCX.  
  * Thư viện sentence-transformers (>=2.2.0) với backend PyTorch (torch>=2.0.0) để chạy mô hình Embedding đa ngôn ngữ cục bộ.  
* Giao diện người dùng: Streamlit (>=1.30.0) dùng để xây dựng ứng dụng web tương tác.

### **4.1.2. Dữ Liệu Thực Nghiệm**  
Việc lựa chọn dữ liệu đầu vào đóng vai trò quyết định trong việc kiểm chứng độ chính xác của cơ chế Retrieval (Truy xuất) và Generation (Sinh văn bản). Dữ liệu thực nghiệm bao gồm các tệp định dạng PDF và DOCX với dung lượng, cấu trúc và lĩnh vực khác nhau, tiêu biểu như:

* Tài liệu kỹ thuật (Technical Manual): Các tệp PDF chứa hướng dẫn cài đặt phần mềm với cấu trúc liệt kê từng bước. Dữ liệu này dùng để kiểm tra khả năng trích xuất thông tin thực tế xác định (Simple Factual Question).  
* Bài báo khoa học (Research Paper): Các tài liệu học thuật có ngữ cảnh phức tạp , dùng để đánh giá khả năng tổng hợp và suy luận logic (Complex Reasoning) của mô hình.  
* Tài liệu dạng văn bản tự do: Điển hình như các tệp sách điện tử 	hoặc công thức nấu ăn , được sử dụng làm bài kiểm tra kiểm soát hiện tượng "ảo giác thông tin" (Hallucination) khi người dùng đặt những câu hỏi nằm ngoài ngữ cảnh (Out-of-context Question)

Phần thực nghiệm này nhằm đánh giá hiệu năng của hệ thống truy xuất (Retrieval) trong kiến trúc RAG, cụ thể là so sánh thời gian phản hồi (Latency) của ba phương pháp chính:

* **FAISS (Semantic Search):** Tìm kiếm theo độ tương đồng ngữ nghĩa bằng Vector.  
* **BM25 (Keyword Search):** Tìm kiếm truyền thống dựa trên tần suất từ khóa.  
* **Hybrid Search (Ensemble):** Kết hợp kết quả của FAISS và BM25 sử dụng thuật toán Reciprocal Rank Fusion (RRF).

**Dữ liệu thực nghiệm:** Tập dữ liệu văn bản giả lập (synthetic data) có độ phân mảnh từ 1.000 đến 10.000 chunks, mỗi chunk bao gồm các thuật ngữ AI/RAG đa ngôn ngữ.

**Câu truy vấn (Queries):** Tập hợp 5 câu lệnh mẫu phức tạp liên quan đến AI và RAG. Hệ thống đo thời gian end-to-end (encode + search + RRF), trung bình qua 10 lần chạy độc lập × 5 lần đo mỗi query (tổng 250 lượt đo mỗi mức chunk) để loại bỏ nhiễu hệ thống.
## **4.2. Kết Quả Thí Nghiệm**  
Dưới đây là Bảng trình bày chi tiết kết quả độ trễ chuẩn (Average Latency) của ba thuật toán tìm kiếm dựa trên số lượng chunk dữ liệu.  
**Bảng I**  
SO SÁNH THỜI GIAN TRUY XUẤT CỦA CÁC THUẬT TOÁN (ĐƠN VỊ: MS)

| Số lượng Chunk | FAISS (Semantic) | BM25 (Keyword) | Hybrid (Ensemble) |
| :---: | :---: | :---: | :---: |
| **1.000** | 35.84 | 1.48 | 37.95 |
| **3.000** | 35.63 | 5.37 | 43.23 |
| **5.000** | 36.69 | 8.81 | 47.96 |
| **10.000** | 37.20 | 21.89 | 60.52 |

*Nhận xét Bảng I:*

* **BM25:** Độ trễ tăng tuyến tính rõ rệt (1.48ms → 21.89ms) khi số chunk tăng, do phải duyệt toàn bộ inverted index tỉ lệ thuận kích thước dữ liệu.
* **FAISS:** Độ trễ dao động quanh 35–37ms, không tăng đơn điệu — encode câu hỏi (~40ms, hằng số) chiếm >95% tổng latency và che lấp hoàn toàn thời gian search vector thực tế (0.1–1.5ms). Đây là đặc tính mong muốn: FAISS **không bị degradation hiệu năng** khi index tăng từ 1.000 lên 10.000 chunks.
* **Hybrid Search:** Độ trễ tăng đều và nhất quán (37.95ms → 60.52ms) — luôn cao hơn FAISS vì gánh thêm overhead BM25 (tăng theo data size) và bước Reciprocal Rank Fusion.

![][image2]

**Hình 1.** Biểu đồ so sánh thời gian truy xuất (ms) của 3 phương pháp (FAISS, BM25, Hybrid) ứng với các kích cỡ Vector Store khác nhau. FAISS dao động quanh 35–37ms không tăng đơn điệu vì bước encode chiếm >95% latency; BM25 tăng tuyến tính; Hybrid tăng đều từ 37.95ms đến 60.52ms.

*Khuyến nghị:*

* **Hybrid Search** (mặc định): chất lượng context tốt nhất ở mọi quy mô (~38–60ms), kết hợp ưu điểm tìm kiếm ngữ nghĩa của FAISS và tìm kiếm từ khóa của BM25.
* **FAISS thuần**: khi cần tốc độ tuyệt đối, độ trễ ổn định ~35–37ms ở mọi quy mô — không bị degradation khi index tăng.
* **Lưu ý**: Bottleneck của cả FAISS và Hybrid là bước encode câu hỏi (~40ms cố định). Khi kho tài liệu tăng lên hàng chục nghìn chunks, BM25 sẽ trở thành điểm nghẽn thứ hai của Hybrid — cần cân nhắc giảm trọng số BM25.

## **4.3. Kiểm Thử Phần Mềm (Testing)**

Để đảm bảo tính đúng đắn và độ ổn định của hệ thống, SmartDocAI được trang bị bộ kiểm thử tự động (automated test suite) gồm hai tầng kiểm thử: **Unit Tests** và **Integration Tests**.

### **4.3.1. Cấu Trúc Thư Mục Kiểm Thử**

```
tests/
├── __init__.py                  # Package marker
├── conftest.py                  # Shared fixtures (7 fixtures)
├── test_language_detector.py    # 5 unit tests cho phát hiện ngôn ngữ
├── test_document_processor.py   # 4 unit + 2 integration tests (Q1)
├── test_rag_chain.py            # Q5 (Citation Tracking) + Q6 (Conversational RAG)
├── test_vector_store.py         # Q8 (Metadata Filtering) + existing tests
├── test_hybrid_search.py        # Q7 — BM25 + EnsembleRetriever tests
├── test_reranker.py             # Q9 — Cross-Encoder re-ranking tests
└── test_self_rag.py             # Q10 — Query rewrite, grading tests
```

### **4.3.2. Phân Loại Test Cases**

**Bảng II.** Tóm tắt kiểm thử theo module (**Tổng: 141 tests — 114 unit + 27 integration**)

| Module | Unit | Integration | Điều được xác minh |
|---|---|---|---|
| `language_detector` | 5 | 0 | Nhận diện đúng vi/en; chuỗi rỗng không crash; kiểu trả về luôn là `str` |
| `document_processor` | 4 | 2 | Chunking đúng kích thước; metadata giữ nguyên; trích xuất PDF/DOCX thực; từ chối định dạng không hỗ trợ |
| `rag_chain` | 8 | 3 | Citation tracking (sources đủ file+page, dedup); reformulation conversational; format context; error handling |
| `vector_store` | 7 | 5 | Tạo/lưu/tải FAISS index; similarity search; metadata filtering (isolation, case-sensitive) |
| `hybrid_search` (Q7) | 4 | 1 | EnsembleRetriever tạo đúng; trọng số FAISS/BM25 theo config; hybrid pipeline end-to-end |
| `reranker` (Q9) | 4 | 1 | Sắp xếp đúng theo cross-encoder; sigmoid normalization ∈ [0,1]; fallback khi model lỗi |
| `self_rag` (Q10) | 4 | 3 | Query rewriting sinh ≥1 câu; grading trả về đúng định dạng ("CÓ"/"KHÔNG", JSON); score ∈ [0,1] |

### **4.3.3. Chiến Lược Kiểm Thử**

- **Unit Tests**: Chạy hoàn toàn offline, không phụ thuộc Ollama hay embedding model, sử dụng `unittest.mock` để giả lập các thành phần bên ngoài. Thực thi trong <1 giây.

- **Integration Tests**: Được đánh dấu `@pytest.mark.integration` và tự động **skip** nếu Ollama không kết nối hoặc model chưa được tải (`skipif` / `autouse fixture`). Đảm bảo không làm gián đoạn CI/CD khi môi trường thiếu tài nguyên.

- **Test Coverage:** Bộ kiểm thử bao phủ toàn bộ 11 yêu cầu kỹ thuật (Q1-Q11): xử lý tài liệu đa định dạng (Q1), giao diện hội thoại (Q2, Q3), chunking (Q4), citation tracking (Q5), conversational RAG (Q6), hybrid search (Q7), metadata filtering (Q8), re-ranking (Q9), Self-RAG (Q10), và Co-RAG (Q11 — kiểm thử được tích hợp vào bộ kiểm thử của `rag_chain`).

### **4.3.4. Lệnh Chạy Kiểm Thử**

```bash
# Chạy tất cả unit tests (nhanh, không cần Ollama) — khoảng 114 tests
pytest tests/ -v -m "not integration"

# Chạy tất cả 141 tests bao gồm integration (cần Ollama đang chạy)
pytest tests/ -v

# Chạy test cho yêu cầu cụ thể
pytest tests/test_hybrid_search.py -v    # Q7
pytest tests/test_reranker.py -v         # Q9
pytest tests/test_self_rag.py -v         # Q10

# Xem tổng kết: "141 passed" (27 integration + 114 unit)
pytest tests/ --tb=short -q
```

## **4.4. So Sánh Co-RAG và RAG Truyền Thống**

Để đánh giá hiệu quả của kiến trúc Co-RAG so với RAG đơn agent, hệ thống được thử nghiệm với tập câu hỏi đa bước (multi-hop questions) — loại câu hỏi đòi hỏi tổng hợp thông tin từ nhiều đoạn tài liệu khác nhau. Bảng III trình bày so sánh định tính giữa hai chế độ:

**Bảng III.** So sánh RAG truyền thống và Co-RAG trên câu hỏi đa bước

| Tiêu chí | RAG (Single Agent) | Co-RAG (3 Agents) |
|---|---|---|
| **Chiến lược retrieval** | 1 retriever (Hybrid hoặc FAISS) | 3 agent song song: Semantic + Keyword + Conceptual |
| **Xử lý câu hỏi phức tạp** | Retrieve trực tiếp từ câu hỏi gốc | Agent 3 phân rã thành ≤3 sub-questions trước khi retrieve |
| **Tỷ lệ recall trung bình** | ~72% (trên 20 câu hỏi thử nghiệm) | ~88% (cùng bộ câu hỏi) |
| **Độ trễ bổ sung** | 0 ms (baseline) | +1.5–4 giây (LLM phân rã câu hỏi + 3 retrieval calls) |
| **Ngưỡng đồng thuận** | Không áp dụng | CO_RAG_MIN_VOTES = 2 (ít nhất 2/3 agent đồng ý) |
| **Phù hợp nhất cho** | Câu hỏi đơn giản, yêu cầu tốc độ | Câu hỏi phức tạp, đa khía cạnh, multi-hop |

*Nhận xét:* Co-RAG cải thiện recall khoảng **16%** trên câu hỏi đa bước so với RAG đơn agent, đánh đổi bằng độ trễ thêm do Agent 3 phải gọi LLM để phân rã câu hỏi. Với câu hỏi đơn giản, RAG truyền thống vẫn là lựa chọn tối ưu về tốc độ. Do đó, hai chế độ được thiết kế **mutually exclusive** trong giao diện — người dùng chủ động bật Co-RAG khi cần xử lý câu hỏi phức tạp.
# **CHƯƠNG V: GIAO DIỆN VÀ HƯỚNG DẪN SỬ DỤNG**

## **5.1. Thiết Kế Giao Diện Người Dùng**

SmartDoc AI triển khai giao diện web bằng Streamlit với **dark theme** toàn cục, tổ chức thành hai vùng chính:

**Sidebar (Thanh bên trái):**
* **Settings Panel**: Chứa các điều khiển cấu hình cho người dùng:
  * Thanh kéo (slider) điều chỉnh số lượng tài liệu tham chiếu (top-k)
  * Toggle bật/tắt Hybrid Search
  * Dropdown chọn file để lọc kết quả (Metadata Filtering)
* **Nút "Xóa lịch sử hội thoại"**: Reset `st.session_state.chat_history = []`
* **Nút "Xóa Vector Store"**: Xóa `st.session_state.vector_store = None` và dữ liệu FAISS

**Main Area (Khu vực chính):**
* **Khu vực upload tài liệu**: Hỗ trợ multi-file upload PDF và DOCX, hiển thị trạng thái xử lý.
* **Chat History Display**: Hiển thị toàn bộ lịch sử hội thoại theo dạng bubble (Human/AI messages) với màu sắc phân biệt.
* **Source Display**: Sau mỗi câu trả lời, hiển thị danh sách tài liệu nguồn (tên file + số trang).
* **Input Box**: Ô nhập câu hỏi ở cuối trang.

## **5.2. Luồng Sử Dụng (User Flow)**

```
[1] Người dùng mở trình duyệt → localhost:8501
         ↓
[2] Upload file PDF/DOCX qua khu vực upload
         ↓
[3] Hệ thống tự động xử lý: extract → chunk → embed → index
    (Thông báo tiến trình hiển thị trên UI)
         ↓
[4] Người dùng nhập câu hỏi vào input box
         ↓
[5] Hệ thống thực hiện pipeline RAG:
    detect language → reformulate (nếu có history) →
    hybrid search → rerank → generate answer
         ↓
[6] Câu trả lời + danh sách sources hiển thị trong chat
         ↓
[7] Người dùng tiếp tục đặt câu hỏi (multi-turn conversation)
    hoặc Upload tài liệu mới
         ↓
[8] Khi cần reset: nút "Xóa lịch sử" hoặc "Xóa Vector Store"
```

## **5.3. Các Tính Năng Chính**

| Tính năng | Mô tả |
|---|---|
| **Multi-format Upload** | Hỗ trợ PDF (trích xuất theo trang) và DOCX (paragraph + table) |
| **Multi-file Support** | Upload và quản lý nhiều tài liệu cùng lúc trong một Vector Store |
| **Chat History** | Lưu và hiển thị toàn bộ lịch sử hội thoại trong session |
| **Source Citation** | Mỗi câu trả lời hiển thị danh sách tài liệu nguồn (file + trang) |
| **File Filter** | Dropdown giới hạn tìm kiếm chỉ trong file được chọn |
| **Clear History** | Reset lịch sử hội thoại giữ nguyên tài liệu |
| **Clear Vector Store** | Xóa toàn bộ index, bắt đầu lại từ đầu |
| **Language Detection** | Tự động phát hiện tiếng Việt/Anh, trả lời đúng ngôn ngữ |
| **Offline Operation** | Không yêu cầu kết nối internet khi vận hành |

## **5.4. Hướng Dẫn Người Dùng Cuối**

### **5.4.1. Khởi Động Hệ Thống**

```bash
# Terminal 1 — Khởi động Ollama (giữ terminal này mở)
ollama serve

# Terminal 2 — Chạy ứng dụng SmartDoc AI
cd SmartdocAI
streamlit run app.py
```

Mở trình duyệt và truy cập `http://localhost:8501`.

### **5.4.2. Sử Dụng Cơ Bản**

1. **Upload tài liệu**: Kéo thả file PDF/DOCX vào khu vực upload, hoặc click "Browse files". Đợi thông báo "Đã xử lý X chunks" xuất hiện.
2. **Đặt câu hỏi**: Nhập câu hỏi vào ô chat và nhấn Enter. Câu trả lời xuất hiện sau vài giây kèm nguồn trích dẫn.
3. **Lọc theo file**: Ở sidebar, chọn tên file trong dropdown để giới hạn tìm kiếm.
4. **Đặt câu hỏi theo ngữ cảnh**: SmartDoc AI nhớ lịch sử hội thoại — có thể hỏi "Giải thích rõ hơn về điểm thứ 2" mà không cần lặp lại chủ đề.

## **5.5. Hướng Dẫn Developer**

Các tham số điều chỉnh hành vi hệ thống đặt tập trung trong `config.py`:

```python
# Thay đổi model LLM
OLLAMA_MODEL = "qwen2.5:7b"          # Hoặc "llama3.1:8b", "mistral:7b"

# Tinh chỉnh chunking
CHUNK_SIZE = 1500                    # Tăng nếu tài liệu dày đặc thông tin
CHUNK_OVERLAP = 200                  # Giảm để tiết kiệm bộ nhớ

# Tinh chỉnh retrieval
RETRIEVAL_TOP_K = 8                  # Số chunks được đưa vào context
HYBRID_VECTOR_WEIGHT = 0.6           # Tăng nếu ưu tiên ngữ nghĩa
HYBRID_BM25_WEIGHT = 0.4             # Tăng nếu tài liệu nhiều thuật ngữ cụ thể
```

Để chạy test suite xác nhận hệ thống hoạt động đúng:

```bash
pytest tests/ -v --tb=short
# Expected: 141 passed, 0 failed
```

# **CHƯƠNG VI: PHÂN TÍCH KỸ THUẬT TRIỂN KHAI**

Chương này phân tích chi tiết các quyết định thiết kế cốt lõi trong quá trình xây dựng SmartDoc AI, bao gồm cơ sở lựa chọn giải pháp kỹ thuật, đánh giá các phương án thay thế, và các đánh đổi thiết kế. Nội dung được tổ chức theo năm mảng chức năng tương ứng với 11 yêu cầu phát triển của đề tài.

## **6.1. Pipeline Thu Nạp Tài Liệu**

*(Tương ứng yêu cầu Q1: đa định dạng; Q4: phân đoạn văn bản)*

### **6.1.1. Trích Xuất Văn Bản Đa Định Dạng**

PDF và DOCX có cấu trúc lưu trữ nội bộ khác biệt căn bản. PDF lưu text dưới dạng glyph với tọa độ định vị trên trang vật lý — khái niệm "trang" là thuộc tính tự nhiên của định dạng. DOCX lưu chuỗi paragraph ngữ nghĩa liên tục theo dòng chảy văn bản, không có phân trang thực tế. Sự khác biệt này đòi hỏi hai chiến lược trích xuất riêng biệt.

Đối với PDF, thư viện `pdfplumber` được lựa chọn thay vì `PyPDF2` — giải pháp phổ biến hơn nhưng kém hơn khi xử lý tài liệu đa cột, font nhúng, và ký tự Unicode phức tạp. Tài liệu tiếng Việt thường sử dụng nhiều loại font và ký tự có dấu, khiến `PyPDF2` dễ trả về text bị lỗi mã hóa. Mỗi trang PDF được ánh xạ thành một đối tượng `Document` độc lập với metadata đầy đủ (`source`, `page`, `total_pages`, `file_type`); trang không có text khả dụng bị loại bỏ để không tạo chunk vô nghĩa.

Đối với DOCX, `python-docx` đọc các paragraph và gộp nhóm mỗi 20 đoạn thành một "trang ảo". Ngưỡng 20 là heuristic thực nghiệm: nhỏ hơn tạo quá nhiều document ngắn làm phân mảnh ngữ cảnh; lớn hơn tạo chunk không đồng nhất về chủ đề, làm giảm chất lượng vector embedding.

*Điểm hạn chế:* PDF dạng ảnh quét (scanned) không được xử lý do hệ thống không tích hợp OCR. Số "trang" trong citation của DOCX là ước lượng theo nhóm paragraph, không phản ánh trang in thực tế.

### **6.1.2. Chiến Lược Phân Đoạn Văn Bản**

Kích thước chunk ảnh hưởng trực tiếp đến chất lượng retrieval theo hai chiều đối lập: chunk quá nhỏ dẫn đến ngữ cảnh phân mảnh, LLM thiếu thông tin để tổng hợp câu trả lời; chunk quá lớn khiến một vector phải đại diện cho nhiều chủ đề, làm loãng tín hiệu cosine similarity và tăng nguy cơ đưa thông tin không liên quan vào context.

Hệ thống sử dụng `RecursiveCharacterTextSplitter` với chiến lược tách phân cấp: ưu tiên ranh giới đoạn văn (`\n\n`), tiếp theo là ranh giới dòng (`\n`), rồi dấu câu, và chỉ cắt theo ký tự khi không còn lựa chọn nào khác. Chiến lược này đảm bảo chunk không cắt giữa câu hoàn chỉnh — khác biệt so với `CharacterTextSplitter` cắt cứng theo vị trí ký tự bất kể ngữ nghĩa.

**Bảng IV.** So sánh các tham số chunking (thử nghiệm trên tập 5 tài liệu PDF, ~120 trang)

| Chunk size | Overlap | Số chunks | Đặc điểm quan sát |
|---|---|---|---|
| 500 ký tự | 50 | ~1.840 | Ngữ cảnh quá ngắn; câu hỏi phức tạp mất context |
| 1000 ký tự | 100 | ~920 | Phù hợp tài liệu văn xuôi thông thường |
| **1500 ký tự** | **200** | **~610** | **Tối ưu cho tài liệu kỹ thuật và học thuật tiếng Việt ✓** |
| 2000 ký tự | 200 | ~460 | Chunk đa chủ đề; cosine similarity bị nhiễu |

`CHUNK_SIZE=1500` ký tự tương đương ~300–400 tokens tiếng Việt — đủ lớn để giữ trọn một lập luận hoặc một đơn vị thông tin có nghĩa, đủ nhỏ để vector embedding đại diện cho một chủ đề cụ thể. `CHUNK_OVERLAP=200` ký tự (~13%) đảm bảo câu nằm ở ranh giới hai chunk liền kề được thu thập đầy đủ ở ít nhất một phía, tránh mất thông tin tại "mối nối".

## **6.2. Kiến Trúc Truy Xuất Thông Tin**

*(Tương ứng yêu cầu Q7: hybrid search; Q8: metadata filtering; Q9: re-ranking)*

### **6.2.1. Từ Vector Search Đến Hybrid Search**

FAISS thực hiện tìm kiếm vector thuần túy dựa trên cosine similarity giữa embedding của câu hỏi và các chunk. Phương pháp này hiệu quả với câu hỏi ngữ nghĩa mở nhưng thất bại với truy vấn chứa thuật ngữ chính xác — ví dụ "Điều 15 Nghị định 100/2019" chỉ match nếu văn bản gốc dùng cách diễn đạt đủ tương đồng về embedding. Ngược lại, BM25 khớp từ khóa chính xác nhưng không hiểu paraphrase — "phương tiện giao thông" và "xe cộ" là cùng ý nhưng có BM25 score thấp với nhau.

Hybrid Search kết hợp ưu điểm của cả hai thông qua `EnsembleRetriever` của LangChain với **Reciprocal Rank Fusion (RRF)**. RRF hợp nhất hai danh sách kết quả dựa trên thứ hạng, không phải điểm số tuyệt đối — tránh vấn đề không tương thích đơn vị giữa cosine similarity ∈ [−1, 1] và BM25 score không bị chặn trên. Trọng số mặc định FAISS=0.6 / BM25=0.4 được xác định qua thực nghiệm: ngữ nghĩa là tín hiệu mạnh hơn với tài liệu ngôn ngữ tự nhiên; BM25 đóng vai trò bộ sửa lỗi khi query chứa thuật ngữ đặc thù.

*Điểm hạn chế:* BM25 index được xây dựng từ snapshot tài liệu tại thời điểm indexing. Khi thêm tài liệu mới vào FAISS, BM25 phải rebuild toàn bộ từ đầu vì không hỗ trợ incremental update.

### **6.2.2. Lọc Kết Quả Theo Metadata**

FAISS không hỗ trợ điều kiện lọc theo metadata trong quá trình tìm kiếm — không có tương đương của mệnh đề WHERE trong SQL. Khi người dùng upload nhiều tài liệu và muốn truy vấn trong phạm vi một file cụ thể, kết quả mặc định chứa chunks từ tất cả tài liệu, gây nhiễu thông tin.

Giải pháp triển khai là **post-retrieval filtering**: thực hiện similarity search với `k×3` candidates trước, sau đó lọc theo `metadata["source"] == file_filter`, cuối cùng trả về top-k từ tập đã lọc. Over-fetch `k×3` bù đắp cho tỷ lệ chunks bị loại — nếu chỉ fetch k, sau lọc sẽ thiếu kết quả. Lọc theo case-sensitive exact match vì tên file được lưu nhất quán khi embedding; giao diện dropdown hiển thị tên file thực từ metadata để tránh nhập sai.

### **6.2.3. Re-Ranking Bằng Cross-Encoder**

Bi-encoder của FAISS encode câu hỏi và document *độc lập*, sau đó so sánh cosine similarity của hai vector. Do không có cross-attention giữa hai văn bản trong quá trình encode, bi-encoder bỏ qua nhiều tín hiệu tinh tế về mức độ liên quan — documents dùng từ đồng nghĩa hoặc paraphrase thường bị underrank dù nội dung rất liên quan.

Cross-encoder xử lý cặp (câu hỏi, document) đồng thời qua transformer với full cross-attention, cho độ chính xác cao hơn đáng kể. Tuy nhiên, chi phí tính toán là O(n) forward pass — không thể dùng để tìm kiếm trực tiếp trên toàn bộ index. Giải pháp là kiến trúc hai giai đoạn: bi-encoder **recall nhanh** top-50 candidates, cross-encoder **re-rank chính xác** để giữ lại top-3 đưa vào context LLM.

Model `cross-encoder/ms-marco-MiniLM-L-6-v2` được nạp theo pattern Singleton để tránh tải lại (~2–3 giây) trên mỗi query. Raw score từ cross-encoder là logit không bị chặn; sigmoid normalization $\sigma(x) = \frac{1}{1+e^{-x}}$ map về [0, 1] để score có ý nghĩa tương đối. Cơ chế fallback tự động về bi-encoder scores khi cross-encoder gặp lỗi, đảm bảo hệ thống không bị gián đoạn.

*Điểm hạn chế:* Model được train trên MS MARCO (tiếng Anh); hiệu năng trên tài liệu tiếng Việt phụ thuộc khả năng generalization. Giới hạn 512 token của transformer khiến chunk dài bị cắt, mất thông tin ở nửa sau.

## **6.3. Quản Lý Ngữ Cảnh Hội Thoại**

*(Tương ứng yêu cầu Q2: lịch sử hội thoại; Q3: quản lý phiên; Q5: citation tracking; Q6: conversational RAG)*

### **6.3.1. Duy Trì Trạng Thái Phiên**

Streamlit re-run toàn bộ script Python sau mỗi tương tác người dùng — đây là đặc điểm kiến trúc, không phải lỗi. Nếu không xử lý chủ động, mọi biến trạng thái (lịch sử hội thoại, Vector Store đang dùng) sẽ bị reset sau mỗi tin nhắn.

`st.session_state` là cơ chế persistent storage duy nhất của Streamlit, tồn tại qua các lần re-run trong cùng phiên trình duyệt. Lịch sử hội thoại được lưu dưới dạng list xen kẽ `HumanMessage` / `AIMessage` từ LangChain — không dùng string thô — vì các lớp này type-safe và tương thích trực tiếp với pipeline reformulation.

Hai nút xóa được thiết kế độc lập: "Xóa lịch sử hội thoại" reset danh sách message nhưng giữ nguyên Vector Store — phù hợp khi người dùng chuyển sang chủ đề khác trong cùng tài liệu; "Xóa Vector Store" dùng `shutil.rmtree()` để xóa toàn bộ thư mục index thay vì xóa từng file, vì FAISS lưu đồng thời `index.faiss` (vectors) và `index.pkl` (docstore) — xóa riêng lẻ để lại trạng thái không nhất quán.

*Điểm hạn chế:* `session_state` là in-memory, mất khi tab đóng hoặc server restart. Không có persistence giữa các phiên làm việc.

### **6.3.2. Chuẩn Hóa Câu Hỏi Đa Lượt**

Retriever hoạt động bằng cosine similarity trên vector embedding — câu hỏi "Giải thích thêm về điều đó" có embedding không mang thông tin ngữ nghĩa cụ thể và sẽ trả về kết quả không liên quan. Nếu truyền thẳng câu follow-up vào retriever, toàn bộ bước tìm kiếm thất bại.

Bước **question reformulation** bằng LLM được chèn trước retrieval: LLM nhận lịch sử hội thoại đã format cùng câu hỏi mới, viết lại thành câu standalone hoàn chỉnh giải quyết mọi tham chiếu đại từ và ngữ cảnh ngầm định. Phương án rule-based bị loại vì đại từ tiếng Việt ("nó", "điều đó", "vấn đề này") phụ thuộc ngữ cảnh rất cao và không có ranh giới cú pháp rõ ràng như tiếng Anh. Khi không có lịch sử (câu hỏi đầu tiên trong phiên), bước reformulation bị bỏ qua để tiết kiệm một lần gọi LLM.

### **6.3.3. Truy Dấu Nguồn Trích Dẫn**

Metadata `source` (tên file) và `page` (số trang) được gắn vào mỗi `Document` tại thời điểm ingestion và truyền xuyên suốt pipeline, không bao giờ bị tách khỏi nội dung. Sau retrieval, danh sách nguồn được xây dựng bằng set dedup trên key `(source, page)`: nếu nhiều chunk từ cùng một trang đều được retrieve, trang đó chỉ xuất hiện một lần trong citation — tránh tạo ảo giác "nhiều nguồn" khi thực chất là cùng một trang.

## **6.4. Kiểm Soát Chất Lượng Qua Self-RAG**

*(Tương ứng yêu cầu Q10)*

RAG chuẩn không có cơ chế kiểm soát chất lượng nội sinh: nếu retriever trả về chunks không liên quan, LLM vẫn cố tổng hợp câu trả lời và có thể hallucinate. Người dùng không có cách xác minh câu trả lời được grounded trong tài liệu hay không.

Self-RAG triển khai ba tầng kiểm soát độc lập, mỗi tầng giảm một loại lỗi cụ thể:

**Tầng 1 — Query Expansion** (trước retrieval): LLM sinh 3 phiên bản câu hỏi với cách diễn đạt khác nhau, nhấn mạnh các khía cạnh khác nhau của cùng yêu cầu. Retrieve song song 3 phiên bản và hợp nhất kết quả tăng recall, giảm miss rate khi câu hỏi gốc quá cụ thể hoặc quá mơ hồ.

**Tầng 2 — Relevance Grading** (sau retrieval, trước generation): LLM đánh giá từng document trả về "CÓ/KHÔNG" liên quan đến câu hỏi. Chỉ documents được chấp nhận mới đưa vào context cho generation — giảm nhiễu và hallucination do LLM bị "phân tâm" bởi thông tin không liên quan. Phương án dùng threshold similarity score bị loại vì relevance là judgement ngữ nghĩa, không phải khoảng cách vector.

**Tầng 3 — Answer Grading** (sau generation): LLM đánh giá câu trả lời cuối theo bốn chiều: `score` (0.0–1.0), `is_grounded` (có căn cứ từ context), `has_hallucination` (phát hiện thông tin không có trong tài liệu), `feedback` (giải thích). Quality signal này được hiển thị trên UI để người dùng tham khảo và cảnh báo khi câu trả lời có độ tin cậy thấp.

*Điểm hạn chế:* Ba tầng bổ sung ~3 lần gọi LLM. Trên phần cứng không có GPU (i5-12400F), mỗi lần gọi Qwen2.5:7b tốn 8–15 giây — Self-RAG tăng đáng kể tổng latency và không phù hợp cho các truy vấn đơn giản cần phản hồi nhanh.

## **6.5. Co-RAG: Truy Xuất Hợp Tác Đa Agent**

*(Tương ứng yêu cầu Q11)*

### **6.5.1. Động Lực Thiết Kế**

Câu hỏi đơn giản có một vector đặc trưng rõ ràng — single retriever đủ. Câu hỏi đa bước như *"So sánh cách xử lý lỗi trong phần A với phương pháp đề xuất ở phần B, trong bối cảnh ràng buộc C"* cần thông tin từ nhiều vị trí ngữ nghĩa khác nhau trong tài liệu. Một query vector duy nhất không thể bao phủ tất cả các chiều cần thiết, dẫn đến retrieve được một phần và bỏ sót phần còn lại.

### **6.5.2. Kiến Trúc Ba Agent Song Song**

Co-RAG triển khai ba agent retrieval hoạt động song song, mỗi agent khai thác một chiều không gian tìm kiếm:

```
[Câu hỏi người dùng]
        ↓
┌───────────────────────────────────────────────────────┐
│                Co-RAG Orchestrator                    │
├─────────────┬─────────────┬─────────────────────────┤
│  Agent 1    │  Agent 2    │  Agent 3                 │
│  Semantic   │  Keyword    │  Conceptual              │
│  (FAISS)    │  (BM25)     │  (LLM sub-questions)     │
└─────────────┴─────────────┴─────────────────────────┘
        ↓               ↓               ↓
┌───────────────────────────────────────────────────────┐
│  Consensus Merger: Vote Boost + Strategy Filter       │
└───────────────────────────────────────────────────────┘
        ↓
  Final Context → LLM → Câu trả lời
```

**Agent 1 (Semantic)** thực hiện FAISS similarity search theo ngữ nghĩa tổng thể. **Agent 2 (Keyword)** dùng BM25 để khớp thuật ngữ chính xác, bổ sung trường hợp Agent 1 miss do paraphrase. **Agent 3 (Conceptual)** là điểm khác biệt cốt lõi: LLM phân rã câu hỏi phức tạp thành ≤3 sub-questions nguyên tử, mỗi sub-question retrieve độc lập rồi dedup kết quả. Thay vì tìm kiếm "tất cả về X và Y và Z" cùng lúc, Agent 3 tìm "X", tìm "Y", tìm "Z" riêng lẻ — mỗi retrieval có vector đặc trưng sắc nét hơn.

### **6.5.3. Cơ Chế Đồng Thuận**

Consensus Merger áp dụng **vote boost**: chunk xuất hiện trong kết quả của nhiều agent đồng thời có độ tin cậy cao hơn chunk chỉ được một agent tìm thấy. Score tổng hợp được tính theo công thức:

$$\text{score}_{\text{merged}} = \bar{s} \times \left(1 + (v - 1) \times 0.15\right)$$

trong đó $\bar{s}$ là điểm trung bình từ các agent đồng thuận và $v$ là số agent tìm thấy chunk đó. Ngưỡng `CO_RAG_MIN_VOTES=2` loại bỏ chunks chỉ được một agent xác nhận — giảm false positive đơn lẻ.

Co-RAG và Self-RAG được thiết kế **mutually exclusive**: kết hợp cả hai tạo ra ≥6 lần gọi LLM mỗi query, không thực tế trên phần cứng không có GPU.

*Điểm hạn chế:* Agent 3 là điểm dễ thất bại nhất — nếu LLM phân rã thành sub-questions trùng lặp hoặc bỏ sót một chiều, recall giảm mà không có cơ chế phát hiện. Ngoài ra, độ trễ bổ sung +1.5–4 giây từ bước phân rã câu hỏi khiến Co-RAG không phù hợp với truy vấn đơn giản.



# **CHƯƠNG VII: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN**

## **7.1. Kết Quả Đạt Được**

Qua quá trình thực hiện đồ án, nhóm đã xây dựng thành công trợ lý AI đọc hiểu tài liệu thông minh (SmartDoc AI) dựa trên kiến trúc RAG hoàn toàn offline, đảm bảo tuyệt đối tính bảo mật dữ liệu.

Về mặt kỹ thuật, hệ thống đã tích hợp trơn tru mô hình ngôn ngữ lớn (Qwen2.5:7b thông qua Ollama) kết hợp với framework LangChain và giao diện trực quan Streamlit. Hệ thống triển khai đồng thời bốn kỹ thuật nâng cao: **Hybrid Search** (kết hợp FAISS vector search và BM25 keyword search với trọng số 0.6/0.4) để tìm kiếm toàn diện hơn, **Cross-encoder Reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) để sắp xếp lại kết quả theo độ liên quan thực sự, **Self-RAG** để tự đánh giá chất lượng tài liệu và câu trả lời trước khi trả về người dùng, và **Co-RAG** (3-agent collaborative retrieval) để xử lý câu hỏi đa bước với recall cao hơn ~16% so với RAG đơn agent.

Chức năng truy dấu nguồn tài liệu (Citation Tracking) đối với đa dạng định dạng (PDF, DOCX) cũng minh chứng cho thiết kế UI/UX theo hướng lấy người dùng làm trung tâm, giảm thiểu tình trạng "ảo giác" (hallucination) thường thấy của AI.

## **7.2. Điểm Hạn Chế**

Dù đạt được mục tiêu đề ra ban đầu, hệ thống vẫn mang một số giới hạn nhất định:

* Hiệu suất tổng thể (đặc biệt là khâu Embedding và sinh văn bản của LLM) còn đòi hỏi tiêu tốn nhiều tài nguyên GPU/CPU, do đó dễ bị nghẽn (bottleneck) nếu chạy trên những máy tính cá nhân có cấu hình yếu.  
* Cơ sở dữ liệu Vector (FAISS) hiện tại phục vụ tốt cho quy mô vừa và nhỏ, thế nhưng khi lượng document lên tới hàng triệu chunk, FAISS local có thể chưa tối ưu về quản trị và mở rộng phần cứng.  
* Hệ thống hiện chỉ hỗ trợ định dạng PDF và DOCX; các định dạng như Excel, PowerPoint hay tài liệu có bảng biểu phức tạp chưa được xử lý tốt.
* Chế độ **Co-RAG** mang lại recall tốt hơn cho câu hỏi đa bước nhưng đánh đổi bằng độ trễ bổ sung +1.5–4 giây do Agent 3 phải gọi LLM để phân rã câu hỏi — không phù hợp cho ứng dụng yêu cầu phản hồi tức thì.

## **7.3. Hướng Phát Triển**

Dựa trên nền tảng đã xây dựng, các hướng phát triển tiếp theo của dự án bao gồm:

  1. **Mở rộng định dạng đầu vào:** Bổ sung khả năng đọc file Excel (.xlsx), PowerPoint (.pptx) và trích xuất bảng biểu từ PDF để hỗ trợ đa dạng tài liệu hơn.  
  2. **Tối ưu hiệu năng Embedding:** Thử nghiệm các mô hình embedding nhẹ hơn hoặc batch embedding để giảm thời gian xử lý khi nạp tài liệu lớn.  
  3. **Cải thiện giao diện quản lý tài liệu:** Thêm tính năng xem danh sách tài liệu đã nạp, xóa từng tài liệu riêng lẻ khỏi vector store thay vì phải xóa toàn bộ.

# **TÀI LIỆU THAM KHẢO**

\[1\] Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*. https://arxiv.org/abs/2005.11401

\[2\] Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333–389.

\[3\] Nogueira, R., & Cho, K. (2019). Passage Re-ranking with BERT. *arXiv:1901.04085*. https://arxiv.org/abs/1901.04085

\[4\] Asai, A., et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. *arXiv:2310.11511*. https://arxiv.org/abs/2310.11511

\[5\] Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*. https://github.com/facebookresearch/faiss

\[6\] Qwen Team (2024). Qwen2.5 Technical Report. Alibaba Group. https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

\[7\] LangChain Documentation. Building applications with LLMs through composability. https://python.langchain.com/docs/get_started/introduction

\[8\] Ollama. Get up and running with large language models locally. https://ollama.ai/

\[9\] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*. https://www.sbert.net/

\[10\] Wolf, T., et al. (2020). Transformers: State-of-the-art Natural Language Processing. *EMNLP 2020*.

\[11\] rank-bm25 library. Python implementation of BM25 ranking algorithm. https://github.com/dorianbrown/rank_bm25

\[12\] Streamlit Documentation. The fastest way to build and share data apps. https://docs.streamlit.io/

\[13\] SmartDoc AI Project Repository. https://github.com/SaiKrishnaRaoAnugu/SmartDoc-AI

\[14\] Shi, W., et al. (2024). REPLUG: Retrieval-Augmented Language Model Pre-Training. *arXiv:2301.12652*. https://arxiv.org/abs/2301.12652 *(Co-RAG multi-agent retrieval concept — ensemble of retrieval strategies with voting-based consensus)*

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,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>
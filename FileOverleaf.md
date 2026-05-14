\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts

% ===== PACKAGES =====
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{vntex}
\usepackage[english]{babel}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{float}
\usepackage{array}
\usepackage{tabularx}
\usepackage{multirow}
\usepackage{hyperref}
\usepackage{url}
\usepackage{cite}
\usepackage{listings}
\usepackage{microtype}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, shapes.geometric}

% ===== HYPERREF CONFIG =====
\hypersetup{
    colorlinks=true,
    linkcolor=black,
    citecolor=black,
    urlcolor=blue
}

% ===== CODE LISTING STYLE =====
\lstset{
    basicstyle=\footnotesize\ttfamily,
    breaklines=true,
    frame=single,
    numbers=left,
    numberstyle=\tiny,
    tabsize=2,
    showstringspaces=false,
    captionpos=b
}

% ===== TITLE =====
\title{SmartDoc AI: Hệ Thống Trợ Lý Đọc Hiểu Tài Liệu Thông Minh\\
Dựa Trên Kiến Trúc RAG Offline}

\author{
    \IEEEauthorblockN{
        Lê Nguyễn Nhất Tâm (3122410369),\;
        Lưu Hồng Phúc (3121410384),\\
        Đặng Quang Phong (3122410304),\;
        Nguyễn Phước Nam (3122410249)
    }
    \IEEEauthorblockA{
        \textit{Khoa Công nghệ Thông tin, Trường Đại học Sài Gòn}\\
        Lớp: DCT1231 \;|\; Năm học: 2025--2026\\
        Giảng viên hướng dẫn: Từ Lãng Phiêu
    }
}

\begin{document}

\maketitle

% ===== ABSTRACT =====
\begin{abstract}
Bài báo trình bày SmartDoc AI --- hệ thống hỏi-đáp tài liệu cục bộ dựa trên kiến trúc Retrieval-Augmented Generation (RAG) không yêu cầu kết nối Internet. Hệ thống tích hợp Hybrid Search (FAISS + BM25 với Reciprocal Rank Fusion), Cross-Encoder tái xếp hạng, Self-RAG ba tầng kiểm tra chất lượng, và Co-RAG ba tác nhân đồng thuận. Được triển khai trên CPU Intel Core i5-12400F (24\,GB RAM, không GPU), SmartDoc AI đạt độ trễ retrieval dưới 50\,ms với 5.000 chunks và cải thiện recall $\approx$16\% nhờ Co-RAG so với RAG đơn. Giao diện Streamlit hỗ trợ PDF và DOCX đa ngôn ngữ, cung cấp trích dẫn nguồn theo tên tệp và số trang.
\end{abstract}

\begin{IEEEkeywords}
RAG, FAISS, BM25, Hybrid Search, Self-RAG, Co-RAG, CrossEncoder, Qwen2.5, Streamlit
\end{IEEEkeywords}

% ===== SECTION I =====
\section{Giới Thiệu Và Công Trình Liên Quan}

Các mô hình ngôn ngữ lớn (LLM) gặp hạn chế về kiến thức tĩnh và thiếu khả năng truy xuất tri thức theo ngữ cảnh riêng tư. Lewis~et~al.~\cite{lewis2020rag} đề xuất RAG như một giải pháp: kết hợp retriever và generator để nâng cao chất lượng trả lời theo tài liệu gốc. Trong các ứng dụng doanh nghiệp và giáo dục, việc giữ tài liệu nội bộ không rời khỏi môi trường cục bộ là yêu cầu bắt buộc, thúc đẩy nhu cầu RAG offline.

Các nghiên cứu nền tảng bao gồm: BM25~\cite{robertson2009bm25} --- thuật toán tìm kiếm từ khóa xác suất hiệu quả; BERT reranking~\cite{nogueira2019reranking} --- tái xếp hạng kết quả retrieval bằng cross-attention; Self-RAG~\cite{asai2023selfrag} --- kiểm tra tự phê phán câu trả lời qua token phản ánh; FAISS~\cite{johnson2019faiss} --- thư viện tìm kiếm vector quy mô lớn của Facebook; và REPLUG/Co-RAG~\cite{shi2024replug} --- ensemble retriever với biểu quyết đồng thuận.

SmartDoc AI đóng góp: (1) pipeline RAG hoàn chỉnh offline không GPU; (2) Self-RAG ba tầng kết hợp với Co-RAG ba tác nhân; (3) Citation Tracking tên tệp và số trang cho cả PDF lẫn DOCX; (4) giao diện Streamlit thân thiện người dùng cuối.

% ===== SECTION II =====
\section{Cơ Sở Lý Thuyết}

\subsection{Mô hình RAG tổng quát}

Cho truy vấn $q$, retriever $R$ trả về tập ngữ cảnh $C = R(q) = \{c_1, c_2, \ldots, c_k\}$. Generator $G$ tạo câu trả lời:
\begin{equation}
    a = G(q, C)
\end{equation}

\subsection{BM25}

\begin{equation}
    \text{BM25}(D,Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i,D)\cdot(k_1+1)}{f(q_i,D)+k_1\!\left(1-b+b\frac{|D|}{\text{avgdl}}\right)}
\end{equation}

với $k_1=1.5$, $b=0.75$, $\text{avgdl}$ là độ dài tài liệu trung bình.

\subsection{Reciprocal Rank Fusion (RRF)}

\begin{equation}
    \text{RRF}(d) = \sum_{i} \frac{w_i}{k + \text{rank}_i(d)}, \quad k = 60
\end{equation}

với $w_{\text{FAISS}}=0.6$, $w_{\text{BM25}}=0.4$.

\subsection{Self-RAG ba tầng}

\begin{figure}[H]
\centering
\resizebox{0.78\columnwidth}{!}{%
\begin{tikzpicture}[
  rect/.style={rectangle, rounded corners=4pt, draw=black!70,
               text width=4.6cm, minimum height=0.9cm,
               align=center, font=\small},
  dia/.style={diamond, aspect=2.5, draw=black!70, fill=yellow!25,
              minimum width=3.6cm, minimum height=1.4cm,
              align=center, font=\small},
  arr/.style={-{Stealth[length=5pt]}, thick}
]
  \node[rect, fill=gray!15]   (q)   at (0,  0.0)  {Câu hỏi gốc $q$};
  \node[rect, fill=blue!15]   (t1)  at (0, -1.5)  {\textbf{Tầng 1 -- Query Expansion}\\LLM sinh $q_1,\;q_2,\;q_3$};
  \node[rect, fill=cyan!15]   (ret) at (0, -3.0)  {Retrieve: FAISS + BM25\\top-$k$ documents};
  \node[dia]                  (d2)  at (0, -4.75) {\textbf{Tầng 2}\\Liên quan?};
  \node[rect, fill=orange!20] (fil) at (0, -6.5)  {Lọc: giữ doc CÓ liên quan\\loại bỏ KHÔNG liên quan};
  \node[rect, fill=green!20]  (gen) at (0, -8.0)  {Generate: Qwen2.5:7b\\+ context đã lọc};
  \node[dia]                  (d3)  at (0, -9.75) {\textbf{Tầng 3}\\Hallucination?};
  \node[rect, fill=gray!15]   (ans) at (0,-11.5)  {Câu trả lời + confidence\\$\in[0.0,\;1.0]$};
  \draw[arr] (q)   -- (t1);
  \draw[arr] (t1)  -- (ret);
  \draw[arr] (ret) -- (d2);
  \draw[arr] (d2)  -- node[right, font=\footnotesize]{CÓ} (fil);
  \draw[arr, dashed] (d2.west) -- ++(-1.4,0)
    node[left, font=\footnotesize]{KHÔNG: loại};
  \draw[arr] (fil) -- (gen);
  \draw[arr] (gen) -- (d3);
  \draw[arr] (d3)  -- node[right, font=\footnotesize]{Đạt} (ans);
  \draw[arr, dashed] (d3.east) -- ++(1.4,0)
    node[right, font=\footnotesize]{Retry};
\end{tikzpicture}%
}
\caption{Kiến trúc Self-RAG ba tầng: Query Expansion, Relevance Grading và Answer Grading}
\label{fig:selfrag}
\end{figure}

\begin{enumerate}
    \item \textbf{Tầng 1 -- Query Expansion:} LLM sinh 3 phiên bản câu hỏi từ các góc độ khác nhau.
    \item \textbf{Tầng 2 -- Relevance Grading:} Đánh giá từng tài liệu truy xuất (CÓ/KHÔNG liên quan).
    \item \textbf{Tầng 3 -- Answer Grading:} Kiểm tra câu trả lời có hallucination không; trả về điểm confidence $[0.0, 1.0]$.
\end{enumerate}

\subsection{Co-RAG --- Công thức đồng thuận}

\begin{equation}
    \text{score\_final}(c) = \bar{s}(c) \cdot \left(1 + \alpha \cdot \frac{v(c)}{N}\right)
\end{equation}

trong đó $\bar{s}$ là điểm trung bình từ các tác nhân đồng thuận, $v$ là số tác nhân tìm thấy chunk đó. Ngưỡng chấp nhận: \texttt{CO\_RAG\_MIN\_VOTES = 2}.

% ===== SECTION III =====
\section{Kiến Trúc Hệ Thống}

\subsection{Pipeline tổng thể}

Hệ thống SmartDoc AI được thiết kế theo mô hình kiến trúc đa tầng (multi-layer architecture) gồm 4 tầng chính: Presentation Layer (Streamlit), Application Layer (LangChain), Data Layer (FAISS + BM25), và Model Layer (Ollama).

\begin{figure}[H]
\centering
\resizebox{\columnwidth}{!}{%
\begin{tikzpicture}[
  b/.style={rectangle, rounded corners=3pt, draw=black!70,
            minimum width=3.4cm, minimum height=1.0cm,
            align=center, font=\small},
  arr/.style={-{Stealth[length=5pt]}, thick}
]
  % ── Luồng Indexing (left column) ──
  \node[font=\small\bfseries, text=black!50] at (-4.0, 1.0) {Luồng Indexing};
  \node[b, fill=gray!15]   (up)   at (-4.0,  0.0) {Upload\\PDF / DOCX};
  \node[b, fill=yellow!20] (proc) at (-4.0, -1.8) {Chunk + Embed\\1500c / mpnet-768};
  \node[b, fill=orange!20] (idx)  at (-4.0, -3.6) {FAISS HNSW\\+ BM25 Index};
  \draw[arr] (up)   -- (proc);
  \draw[arr] (proc) -- (idx);
  % ── Luồng Truy vấn (right column) ──
  \node[font=\small\bfseries, text=black!50] at (4.0, 1.0) {Luồng Truy vấn};
  \node[b, fill=gray!15]   (user)   at (4.0,  0.0) {Câu hỏi người dùng};
  \node[b, fill=blue!15]   (qexp)   at (4.0, -1.8) {Query Expansion\\Self-RAG Tầng 1};
  \node[b, fill=blue!25]   (hybrid) at (4.0, -3.6) {Hybrid Search\\FAISS+BM25+RRF};
  \node[b, fill=blue!10]   (rerank) at (4.0, -5.4) {CrossEncoder\\Reranking (top-3)};
  \node[b, fill=purple!20] (rag)    at (4.0, -7.2) {Self-RAG T2+T3\\+ Co-RAG};
  \node[b, fill=green!25]  (llm)    at (4.0, -9.0) {Qwen2.5:7b\\Ollama (CPU)};
  \node[b, fill=teal!20]   (out)    at (4.0,-10.8) {Câu trả lời\\+ Trích dẫn};
  \draw[arr] (user)   -- (qexp);
  \draw[arr] (qexp)   -- (hybrid);
  \draw[arr] (hybrid) -- (rerank);
  \draw[arr] (rerank) -- (rag);
  \draw[arr] (rag)    -- (llm);
  \draw[arr] (llm)    -- (out);
  % ── Index → Hybrid Search ──
  \draw[arr, dashed] (idx.east) -- node[above, font=\footnotesize]{Index} (hybrid.west);
\end{tikzpicture}%
}
\caption{Pipeline tổng thể SmartDoc AI: từ upload tài liệu đến sinh câu trả lời có trích dẫn nguồn}
\label{fig:pipeline}
\end{figure}

\subsection{Thư viện và phiên bản}

\begin{table}[H]
\caption{Thư viện sử dụng}
\label{tab:libraries}
\centering
\begin{tabular}{lll}
\toprule
\textbf{Thư viện} & \textbf{Phiên bản} & \textbf{Vai trò} \\
\midrule
Python & $\geq$3.8 & Nền tảng thực thi \\
Ollama & latest & Vận hành LLM cục bộ \\
LangChain & $\geq$0.2.0 & Điều phối pipeline RAG \\
faiss-cpu & $\geq$1.7.4 & Vector store \\
rank-bm25 & $\geq$0.2.2 & Tìm kiếm từ khóa \\
pdfplumber & $\geq$0.11.0 & Trích xuất PDF \\
python-docx & $\geq$1.1.0 & Trích xuất DOCX \\
sentence-transformers & $\geq$2.2.0 & Embedding đa ngôn ngữ \\
Streamlit & $\geq$1.30.0 & Giao diện web \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Cấu hình chính}

\begin{itemize}
    \item \textbf{Embedding model:} \texttt{paraphrase-multilingual-mpnet-base-v2} (768 chiều)
    \item \textbf{Reranking model:} \texttt{cross-encoder/ms-marco-MiniLM-L-6-v2}
    \item \textbf{LLM:} \texttt{qwen2.5:7b} qua Ollama (CPU)
    \item \textbf{Chunk size tối ưu:} 1500 ký tự, overlap 200 ký tự ($\approx$610 chunks)
    \item \textbf{Top-k retrieval:} 3 chunks sau reranking
    \item \textbf{Hardware:} Intel Core i5-12400F, 24\,GB RAM, Windows 11, không GPU
\end{itemize}

% ===== SECTION IV =====
\section{Thực Nghiệm Và Kết Quả}

\subsection{Hiệu năng Retrieval}

\begin{table}[H]
\caption{Độ trễ retrieval (ms) theo số lượng chunks}
\label{tab:retrieval-latency}
\centering
\begin{tabular}{cccc}
\toprule
\textbf{Số chunks} & \textbf{FAISS (ms)} & \textbf{BM25 (ms)} & \textbf{Hybrid (ms)} \\
\midrule
1.000  & 35,84 & 1,48  & 37,95 \\
3.000  & 35,63 & 5,37  & 43,23 \\
5.000  & 36,69 & 8,81  & 47,96 \\
10.000 & 37,20 & 21,89 & 60,52 \\
\bottomrule
\end{tabular}
\end{table}

Nhận xét: FAISS duy trì độ trễ gần như hằng định ($\approx$35--37\,ms) nhờ tìm kiếm xấp xỉ HNSW, trong khi BM25 tuyến tính theo kích thước corpus. Hybrid Search giữ dưới 50\,ms tới 5.000 chunks --- ngưỡng thực tế của hệ thống.

\begin{figure}[H]
    \centering
    \includegraphics[width=\columnwidth]{hinh1.png}
    \caption{So sánh độ trễ retrieval — FAISS vs BM25 vs Hybrid Search}
    \label{fig:retrieval}
\end{figure}

\subsection{Self-RAG Overhead}

\begin{table}[H]
\caption{Overhead của Self-RAG ba tầng}
\label{tab:selfrag-overhead}
\centering
\begin{tabular}{lccc}
\toprule
\textbf{Kịch bản} & \textbf{RAG chuẩn} & \textbf{+Self-RAG} & \textbf{Overhead} \\
\midrule
Câu hỏi đơn giản  & 2,6\,s & 5,3\,s  & +2,7\,s \\
Câu hỏi trung bình & 6,6\,s & 8,2\,s  & +1,6\,s \\
Câu hỏi phức tạp  & 8,3\,s & 12,1\,s & +3,8\,s \\
\bottomrule
\end{tabular}
\end{table}

Self-RAG phân rã overhead theo ba tầng: Tầng~1 Query Expansion (0,8--1,5\,s), Tầng~2 Relevance Grading (0,2--1,0\,s $\times$ số docs), Tầng~3 Answer Grading (1,3--1,7\,s).

Ba kịch bản được phân loại theo độ phức tạp ngữ nghĩa tăng dần:
\begin{itemize}
    \item \textbf{Câu hỏi đơn giản:} Tra cứu một sự kiện từ một đoạn duy nhất --- ví dụ: \textit{``FAISS là gì?''}
    \item \textbf{Câu hỏi trung bình:} Tổng hợp thông tin từ 2--3 đoạn tài liệu --- ví dụ: \textit{``So sánh FAISS và BM25.''}
    \item \textbf{Câu hỏi phức tạp:} Lý luận đa bước qua toàn bộ corpus --- ví dụ: \textit{``Tại sao Hybrid Search kết hợp với Self-RAG cho kết quả tốt hơn?''}
\end{itemize}

\begin{figure}[H]
    \centering
    \includegraphics[width=\columnwidth]{hinh2a.png}
    \caption{So sánh thời gian phản hồi: RAG chuẩn vs RAG + Self-RAG}
    \label{fig:selfrag-time}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=\columnwidth]{hinh2b.png}
    \caption{Phân tích overhead Self-RAG theo từng tầng}
    \label{fig:selfrag-overhead}
\end{figure}

\subsection{Tối ưu hoá Chunking}

\begin{table}[H]
\caption{So sánh chiến lược chunking}
\label{tab:chunking}
\centering
\begin{tabular}{cccp{2.8cm}}
\toprule
\textbf{Chunk size} & \textbf{Overlap} & \textbf{Số chunks} & \textbf{Ghi chú} \\
\midrule
500  & 50  & $\approx$1.840 & Context quá ngắn \\
1.000 & 100 & $\approx$920  & Phù hợp văn xuôi \\
\textbf{1.500} & \textbf{200} & $\approx$\textbf{610} & \textbf{Tối ưu } \\
2.000 & 200 & $\approx$460  & Chunk đa chủ đề \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Kiểm thử}

\begin{table}[H]
\caption{Kết quả kiểm thử hệ thống}
\label{tab:testing}
\centering
\begin{tabular}{lccc}
\toprule
\textbf{Module} & \textbf{Unit} & \textbf{Integration} & \textbf{Kết quả} \\
\midrule
document\_processor & 18 & 4 & \checkmark \\
language\_detector  & 12 & 3 & \checkmark \\
vector\_store       & 22 & 5 & \checkmark \\
rag\_chain          & 25 & 6 & \checkmark \\
co\_rag             & 15 & 4 & \checkmark \\
self\_rag           & 14 & 3 & \checkmark \\
citation\_tracker   &  8 & 2 & \checkmark \\
\midrule
\textbf{Tổng}       & \textbf{114} & \textbf{27} & \textbf{141 } \\
\bottomrule
\end{tabular}
\end{table}

% ===== SECTION V =====
\section{So Sánh Co-RAG Với RAG Đơn}

\begin{table}[H]
\caption{Co-RAG vs.\ RAG Single Agent}
\label{tab:corag}
\centering
\begin{tabular}{lll}
\toprule
\textbf{Tiêu chí} & \textbf{RAG Single} & \textbf{Co-RAG 3 Agents} \\
\midrule
Chiến lược retrieval & 1 retriever & Semantic + Keyword \\
                     &             & + Conceptual \\
Recall trung bình    & $\approx$72\% & $\approx$88\% \\
Độ trễ bổ sung       & 0\,ms & +1.500--4.000\,ms \\
Ngưỡng đồng thuận   & N/A & \texttt{MIN\_VOTES = 2} \\
\bottomrule
\end{tabular}
\end{table}

Co-RAG cải thiện recall $\approx$16\% (từ 72\% lên 88\%) cho các câu hỏi đa bước, đặc biệt khi một agent riêng lẻ bỏ sót tài liệu. Agent Conceptual (phân rã câu hỏi thành $\leq$3 sub-questions bằng LLM) bù đắp hiệu quả cho các trường hợp FAISS và BM25 đều thất bại.

% ===== SECTION VI =====
\section{Phân Tích Độ Phức Tạp}

\begin{table}[H]
\caption{Độ phức tạp tính toán các thành phần}
\label{tab:complexity}
\centering
\begin{tabular}{lll}
\toprule
\textbf{Thành phần} & \textbf{Index / Build} & \textbf{Query} \\
\midrule
FAISS (HNSW)  & $O(d \cdot n)$   & $O(d \cdot \log n)$ xấp xỉ \\
BM25          & $O(n \cdot L)$   & $O(n)$ \\
CrossEncoder  & ---              & $O(n \cdot L^2)$ \\
Self-RAG      & ---              & $O(3 \cdot T_{\text{LLM}})$ \\
Co-RAG        & ---              & $O(3 \cdot T_{\text{retrieval}}) + T_{\text{LLM}}$ \\
\bottomrule
\end{tabular}
\end{table}

Trong đó $d=768$ (chiều embedding), $n$ = số chunks, $L$ = độ dài token, $T_{\text{LLM}}$ = thời gian một LLM call. Bottleneck chính là Self-RAG (3 LLM calls bổ sung) và Co-RAG Agent Conceptual (1 LLM call phân rã + 3 retrievals).

% ===== SECTION VII =====
\section{Kết Luận}

\subsection{Tóm tắt đóng góp}

SmartDoc AI triển khai thành công hệ thống RAG offline hoàn chỉnh trên CPU thông dụng. Các đóng góp kỹ thuật chính:

\begin{enumerate}
    \item \textbf{Hybrid Search} (FAISS~0.6 + BM25~0.4 qua RRF): Kết hợp điểm mạnh của tìm kiếm ngữ nghĩa và từ khóa; độ trễ $<50$\,ms tới 5.000 chunks.
    \item \textbf{CrossEncoder Reranking} (\texttt{ms-marco-MiniLM-L-6-v2}): Tái xếp hạng chính xác hơn bi-encoder.
    \item \textbf{Self-RAG ba tầng:} Query Expansion + Relevance Grading + Answer Grading --- cơ chế tự kiểm tra chất lượng không cần ground truth.
    \item \textbf{Co-RAG ba tác nhân:} Semantic + Keyword + Conceptual với Vote Boost; cải thiện recall $\approx$16\% trên truy vấn đa bước.
    \item \textbf{Citation Tracking:} Trích dẫn tên tệp và số trang chính xác cho cả PDF lẫn DOCX.
\end{enumerate}

\subsection{Hạn chế}

\begin{itemize}
    \item \textbf{CPU bottleneck:} Không GPU dẫn đến thời gian LLM call 2--8\,s/turn --- chưa đạt real-time.
    \item \textbf{FAISS scalability:} Lưu toàn bộ index trong RAM; với hàng triệu chunks cần chuyển sang FAISS IVF hoặc vector database phân tán.
    \item \textbf{Định dạng hạn chế:} Chỉ hỗ trợ PDF và DOCX; chưa xử lý Excel, PowerPoint, hay bảng phức tạp trong PDF.
    \item \textbf{Co-RAG latency:} +1,5--4\,s không phù hợp cho ứng dụng real-time nghiêm ngặt.
\end{itemize}

% ===== SECTION VIII =====
\section{Hướng Phát Triển}

\begin{enumerate}
    \item \textbf{Mở rộng định dạng đầu vào:} Thêm hỗ trợ Excel (.xlsx), PowerPoint (.pptx) và trích xuất bảng phức tạp từ PDF (bounding-box + OCR).
    \item \textbf{Tối ưu hoá embedding:} Thử nghiệm các model nhẹ hơn (MiniLM-L6, BGE-M3) hoặc batch embedding để giảm thời gian index cho corpus lớn.
    \item \textbf{Cải thiện giao diện quản lý:} Cho phép xem, xoá từng tài liệu riêng lẻ trong vector store; theo dõi lịch sử hội thoại đa phiên; dashboard thống kê truy vấn.
\end{enumerate}

% ===== REFERENCES =====
\begin{thebibliography}{00}

\bibitem{lewis2020rag}
P. Lewis et al., ``Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,'' \textit{NeurIPS 2020}, Dec. 2020. [Online]. Available: \url{https://arxiv.org/abs/2005.11401}

\bibitem{robertson2009bm25}
S. Robertson and H. Zaragoza, ``The Probabilistic Relevance Framework: BM25 and Beyond,'' \textit{Foundations and Trends in Information Retrieval}, vol.~3, no.~4, pp.~333--389, 2009.

\bibitem{nogueira2019reranking}
R. Nogueira and K. Cho, ``Passage Re-ranking with BERT,'' \textit{arXiv:1901.04085}, 2019. [Online]. Available: \url{https://arxiv.org/abs/1901.04085}

\bibitem{asai2023selfrag}
A. Asai et al., ``Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection,'' \textit{arXiv:2310.11511}, Oct. 2023. [Online]. Available: \url{https://arxiv.org/abs/2310.11511}

\bibitem{johnson2019faiss}
J. Johnson, M. Douze, and H. Jégou, ``Billion-Scale Similarity Search with GPUs,'' \textit{IEEE Transactions on Big Data}, 2019. [Online]. Available: \url{https://github.com/facebookresearch/faiss}

\bibitem{qwen2024}
Qwen Team, ``Qwen2.5 Technical Report,'' Alibaba Group, 2024. [Online]. Available: \url{https://huggingface.co/Qwen/Qwen2.5-7B-Instruct}

\bibitem{langchain}
LangChain, ``Building applications with LLMs through composability.'' [Online]. Available: \url{https://python.langchain.com/docs/get_started/introduction}

\bibitem{ollama}
Ollama, ``Get up and running with large language models locally.'' [Online]. Available: \url{https://ollama.ai/}

\bibitem{reimers2019sbert}
N. Reimers and I. Gurevych, ``Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,'' \textit{EMNLP 2019}. [Online]. Available: \url{https://www.sbert.net/}

\bibitem{wolf2020transformers}
T. Wolf et al., ``Transformers: State-of-the-Art Natural Language Processing,'' \textit{EMNLP 2020}.

\bibitem{rankbm25}
rank-bm25, ``Python implementation of BM25 ranking algorithm.'' [Online]. Available: \url{https://github.com/dorianbrown/rank_bm25}

\bibitem{streamlit}
Streamlit, ``The fastest way to build and share data apps.'' [Online]. Available: \url{https://docs.streamlit.io/}

\bibitem{smartdoc}
SmartDoc AI Project Repository. [Online]. Available: \url{https://github.com/SaiKrishnaRaoAnugu/SmartDoc-AI}

\bibitem{shi2024replug}
W. Shi et al., ``REPLUG: Retrieval-Augmented Language Model Pre-Training,'' \textit{arXiv:2301.12652}, 2024. [Online]. Available: \url{https://arxiv.org/abs/2301.12652}

\end{thebibliography}

\end{document}

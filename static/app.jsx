const { useState, useEffect, useRef, useCallback } = React;

// ── SVG Icons Object (Offline-friendly) ──
const Icons = {
    Brand: () => (
        <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
        </svg>
    ),
    Refresh: () => (
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
        </svg>
    ),
    Trash: () => (
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="3 6 5 6 21 6"/>
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
            <line x1="10" y1="11" x2="10" y2="17"/>
            <line x1="14" y1="11" x2="14" y2="17"/>
        </svg>
    ),
    File: () => (
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
        </svg>
    ),
    Page: () => (
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
            <line x1="9" y1="9" x2="15" y2="9"/>
            <line x1="9" y1="13" x2="15" y2="13"/>
            <line x1="9" y1="17" x2="13" y2="17"/>
        </svg>
    ),
    Chunk: () => (
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
            <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
            <line x1="12" y1="22.08" x2="12" y2="12"/>
        </svg>
    ),
    Settings: () => (
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3"/>
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
        </svg>
    ),
    ChevronUp: () => (
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="18 15 12 9 6 15"/>
        </svg>
    ),
    ChevronDown: () => (
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="6 9 12 15 18 9"/>
        </svg>
    ),
    Send: () => (
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"/>
            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
        </svg>
    ),
    Sun: () => (
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="5"/>
            <line x1="12" y1="1" x2="12" y2="3"/>
            <line x1="12" y1="21" x2="12" y2="23"/>
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
            <line x1="1" y1="12" x2="3" y2="12"/>
            <line x1="21" y1="12" x2="23" y2="12"/>
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
        </svg>
    ),
    Moon: () => (
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
        </svg>
    ),
    CloudUpload: () => (
        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21.2 15c.7-1.2 1-2.5.7-3.9-.3-2-1.9-3.6-3.9-3.9C16.7 4.1 13.5 2 10 3 7.4 3.7 5.3 5.8 4.6 8.4c-2 .3-3.6 1.9-3.9 3.9-.3 2.1.8 4 2.6 4.9"/>
            <polyline points="16 12 12 8 8 12"/>
            <line x1="12" y1="8" x2="12" y2="21"/>
        </svg>
    )
};

// ── Text Highlighter Utility (Exact app.py Logic) ──
function highlightText(text, query, answer) {
    if (!text) return "";
    
    const escapeRegex = (string) => string.replace(/[/\-\\^$*+?.()|[\]{}]/g, '\\$&');
    
    let candidates = [];
    
    // Layer 1: N-grams from assistant answer (7 to 4 words)
    if (answer) {
        // Clear markdown syntax
        const cleanAnswer = answer.replace(/[\*\_`\#\>\|]/g, ' ');
        const answerWords = cleanAnswer.trim().split(/\s+/);
        for (let n = 7; n >= 4; n--) {
            for (let i = 0; i <= answerWords.length - n; i++) {
                let phrase = answerWords.slice(i, i + n).join(' ');
                phrase = phrase.replace(/[.,;:!?()]+$/, '').replace(/^[.,;:!?()]+/, '').trim();
                if (phrase.length < 10) continue;
                
                const regex = new RegExp(escapeRegex(phrase), 'i');
                if (regex.test(text)) {
                    candidates.push({ phrase, layer: 1 });
                }
            }
        }
    }
    
    // Layer 2: Keywords from user question (word length >= 3)
    if (query) {
        const queryTokens = query.split(/[\s\W]+/).filter(t => t.length >= 3);
        for (const token of queryTokens) {
            candidates.push({ phrase: token, layer: 2 });
        }
    }
    
    if (candidates.length === 0) {
        return escapeHtml(text);
    }
    
    // Sort: layer 1 first, then longer phrases first
    candidates.sort((a, b) => {
        if (a.layer !== b.layer) return a.layer - b.layer;
        return b.phrase.length - a.phrase.length;
    });
    
    // Filter out substrings of already matched phrases
    const uniquePhrases = [];
    const seenLower = new Set();
    for (const cand of candidates) {
        const pl = cand.phrase.toLowerCase();
        let isSubPhrase = false;
        for (const seen of seenLower) {
            if (seen.includes(pl)) {
                isSubPhrase = true;
                break;
            }
        }
        if (!isSubPhrase) {
            uniquePhrases.push(cand);
            seenLower.add(pl);
        }
    }
    
    // Temporary marked string
    let markedText = text;
    uniquePhrases.forEach((p, idx) => {
        // Use word boundaries for keywords, standard regex for phrases
        const pattern = p.layer === 2 ? `\\b${escapeRegex(p.phrase)}\\b` : escapeRegex(p.phrase);
        const regex = new RegExp(pattern, 'gi');
        markedText = markedText.replace(regex, (match) => {
            return `__MARK_START_${p.layer}_${idx}__${match}__MARK_END_${p.layer}_${idx}__`;
        });
    });
    
    function escapeHtml(unsafe) {
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }
    
    let escapedText = escapeHtml(markedText);
    
    // Convert temporary markers into HTML mark tags
    uniquePhrases.forEach((p, idx) => {
        const startRegex = new RegExp(`__MARK_START_${p.layer}_${idx}__`, 'g');
        const endRegex = new RegExp(`__MARK_END_${p.layer}_${idx}__`, 'g');
        const className = p.layer === 1 ? 'answer-highlight' : 'query-highlight';
        
        escapedText = escapedText
            .replace(startRegex, `<mark class="${className}">`)
            .replace(endRegex, `</mark>`);
    });
    
    return escapedText;
}

// ── Simple Markdown Renderer (For response bubbles) ──
function renderMarkdown(text) {
    if (!text) return "";
    
    let html = text;
    // Escape HTML tags to prevent XSS
    html = html
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    // Headings
    html = html.replace(/^### (.*$)/gim, '<h4 style="margin: 8px 0 4px; font-weight: 700; font-size: 14px;">$1</h4>');
    html = html.replace(/^## (.*$)/gim, '<h3 style="margin: 12px 0 6px; font-weight: 800; font-size: 15px;">$1</h3>');
    html = html.replace(/^# (.*$)/gim, '<h2 style="margin: 16px 0 8px; font-weight: 800; font-size: 17px;">$1</h2>');
    
    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Italic
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Code block
    html = html.replace(/```([\s\S]*?)```/g, '<pre style="background:var(--bg-surface); padding:10px; border-radius: var(--radius-sm); font-family: var(--font-mono); font-size: 12px; margin: 8px 0; overflow-x: auto;"><code>$1</code></pre>');
    
    // Inline code
    html = html.replace(/`(.*?)`/g, '<code style="background:var(--bg-surface); padding: 2px 6px; border-radius: var(--radius-sm); font-family: var(--font-mono); font-size: 12px;">$1</code>');
    
    // Line breaks
    html = html.replace(/\n/g, '<br />');
    
    // Bullet points
    html = html.replace(/^\s*-\s+(.*$)/gim, '<li style="margin-left: 16px; list-style-type: disc;">$1</li>');
    
    return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

// ── Main App Component ──
function App() {
    // Theme State
    const [theme, setTheme] = useState('dark');
    
    // System & Data State
    const [ollamaStatus, setOllamaStatus] = useState(null);
    const [statusData, setStatusData] = useState({ ollama_model: "", embedding_model: "" });
    const [kpis, setKpis] = useState({ total_files: 0, total_pages: 0, total_chunks: 0 });
    const [processedFiles, setProcessedFiles] = useState([]);
    const [chatHistory, setChatHistory] = useState([]);
    
    // UI Loading States
    const [isProcessing, setIsProcessing] = useState(false);
    const [isThinking, setIsThinking] = useState(false);
    const [chatInput, setChatInput] = useState("");
    
    // Settings State
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);
    const [config, setConfig] = useState({
        chunk_size: 1000,
        chunk_overlap: 200,
        hybrid_enabled: false,
        reranker_enabled: false,
        self_rag_enabled: false,
        self_rag_query_rewrite: true,
        self_rag_relevance_filter: true,
        self_rag_answer_grading: true,
        co_rag_enabled: false,
        co_rag_agent_semantic: true,
        co_rag_agent_keyword: true,
        co_rag_agent_conceptual: true,
        co_rag_merge_strategy: "voting",
        active_file_filter: []
    });

    // Dialog Modals State
    const [showClearHistory, setShowClearHistory] = useState(false);
    const [showClearDocs, setShowClearDocs] = useState(false);
    const [selectedHistory, setSelectedHistory] = useState(null);
    
    // Detail citation highlight state
    // Maps messageIndex -> activeSourceIndex
    const [activeCitations, setActiveCitations] = useState({});

    // Scroll refs
    const messagesEndRef = useRef(null);

    // Set Theme Attribute on body
    useEffect(() => {
        document.documentElement.setAttribute('data-theme', theme);
    }, [theme]);

    // Initial Load & Status Poll
    const fetchStatus = useCallback(async () => {
        try {
            const res = await fetch('/api/status');
            if (res.ok) {
                const data = await res.json();
                setOllamaStatus(data.ollama_status);
                setStatusData({
                    ollama_model: data.ollama_model,
                    embedding_model: data.embedding_model
                });
                setKpis({
                    total_files: data.total_files,
                    total_pages: data.total_pages,
                    total_chunks: data.total_chunks
                });
            }
        } catch (e) {
            console.error("Error fetching status:", e);
            setOllamaStatus(false);
        }
    }, []);

    const fetchConfig = useCallback(async () => {
        try {
            const res = await fetch('/api/config');
            if (res.ok) {
                const data = await res.json();
                setConfig(data);
            }
        } catch (e) {
            console.error("Error fetching config:", e);
        }
    }, []);

    const fetchFiles = useCallback(async () => {
        try {
            const res = await fetch('/api/files');
            if (res.ok) {
                const data = await res.json();
                setProcessedFiles(data);
            }
        } catch (e) {
            console.error("Error fetching files:", e);
        }
    }, []);

    const fetchHistory = useCallback(async () => {
        try {
            const res = await fetch('/api/history');
            if (res.ok) {
                const data = await res.json();
                setChatHistory(data);
            }
        } catch (e) {
            console.error("Error fetching history:", e);
        }
    }, []);

    useEffect(() => {
        fetchStatus();
        fetchConfig();
        fetchFiles();
        fetchHistory();
        
        // Poll status every 15s
        const interval = setInterval(fetchStatus, 15000);
        return () => clearInterval(interval);
    }, [fetchStatus, fetchConfig, fetchFiles, fetchHistory]);

    // Auto-scroll to bottom of chat
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [chatHistory, isThinking]);

    // Save Configuration change
    const updateRemoteConfig = async (updatedConfig) => {
        try {
            const res = await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updatedConfig)
            });
            if (res.ok) {
                const data = await res.json();
                setConfig(data.config);
            }
        } catch (e) {
            console.error("Error updating config:", e);
        }
    };

    const handleConfigChange = (key, value) => {
        const updated = { ...config, [key]: value };
        setConfig(updated);
        updateRemoteConfig(updated);
    };

    // File Upload Handler
    const handleFileUpload = async (e) => {
        const files = e.target.files;
        if (!files || files.length === 0) return;

        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }

        setIsProcessing(true);
        try {
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            if (res.ok) {
                await fetchFiles();
                await fetchStatus();
            } else {
                alert("Lỗi khi xử lý tải tài liệu.");
            }
        } catch (err) {
            console.error("Upload error:", err);
            alert("Đã xảy ra lỗi khi kết nối tới server.");
        } finally {
            setIsProcessing(false);
            e.target.value = ""; // clear input
        }
    };

    // Chat Message Submission
    const handleSendMessage = async (e) => {
        if (e) e.preventDefault();
        if (!chatInput.trim() || isThinking) return;

        const messageText = chatInput;
        setChatInput("");
        setIsThinking(true);

        // Optimistically add user message to local UI chat state
        setChatHistory(prev => [...prev, { role: "user", content: messageText, timestamp: Date.now() / 1000 }]);

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: messageText })
            });

            if (res.ok) {
                const answerMessage = await res.json();
                setChatHistory(prev => {
                    // Replace the optimistic flow or just fetch full history from disk to ensure consistency
                    // Fetching history is safer and syncs everything perfectly
                    return [...prev.filter(m => m.timestamp), answerMessage];
                });
                await fetchStatus(); // update statistics (KPIs/chunks)
            } else {
                // Add error message to state
                setChatHistory(prev => [...prev, {
                    role: "assistant",
                    content: "❌ Không thể kết nối tới mô hình AI. Hãy đảm bảo Ollama đang chạy.",
                    error: "Lỗi kết nối API",
                    timestamp: Date.now() / 1000
                }]);
            }
        } catch (err) {
            console.error("Chat error:", err);
            setChatHistory(prev => [...prev, {
                role: "assistant",
                content: "❌ Lỗi hệ thống khi gửi câu hỏi.",
                error: err.toString(),
                timestamp: Date.now() / 1000
            }]);
        } finally {
            setIsThinking(false);
        }
    };

    // Action Triggers
    const handleClearHistory = async () => {
        try {
            const res = await fetch('/api/clear-history', { method: 'POST' });
            if (res.ok) {
                setChatHistory([]);
                setShowClearHistory(false);
            }
        } catch (e) {
            console.error(e);
        }
    };

    const handleClearDocuments = async () => {
        try {
            const res = await fetch('/api/clear-documents', { method: 'POST' });
            if (res.ok) {
                setChatHistory([]);
                setProcessedFiles([]);
                setKpis({ total_files: 0, total_pages: 0, total_chunks: 0 });
                setConfig(prev => ({ ...prev, active_file_filter: [] }));
                setShowClearDocs(false);
                await fetchConfig();
            }
        } catch (e) {
            console.error(e);
        }
    };

    // Text Citation Pill click handler
    const handleCitationClick = (msgIdx, srcIdx) => {
        setActiveCitations(prev => {
            const key = msgIdx.toString();
            if (prev[key] === srcIdx) {
                // toggle collapse if clicking same pill
                const copy = { ...prev };
                delete copy[key];
                return copy;
            }
            return { ...prev, [key]: srcIdx };
        });
    };

    // Grouping user-assistant pairs for history sidebar
    const getHistoryPairs = () => {
        const pairs = [];
        for (let i = 0; i < chatHistory.length; i++) {
            if (chatHistory[i].role === 'user') {
                const question = chatHistory[i].content;
                let answer = "";
                let sources = [];
                let selfRagMeta = null;
                let coRagMeta = null;
                if (i + 1 < chatHistory.length && chatHistory[i+1].role === 'assistant') {
                    answer = chatHistory[i+1].content;
                    sources = chatHistory[i+1].sources || [];
                    selfRagMeta = chatHistory[i+1].self_rag_meta;
                    coRagMeta = chatHistory[i+1].co_rag_meta;
                }
                pairs.push({ question, answer, sources, selfRagMeta, coRagMeta, index: i });
            }
        }
        return pairs;
    };

    const historyPairs = getHistoryPairs();

    return (
        <div className="app-container">
            {/* ── Sidebar ── */}
            <aside className="sidebar">
                <div className="sidebar-header">
                    <div className="brand">
                        <div className="brand-logo">
                            <Icons.Brand />
                        </div>
                        <div className="brand-text">
                            <h1>SmartDocAI</h1>
                            <p>Trợ lý Tài liệu Thông minh</p>
                        </div>
                    </div>
                </div>

                <div className="sidebar-content">
                    {/* Ollama Status */}
                    <div className="sidebar-section">
                        <span className="section-label">Trạng thái hệ thống</span>
                        <div className="status-card">
                            <div className="status-indicator">
                                <div className={`status-dot ${ollamaStatus ? 'online' : 'offline'}`}></div>
                                <span>
                                    {ollamaStatus 
                                        ? `Ollama: ${statusData.ollama_model || 'online'}` 
                                        : 'Ollama không khả dụng'
                                    }
                                </span>
                            </div>
                            <button className="btn-icon-sm" onClick={fetchStatus} title="Kiểm tra lại">
                                <Icons.Refresh />
                            </button>
                        </div>
                        {!ollamaStatus && ollamaStatus !== null && (
                            <div style={{ fontSize: '11px', color: 'var(--error)', padding: '4px 6px', lineHeight: '1.4' }}>
                                Hãy chắc chắn Ollama đang chạy trên port 11434 và model <strong>{statusData.ollama_model || 'qwen2.5:7b'}</strong> đã được tải về.
                            </div>
                        )}
                    </div>

                    {/* KPI Metrics */}
                    <div className="sidebar-section">
                        <span className="section-label">Dữ liệu hiện tại</span>
                        <div className="kpi-grid">
                            <div className="kpi-card">
                                <div className="kpi-num">{kpis.total_files}</div>
                                <div className="kpi-label">File</div>
                            </div>
                            <div className="kpi-card">
                                <div className="kpi-num">{kpis.total_pages}</div>
                                <div className="kpi-label">Trang</div>
                            </div>
                            <div className="kpi-card">
                                <div className="kpi-num">{kpis.total_chunks}</div>
                                <div className="kpi-label">Chunks</div>
                            </div>
                        </div>
                    </div>

                    {/* Drag and Drop Uploader */}
                    <div className="sidebar-section">
                        <span className="section-label">Tải tài liệu</span>
                        <div className={`dropzone ${isProcessing ? 'active' : ''}`}>
                            <Icons.CloudUpload />
                            <h3>{isProcessing ? 'Đang phân tích...' : 'Kéo thả PDF / DOCX'}</h3>
                            <p>Hoặc nhấp để chọn file</p>
                            <input 
                                type="file" 
                                multiple 
                                accept=".pdf,.docx" 
                                onChange={handleFileUpload} 
                                disabled={isProcessing}
                            />
                        </div>
                        
                        {/* Auto applying changes warning */}
                        {processedFiles.length > 0 && (
                            <div className="upload-settings">
                                <div className="setting-row-sm">
                                    <span>Chunk Size:</span>
                                    <select 
                                        className="select-sm" 
                                        value={config.chunk_size} 
                                        onChange={(e) => handleConfigChange('chunk_size', parseInt(e.target.value))}
                                    >
                                        <option value="500">500 (Chi tiết)</option>
                                        <option value="1000">1000 (Cân bằng)</option>
                                        <option value="1500">1500 (Vừa phải)</option>
                                        <option value="2000">2000 (Rộng)</option>
                                    </select>
                                </div>
                                <div className="setting-row-sm" style={{ marginTop: '6px' }}>
                                    <span>Chunk Overlap:</span>
                                    <select 
                                        className="select-sm" 
                                        value={config.chunk_overlap} 
                                        onChange={(e) => handleConfigChange('chunk_overlap', parseInt(e.target.value))}
                                    >
                                        <option value="50">50</option>
                                        <option value="100">100</option>
                                        <option value="200">200</option>
                                    </select>
                                </div>
                                <div style={{ fontSize: '9px', color: 'var(--text-muted)', marginTop: '6px', textAlign: 'center' }}>
                                    ⚠️ Thay đổi cài đặt chunking yêu cầu upload lại tài liệu.
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Processed Files */}
                    <div className="sidebar-section">
                        <span className="section-label">Tài liệu đã lưu ({processedFiles.length})</span>
                        <div className="file-list">
                            {processedFiles.length === 0 ? (
                                <div className="empty-state-sm">Chưa có tài liệu nào.</div>
                            ) : (
                                processedFiles.map((file, idx) => {
                                    const isDocx = file.name.toLowerCase().endsWith('.docx');
                                    return (
                                        <div className="file-card" key={idx}>
                                            <div className={`file-icon ${isDocx ? 'docx' : 'pdf'}`}>
                                                {isDocx ? 'DOCX' : 'PDF'}
                                            </div>
                                            <div className="file-info">
                                                <div className="file-name" title={file.name}>{file.name}</div>
                                                <div className="file-meta">{file.pages} trang</div>
                                            </div>
                                            <div className="file-badge">{file.chunks}</div>
                                        </div>
                                    );
                                })
                            )}
                        </div>
                    </div>

                    {/* Chat History Sidebar */}
                    <div className="sidebar-section">
                        <span className="section-label">Lịch sử trò chuyện</span>
                        <div className="history-list">
                            {historyPairs.length === 0 ? (
                                <div className="empty-state-sm">Chưa có lịch sử trò chuyện.</div>
                            ) : (
                                [...historyPairs].reverse().map((pair, idx) => (
                                    <div className="history-card" key={idx} onClick={() => setSelectedHistory(pair)}>
                                        <div className="history-q">{pair.question}</div>
                                        <div className="history-a">{pair.answer || "Đang trả lời..."}</div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </div>

                {/* Clear Actions */}
                <div className="sidebar-footer">
                    {chatHistory.length > 0 && (
                        <button className="btn-danger-outline" onClick={() => setShowClearHistory(true)}>
                            <Icons.Trash />
                            <span>Xóa lịch sử chat</span>
                        </button>
                    )}
                    {processedFiles.length > 0 && (
                        <button className="btn-danger-outline" onClick={() => setShowClearDocs(true)}>
                            <Icons.Trash />
                            <span>Xóa sạch tài liệu</span>
                        </button>
                    )}
                </div>
            </aside>

            {/* ── Main Chat Area ── */}
            <main className="main-chat">
                <header className="chat-header">
                    <div className="chat-title">
                        {chatHistory.length > 0 ? "Trò chuyện với Tài liệu" : "SmartDocAI"}
                    </div>
                    <div className="header-controls">
                        <button 
                            className="btn-theme" 
                            onClick={() => setTheme(prev => prev === 'dark' ? 'light' : 'dark')}
                            title="Đổi giao diện"
                        >
                            {theme === 'dark' ? <Icons.Sun /> : <Icons.Moon />}
                        </button>
                    </div>
                </header>

                <div className="messages-container">
                    {chatHistory.length === 0 ? (
                        <div className="welcome-hero">
                            <div className="welcome-logo">S</div>
                            <h2>Trợ lý AI Tài liệu Offline</h2>
                            <p>Hệ thống chatbot hỏi đáp PDF & DOCX sử dụng RAG nâng cao chạy 100% offline trên thiết bị của bạn.</p>
                            
                            <div className="welcome-steps">
                                <div className="welcome-step">
                                    <div className="step-num">1</div>
                                    <div className="step-text">Tải file PDF hoặc DOCX ở thanh bên trái</div>
                                </div>
                                <div className="welcome-step">
                                    <div className="step-num">2</div>
                                    <div className="step-text">Bật/tắt cấu hình tìm kiếm nâng cao (Hybrid, Reranker, Agent...)</div>
                                </div>
                                <div className="welcome-step">
                                    <div className="step-num">3</div>
                                    <div className="step-text">Gõ câu hỏi vào khung chat bên dưới để bắt đầu</div>
                                </div>
                            </div>
                        </div>
                    ) : (
                        chatHistory.map((msg, msgIdx) => {
                            const isUser = msg.role === 'user';
                            return (
                                <div className={`message-card ${msg.role}`} key={msgIdx}>
                                    <div className="avatar">
                                        {isUser ? "U" : "AI"}
                                    </div>
                                    <div className="message-content">
                                        <div className="message-bubble">
                                            {isUser ? msg.content : renderMarkdown(msg.content)}
                                            
                                            {/* Self-RAG Metadata Display */}
                                            {!isUser && msg.self_rag_meta && (
                                                <div className="pipeline-meta-card">
                                                    <div className="pipeline-meta-header">Self-RAG Analysis</div>
                                                    <div className="pipeline-grid">
                                                        <div className="pipeline-stat">
                                                            Độ tự tin: <strong>{Math.round(msg.self_rag_meta.confidence_score * 100)}%</strong>
                                                        </div>
                                                        <div className="pipeline-stat">
                                                            Xác thực tài liệu: <strong>{msg.self_rag_meta.is_grounded ? '✅ Đạt' : '❌ Nghi ngờ'}</strong>
                                                        </div>
                                                        <div className="pipeline-stat">
                                                            Hallucination: <strong>{msg.self_rag_meta.has_hallucination ? '⚠️ Có thể' : '🛡️ Không'}</strong>
                                                        </div>
                                                    </div>
                                                    {msg.self_rag_meta.grading_feedback && (
                                                        <div className="pipeline-feedback">
                                                            {msg.self_rag_meta.grading_feedback}
                                                        </div>
                                                    )}
                                                    <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>
                                                        Tìm kiếm: {msg.self_rag_meta.docs_before_filter} chunks → Giữ lại: {msg.self_rag_meta.docs_after_filter} chunks
                                                        {msg.self_rag_meta.used_multihop && ` | Multi-hop (${msg.self_rag_meta.sub_questions.length} sub-questions)`}
                                                    </div>
                                                </div>
                                            )}

                                            {/* Co-RAG Metadata Display */}
                                            {!isUser && msg.co_rag_meta && (
                                                <div className="pipeline-meta-card">
                                                    <div className="pipeline-meta-header">Co-RAG Analysis (Consensus Merger)</div>
                                                    <div style={{ display: 'flex', gap: '12px', fontSize: '10.5px' }}>
                                                        {Object.entries(msg.co_rag_meta.co_rag_agent_counts).map(([agent, count]) => (
                                                            <span key={agent}>● <strong>{agent}</strong>: {count} docs</span>
                                                        ))}
                                                    </div>
                                                    <div className="pipeline-grid" style={{ marginTop: '4px', fontSize: '10px', color: 'var(--text-muted)' }}>
                                                        <span>Tổng trước khi gộp: <strong>{msg.co_rag_meta.co_rag_total_before_merge}</strong></span>
                                                        <span>Sau đồng thuận: <strong>{msg.co_rag_meta.co_rag_total_after_merge}</strong></span>
                                                        <span>Chiến lược: <strong style={{ color: 'var(--accent)' }}>{msg.co_rag_meta.co_rag_merge_strategy}</strong></span>
                                                    </div>
                                                </div>
                                            )}
                                        </div>

                                        {/* Sources (Citations) */}
                                        {!isUser && msg.sources && msg.sources.length > 0 && (
                                            <div className="citations-wrapper">
                                                <div className="citation-pills-row">
                                                    {msg.sources.map((src, srcIdx) => {
                                                        const activeKey = msgIdx.toString();
                                                        const isActive = activeCitations[activeKey] === srcIdx;
                                                        return (
                                                            <button 
                                                                className={`citation-pill ${isActive ? 'active' : ''}`}
                                                                key={srcIdx}
                                                                onClick={() => handleCitationClick(msgIdx, srcIdx)}
                                                            >
                                                                <span>{srcIdx + 1}. {src.file} (Trang {src.page})</span>
                                                                {isActive ? <Icons.ChevronUp /> : <Icons.ChevronDown />}
                                                            </button>
                                                        );
                                                    })}
                                                </div>

                                                {/* Citation Text Panel */}
                                                {activeCitations[msgIdx.toString()] !== undefined && (
                                                    (() => {
                                                        const activeIndex = activeCitations[msgIdx.toString()];
                                                        const activeSource = msg.sources[activeIndex];
                                                        if (!activeSource) return null;
                                                        
                                                        const pct = Math.min(100, Math.round(activeSource.score * 100));
                                                        const markedHtml = highlightText(activeSource.content, msg.question_ctx, msg.content);
                                                        
                                                        return (
                                                            <div className="citation-detail-panel">
                                                                <div className="citation-detail-header">
                                                                    <div className="citation-detail-title">
                                                                        <span className="step-num" style={{ width: '20px', height: '20px', fontSize: '10px' }}>
                                                                            {activeIndex + 1}
                                                                        </span>
                                                                        <span>{activeSource.file}</span>
                                                                    </div>
                                                                    <div className="citation-detail-badge">
                                                                        Trang {activeSource.page} / {activeSource.total_pages || '?'}
                                                                    </div>
                                                                </div>
                                                                
                                                                <div className="citation-score-row">
                                                                    <span>Độ liên quan:</span>
                                                                    <div className="score-bar-bg">
                                                                        <div className="score-bar-fill" style={{ width: `${pct}%` }}></div>
                                                                    </div>
                                                                    <span className="score-value">{pct}%</span>
                                                                </div>

                                                                <div 
                                                                    className="citation-text"
                                                                    dangerouslySetInnerHTML={{ __html: markedHtml }}
                                                                />
                                                            </div>
                                                        );
                                                    })()
                                                )}
                                            </div>
                                        )}
                                        
                                        {/* Status badge row for Assistant Search mode info */}
                                        {!isUser && msg.search_mode && (
                                            <div style={{ fontSize: '10.5px', color: 'var(--text-muted)', display: 'flex', gap: '8px', paddingLeft: '4px' }}>
                                                <span>Chế độ: <strong>{msg.search_mode.replace('self_rag', 'Self-RAG').replace('co_rag', 'Co-RAG').replace('hybrid', 'Hybrid Search').replace('vector', 'Vector Search')}</strong></span>
                                                {msg.active_filter && msg.active_filter.length > 0 && (
                                                    <span>• Bộ lọc: {msg.active_filter.join(', ')}</span>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            );
                        })
                    )}

                    {isThinking && (
                        <div className="spinner-card">
                            <div className="loader"></div>
                            <p>AI đang phân tích tài liệu và suy nghĩ...</p>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* ── Input and Settings Panel ── */}
                <div className="input-area">
                    {/* Collapsible search drawer toggle */}
                    <div 
                        className="settings-drawer-toggle" 
                        onClick={() => setIsSettingsOpen(prev => !prev)}
                    >
                        <span>⚙️ Cài đặt tìm kiếm tài liệu nâng cao</span>
                        {isSettingsOpen ? <Icons.ChevronUp /> : <Icons.ChevronDown />}
                    </div>

                    {isSettingsOpen && (
                        <div className="settings-drawer">
                            {/* Left Side: Metadata and Base Configs */}
                            <div className="drawer-section">
                                <h4>Lọc tài liệu & Cơ bản</h4>
                                <div className="setting-control">
                                    <label>Chỉ tìm kiếm trong file (Q8):</label>
                                    <div className="multiselect">
                                        {processedFiles.length === 0 ? (
                                            <div style={{ color: 'var(--text-muted)', fontSize: '10.5px' }}>Tải tài liệu lên để kích hoạt lọc</div>
                                        ) : (
                                            processedFiles.map((file, idx) => {
                                                const isChecked = config.active_file_filter.includes(file.name);
                                                return (
                                                    <div 
                                                        className="checkbox-row" 
                                                        key={idx}
                                                        onClick={() => {
                                                            const newFilter = isChecked
                                                                ? config.active_file_filter.filter(name => name !== file.name)
                                                                : [...config.active_file_filter, file.name];
                                                            handleConfigChange('active_file_filter', newFilter);
                                                        }}
                                                    >
                                                        <input 
                                                            type="checkbox" 
                                                            checked={isChecked}
                                                            onChange={() => {}} // handled by onClick on wrapper
                                                        />
                                                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                                            {file.name}
                                                        </span>
                                                    </div>
                                                );
                                            })
                                        )}
                                    </div>
                                </div>

                                <div className="toggle-row" style={{ marginTop: '8px' }}>
                                    <label className="toggle-switch">
                                        <input 
                                            type="checkbox"
                                            checked={config.hybrid_enabled}
                                            disabled={config.self_rag_enabled || config.co_rag_enabled || processedFiles.length === 0}
                                            onChange={(e) => handleConfigChange('hybrid_enabled', e.target.checked)}
                                        />
                                        <span className="toggle-slider"></span>
                                    </label>
                                    <div className="toggle-info">
                                        <span className="toggle-title">Bật Hybrid Search (BM25 + Vector)</span>
                                        <span className="toggle-desc">Kết hợp ngữ nghĩa và từ khóa. Bypass bởi Self/Co-RAG.</span>
                                    </div>
                                </div>

                                <div className="toggle-row">
                                    <label className="toggle-switch">
                                        <input 
                                            type="checkbox"
                                            checked={config.reranker_enabled}
                                            disabled={config.self_rag_enabled || config.co_rag_enabled || processedFiles.length === 0}
                                            onChange={(e) => handleConfigChange('reranker_enabled', e.target.checked)}
                                        />
                                        <span className="toggle-slider"></span>
                                    </label>
                                    <div className="toggle-info">
                                        <span className="toggle-title">Bật Re-ranking (Cross-Encoder)</span>
                                        <span className="toggle-desc">Đánh giá và sắp xếp lại kết quả chính xác hơn.</span>
                                    </div>
                                </div>
                            </div>

                            {/* Right Side: Self-RAG & Co-RAG */}
                            <div className="drawer-section">
                                <h4>Pipeline AI Nâng cao (Q10)</h4>
                                
                                {/* Self-RAG Toggle */}
                                <div className="toggle-row">
                                    <label className="toggle-switch">
                                        <input 
                                            type="checkbox"
                                            checked={config.self_rag_enabled}
                                            disabled={processedFiles.length === 0}
                                            onChange={(e) => {
                                                const checked = e.target.checked;
                                                const updated = { ...config, self_rag_enabled: checked };
                                                if (checked) {
                                                    updated.co_rag_enabled = false;
                                                }
                                                setConfig(updated);
                                                updateRemoteConfig(updated);
                                            }}
                                        />
                                        <span className="toggle-slider"></span>
                                    </label>
                                    <div className="toggle-info">
                                        <span className="toggle-title">Bật Self-RAG (AI Tự Đánh Giá)</span>
                                        <span className="toggle-desc">LLM viết lại câu hỏi, lọc chunks và tự chấm điểm câu trả lời.</span>
                                    </div>
                                </div>

                                {config.self_rag_enabled && (
                                    <div className="sub-options">
                                        <label className="checkbox-row">
                                            <input 
                                                type="checkbox" 
                                                checked={config.self_rag_query_rewrite}
                                                onChange={(e) => handleConfigChange('self_rag_query_rewrite', e.target.checked)}
                                            />
                                            <span>Query Rewriting (Viết lại câu hỏi)</span>
                                        </label>
                                        <label className="checkbox-row">
                                            <input 
                                                type="checkbox" 
                                                checked={config.self_rag_relevance_filter}
                                                onChange={(e) => handleConfigChange('self_rag_relevance_filter', e.target.checked)}
                                            />
                                            <span>Relevance Filtering (Lọc tài liệu rác)</span>
                                        </label>
                                        <label className="checkbox-row">
                                            <input 
                                                type="checkbox" 
                                                checked={config.self_rag_answer_grading}
                                                onChange={(e) => handleConfigChange('self_rag_answer_grading', e.target.checked)}
                                            />
                                            <span>Answer Grading (Tự chấm điểm câu trả lời)</span>
                                        </label>
                                    </div>
                                )}

                                {/* Co-RAG Toggle */}
                                <div className="toggle-row" style={{ marginTop: '4px' }}>
                                    <label className="toggle-switch">
                                        <input 
                                            type="checkbox"
                                            checked={config.co_rag_enabled}
                                            disabled={processedFiles.length === 0}
                                            onChange={(e) => {
                                                const checked = e.target.checked;
                                                const updated = { ...config, co_rag_enabled: checked };
                                                if (checked) {
                                                    updated.self_rag_enabled = false;
                                                }
                                                setConfig(updated);
                                                updateRemoteConfig(updated);
                                            }}
                                        />
                                        <span className="toggle-slider"></span>
                                    </label>
                                    <div className="toggle-info">
                                        <span className="toggle-title">Bật Co-RAG (Multi-Agent RAG)</span>
                                        <span className="toggle-desc">Sử dụng nhiều agents tìm kiếm độc lập và gộp biểu quyết đồng thuận.</span>
                                    </div>
                                </div>

                                {config.co_rag_enabled && (
                                    <div className="sub-options">
                                        <div className="setting-row-sm" style={{ marginBottom: '6px' }}>
                                            <span>Chiến lược Merge:</span>
                                            <select 
                                                className="select-sm"
                                                value={config.co_rag_merge_strategy}
                                                onChange={(e) => handleConfigChange('co_rag_merge_strategy', e.target.value)}
                                            >
                                                <option value="voting">voting (biểu quyết)</option>
                                                <option value="union">union (gộp toàn bộ)</option>
                                                <option value="intersection">intersection (giao hội)</option>
                                            </select>
                                        </div>
                                        <label className="checkbox-row">
                                            <input 
                                                type="checkbox" 
                                                checked={config.co_rag_agent_semantic}
                                                onChange={(e) => handleConfigChange('co_rag_agent_semantic', e.target.checked)}
                                            />
                                            <span>Semantic Agent (Vector FAISS)</span>
                                        </label>
                                        <label className="checkbox-row">
                                            <input 
                                                type="checkbox" 
                                                checked={config.co_rag_agent_keyword}
                                                onChange={(e) => handleConfigChange('co_rag_agent_keyword', e.target.checked)}
                                            />
                                            <span>Keyword Agent (BM25)</span>
                                        </label>
                                        <label className="checkbox-row">
                                            <input 
                                                type="checkbox" 
                                                checked={config.co_rag_agent_conceptual}
                                                onChange={(e) => handleConfigChange('co_rag_agent_conceptual', e.target.checked)}
                                            />
                                            <span>Conceptual Agent (LLM Decompose)</span>
                                        </label>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Chat Input Bar */}
                    <form className="chat-input-bar" onSubmit={handleSendMessage}>
                        <textarea
                            className="chat-textarea"
                            placeholder={processedFiles.length === 0 ? "⚠️ Vui lòng tải tài liệu lên trước khi đặt câu hỏi..." : "Đặt câu hỏi về nội dung tài liệu của bạn..."}
                            value={chatInput}
                            onChange={(e) => setChatInput(e.target.value)}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault();
                                    handleSendMessage();
                                }
                            }}
                            disabled={isThinking || processedFiles.length === 0}
                        />
                        <button 
                            type="submit" 
                            className="btn-send"
                            disabled={!chatInput.trim() || isThinking || processedFiles.length === 0}
                            title="Gửi câu hỏi"
                        >
                            <Icons.Send />
                        </button>
                    </form>
                </div>
            </main>

            {/* ── Modal Dialog: Clear History Confirm ── */}
            {showClearHistory && (
                <div className="modal-overlay">
                    <div className="modal-card">
                        <div className="modal-header">Xác nhận xóa lịch sử</div>
                        <div className="modal-body">Bạn có chắc chắn muốn xóa toàn bộ lịch sử hội thoại hiện tại? Hành động này không thể hoàn tác.</div>
                        <div className="modal-actions">
                            <button className="btn-btn btn-secondary" onClick={() => setShowClearHistory(false)}>Hủy bỏ</button>
                            <button className="btn-btn btn-danger" onClick={handleClearHistory}>Xác nhận xóa</button>
                        </div>
                    </div>
                </div>
            )}

            {/* ── Modal Dialog: Clear Documents Confirm ── */}
            {showClearDocs && (
                <div className="modal-overlay">
                    <div className="modal-card">
                        <div className="modal-header">Xác nhận xóa tài liệu</div>
                        <div className="modal-body">Bạn có chắc chắn muốn xóa toàn bộ tài liệu đã tải lên và vector store? Lịch sử trò chuyện cũng sẽ bị xóa. Hành động này không thể hoàn tác.</div>
                        <div className="modal-actions">
                            <button className="btn-btn btn-secondary" onClick={() => setShowClearDocs(false)}>Hủy bỏ</button>
                            <button className="btn-btn btn-danger" onClick={handleClearDocuments}>Xác nhận xóa</button>
                        </div>
                    </div>
                </div>
            )}

            {/* ── Modal Dialog: History Pair Detail ── */}
            {selectedHistory && (
                <div className="modal-overlay" onClick={() => setSelectedHistory(null)}>
                    <div className="modal-card" style={{ maxWidth: '720px', width: '95%' }} onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header" style={{ borderBottom: '1px solid var(--border-subtle)', paddingBottom: '10px', marginBottom: '14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <span>Chi tiết Lịch sử Hội thoại</span>
                            <button className="btn-icon-sm" onClick={() => setSelectedHistory(null)}>✕</button>
                        </div>
                        
                        <div className="modal-body" style={{ maxHeight: '480px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                            {/* Question */}
                            <div style={{ backgroundColor: 'var(--accent-soft)', border: '1px solid var(--accent-border)', borderRadius: 'var(--radius-md)', padding: '12px 16px' }}>
                                <div style={{ fontSize: '10px', fontWeight: '800', color: 'var(--accent)', textTransform: 'uppercase', marginBottom: '6px' }}>Người dùng</div>
                                <div style={{ fontSize: '13px', fontWeight: '600', color: 'var(--text-primary)' }}>{selectedHistory.question}</div>
                            </div>
                            
                            {/* Answer */}
                            <div>
                                <div style={{ fontSize: '10px', fontWeight: '800', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px' }}>Trợ lý AI</div>
                                <div style={{ fontSize: '13px', color: 'var(--text-primary)', backgroundColor: 'var(--bg-surface)', padding: '12px 16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                                    {renderMarkdown(selectedHistory.answer)}
                                </div>
                            </div>

                            {/* Self-RAG/Co-RAG Meta in dialog */}
                            {selectedHistory.selfRagMeta && (
                                <div className="pipeline-meta-card">
                                    <div className="pipeline-meta-header">Self-RAG Analysis</div>
                                    <div className="pipeline-grid">
                                        <span>Confidence: <strong>{Math.round(selectedHistory.selfRagMeta.confidence_score * 100)}%</strong></span>
                                        <span>Grounded: <strong>{selectedHistory.selfRagMeta.is_grounded ? 'Đạt' : 'Nghi ngờ'}</strong></span>
                                        <span>Hallucination: <strong>{selectedHistory.selfRagMeta.has_hallucination ? 'Có thể' : 'Không'}</strong></span>
                                    </div>
                                    {selectedHistory.selfRagMeta.grading_feedback && (
                                        <div className="pipeline-feedback">{selectedHistory.selfRagMeta.grading_feedback}</div>
                                    )}
                                </div>
                            )}

                            {selectedHistory.coRagMeta && (
                                <div className="pipeline-meta-card">
                                    <div className="pipeline-meta-header">Co-RAG Analysis</div>
                                    <div style={{ display: 'flex', gap: '12px', fontSize: '10.5px' }}>
                                        {Object.entries(selectedHistory.coRagMeta.co_rag_agent_counts).map(([agent, count]) => (
                                            <span key={agent}>● <strong>{agent}</strong>: {count} docs</span>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Sources */}
                            {selectedHistory.sources && selectedHistory.sources.length > 0 && (
                                <div>
                                    <div style={{ fontSize: '10px', fontWeight: '800', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '8px' }}>Nguồn trích dẫn ({selectedHistory.sources.length})</div>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                        {selectedHistory.sources.map((src, srcIdx) => (
                                            <div key={srcIdx} style={{ padding: '10px 12px', backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)' }}>
                                                <div style={{ fontSize: '11px', fontWeight: '700', marginBottom: '4px', display: 'flex', justifyContent: 'space-between' }}>
                                                    <span>{srcIdx + 1}. {src.file}</span>
                                                    <span style={{ color: 'var(--accent)' }}>Trang {src.page} (Độ khớp: {Math.round(src.score * 100)}%)</span>
                                                </div>
                                                <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)', fontStyle: 'italic', backgroundColor: 'var(--bg-primary)', padding: '6px 10px', borderRadius: '4px', marginTop: '6px' }}>
                                                    {src.content}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                        
                        <div className="modal-actions" style={{ marginTop: '16px', borderTop: '1px solid var(--border-subtle)', paddingTop: '12px' }}>
                            <button className="btn-btn btn-secondary" onClick={() => setSelectedHistory(null)}>Đóng</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

// Render React App
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);

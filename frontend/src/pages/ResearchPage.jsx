import { useState, useRef, useEffect } from 'react'
import { Send, Sparkles, Search, GitBranch, Layers, Network, CheckCircle, AlertCircle, FileText } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useAuth } from '../contexts/AuthContext'

const PIPELINE_STAGES = [
    { key: 'classifying', label: 'Classify', icon: Sparkles },
    { key: 'planning', label: 'Plan', icon: GitBranch },
    { key: 'retrieving', label: 'Retrieve', icon: Search },
    { key: 'hop', label: 'Multi-Hop', icon: Network },
    { key: 'generating', label: 'Generate', icon: Layers },
    { key: 'evaluating', label: 'Evaluate', icon: CheckCircle },
]

function getStageStatus(stageKey, completedStages, activeStage) {
    if (completedStages.has(stageKey)) return 'completed'
    if (activeStage === stageKey) return 'active'
    return 'pending'
}

export default function ResearchPage({ result, setResult, history, addToHistory, setActiveHistoryIndex }) {
    const [query, setQuery] = useState('')
    const [loading, setLoading] = useState(false)
    const [activeStage, setActiveStage] = useState(null)
    const [completedStages, setCompletedStages] = useState(new Set())
    const [stageDetail, setStageDetail] = useState('')
    const [error, setError] = useState(null)
    const resultRef = useRef(null)
    const { getToken } = useAuth()

    useEffect(() => {
        if (result && resultRef.current) {
            resultRef.current.scrollIntoView({ behavior: 'smooth' })
        }
    }, [result])

    const handleSubmit = async (e) => {
        e.preventDefault()
        if (!query.trim() || loading) return

        setLoading(true)
        setResult(null)
        setError(null)
        setActiveStage('classifying')
        setCompletedStages(new Set())
        setStageDetail('Starting research pipeline...')
        setActiveHistoryIndex(null)

        try {
            const token = await getToken()
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`,
                },
                body: JSON.stringify({ question: query.trim() }),
            })

            const reader = response.body.getReader()
            const decoder = new TextDecoder()
            let buffer = ''

            while (true) {
                const { value, done } = await reader.read()
                if (done) break

                buffer += decoder.decode(value, { stream: true })
                const lines = buffer.split('\n')
                buffer = lines.pop() || ''

                let eventType = ''
                for (const line of lines) {
                    if (line.startsWith('event: ')) {
                        eventType = line.slice(7).trim()
                    } else if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6))

                            if (eventType === 'stage') {
                                const stage = data.stage

                                setActiveStage((prev) => {
                                    if (prev && prev !== stage) {
                                        setCompletedStages((s) => {
                                            const ns = new Set(s)
                                            ns.add(prev)
                                            return ns
                                        })
                                    }
                                    return stage
                                })

                                setStageDetail(data.detail || '')
                            } else if (eventType === 'answer') {
                                setActiveStage(null)
                                setCompletedStages(new Set(PIPELINE_STAGES.map((s) => s.key)))
                                setStageDetail('')
                                setResult(data)
                                addToHistory({
                                    question: query.trim(),
                                    answer: data.answer,
                                    mode: data.mode,
                                    plan: data.plan,
                                    sources: data.sources,
                                })
                            } else if (eventType === 'error') {
                                setError(data.message)
                            }
                        } catch {
                            // skip malformed JSON
                        }
                    }
                }
            }
        } catch (err) {
            setError(err.message || 'Failed to connect to backend')
        } finally {
            setLoading(false)
        }
    }

    const showPipeline = loading || result || completedStages.size > 0

    return (
        <div className="research-page">
            <div className="research-results">
                {!showPipeline && !error && <WelcomeHero />}

                {showPipeline && (
                    <div className="slide-up">
                        <div className="pipeline-visualizer">
                            {PIPELINE_STAGES.map((stage, i) => {
                                const status = getStageStatus(stage.key, completedStages, activeStage)
                                const Icon = stage.icon
                                return (
                                    <div key={stage.key} className="pipeline-step">
                                        {i > 0 && (
                                            <div className={`pipeline-connector ${status === 'completed' ? 'completed' : status === 'active' ? 'active' : ''}`} />
                                        )}
                                        <div className={`pipeline-node ${status}`}>
                                            <Icon />
                                            {stage.label}
                                        </div>
                                    </div>
                                )
                            })}
                        </div>

                        {stageDetail && (
                            <div className="pipeline-detail">
                                <div className="spinner" />
                                {stageDetail}
                            </div>
                        )}
                    </div>
                )}

                {error && (
                    <div className="answer-container fade-in" style={{ borderColor: 'var(--danger-border)' }}>
                        <div className="answer-header">
                            <AlertCircle size={18} color="var(--danger)" />
                            <span style={{ color: 'var(--danger)', fontWeight: 600 }}>Error</span>
                        </div>
                        <div className="answer-body">
                            <p>{error}</p>
                        </div>
                    </div>
                )}

                {result && (
                    <div ref={resultRef} className="answer-container slide-up">
                        <div className="answer-header">
                            <Sparkles size={18} color="var(--accent-primary)" />
                            <span style={{ fontWeight: 600 }}>Research Answer</span>
                            <span className={`mode-badge ${result.mode}`}>
                                {result.mode === 'deep' ? '🔬 Deep Research' : '⚡ Quick Answer'}
                            </span>
                            {result.grounded !== false && (
                                <span className="mode-badge" style={{ background: 'var(--success-bg)', color: 'var(--success)', borderColor: 'rgba(34, 197, 94, 0.2)' }}>
                                    ✓ Grounded
                                </span>
                            )}
                        </div>

                        <div className="answer-body">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {result.answer}
                            </ReactMarkdown>
                        </div>

                        {result.sources && result.sources.length > 0 && (
                            <div className="sources-section">
                                <h4>📚 Sources Cited</h4>
                                <div className="sources-grid">
                                    {result.sources.map((src, i) => (
                                        <div key={i} className="source-chip">
                                            <FileText size={12} />
                                            {src.title} — Page {src.page}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {result.plan && result.plan.length > 1 && (
                            <div className="subqueries-section">
                                <h4>🔎 Research Sub-Queries</h4>
                                <div className="subquery-list">
                                    {result.plan.map((q, i) => (
                                        <span key={i} className="subquery-tag">{q}</span>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>

            <div className="research-input-bar">
                <form onSubmit={handleSubmit} className="research-input-wrapper">
                    <input
                        type="text"
                        placeholder="Ask a research question... (e.g. Compare adaptive learning approaches in AI education)"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        disabled={loading}
                        id="research-input"
                    />
                    <button type="submit" disabled={loading || !query.trim()} id="research-submit">
                        {loading ? <div className="spinner" style={{ borderTopColor: 'white' }} /> : <Send size={16} />}
                        {loading ? 'Researching...' : 'Research'}
                    </button>
                </form>
            </div>
        </div>
    )
}


function WelcomeHero() {
    return (
        <div className="welcome-hero">
            <span className="hero-badge">Multi-Agent RAG System</span>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '20px', marginBottom: '16px', marginTop: '16px' }}>
                <img src="/logo.png" alt="PaperMind Logo" style={{ height: 56, width: 'auto', objectFit: 'contain' }} />
                <h2 className="logo-gradient" style={{ fontSize: '3.5rem', margin: 0, fontWeight: 800 }}>PaperMind</h2>
            </div>
            <p>
                Ask complex research questions and get comprehensive, citation-grounded answers
                powered by a multi-agent pipeline with hybrid retrieval and self-reflection.
            </p>
            <div className="feature-grid">
                <div className="feature-item">
                    <GitBranch size={20} />
                    <h4>Query Planning</h4>
                    <p>Auto-decomposes complex questions into targeted sub-queries</p>
                </div>
                <div className="feature-item">
                    <Search size={20} />
                    <h4>Hybrid Retrieval</h4>
                    <p>FAISS dense vectors + BM25 sparse search with cross-encoder reranking</p>
                </div>
                <div className="feature-item">
                    <Network size={20} />
                    <h4>Multi-Hop Reasoning</h4>
                    <p>Iterative evidence gathering with automatic research expansion</p>
                </div>
            </div>
        </div>
    )
}

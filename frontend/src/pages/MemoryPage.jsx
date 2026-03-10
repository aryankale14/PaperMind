import { useState, useEffect } from 'react'
import { Brain, Search, Star } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'

export default function MemoryPage() {
    const [memories, setMemories] = useState([])
    const [loading, setLoading] = useState(true)
    const [search, setSearch] = useState('')
    const { getToken } = useAuth()

    useEffect(() => {
        const fetchMemory = async () => {
            try {
                const token = await getToken()
                const res = await fetch('/api/memory', {
                    headers: { Authorization: `Bearer ${token}` },
                })
                const data = await res.json()
                setMemories(data.memories || [])
            } catch {
                setMemories([])
            } finally {
                setLoading(false)
            }
        }
        fetchMemory()
    }, [])

    const filtered = memories.filter((m) => {
        const q = search.toLowerCase()
        return (
            (m.topic && m.topic.toLowerCase().includes(q)) ||
            (m.key_finding && m.key_finding.toLowerCase().includes(q))
        )
    })

    const getImportanceClass = (imp) => {
        if (imp >= 3) return 'high'
        if (imp >= 2) return 'medium'
        return 'low'
    }

    if (loading) {
        return (
            <div className="page-container">
                <div className="page-header">
                    <h2>Research Memory</h2>
                </div>
                <div className="empty-state">
                    <div className="spinner" style={{ margin: '0 auto 16px' }} />
                    <p>Loading memory...</p>
                </div>
            </div>
        )
    }

    return (
        <div className="page-container fade-in">
            <div className="page-header">
                <h2>Research Memory</h2>
                <p>Persistent knowledge extracted from your research sessions — {memories.length} memories stored</p>
            </div>

            <div style={{
                backgroundColor: 'rgba(77, 148, 255, 0.05)',
                border: '1px solid rgba(77, 148, 255, 0.2)',
                borderRadius: 'var(--radius-md)',
                padding: '16px 20px',
                marginBottom: '24px',
                display: 'flex',
                gap: '16px',
                alignItems: 'flex-start'
            }}>
                <Brain size={20} color="var(--accent-primary)" style={{ flexShrink: 0, marginTop: '4px' }} />
                <div>
                    <h4 style={{ margin: '0 0 6px 0', color: 'var(--text-primary)', fontSize: '0.95rem' }}>Understanding Importance Scores</h4>
                    <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.875rem', lineHeight: '1.6' }}>
                        The AI agent evaluates and assigns an <strong>Importance Score</strong> to every memory it extracts to prioritize future context retrieval.
                        <br />
                        <span style={{ color: '#d8b4fe', fontWeight: 600 }}>Score 3 (High):</span> Critical breakthrough findings, major conclusions, or fundamental concepts.<br />
                        <span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>Score 1 (Low):</span> Minor supporting details, tangential context, or highly specific data points.
                    </p>
                </div>
            </div>

            {memories.length > 0 && (
                <div className="search-box">
                    <Search size={16} />
                    <input
                        type="text"
                        placeholder="Search memories..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        id="memory-search"
                    />
                </div>
            )}

            {filtered.length > 0 ? (
                <div className="memory-table-container">
                    <table className="memory-table">
                        <thead>
                            <tr>
                                <th style={{ width: '30px' }}></th>
                                <th>Topic</th>
                                <th>Key Finding</th>
                                <th style={{ width: '90px', textAlign: 'center' }}>Importance</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filtered.map((mem, i) => (
                                <tr key={i} className="slide-up" style={{ animationDelay: `${i * 0.02}s` }}>
                                    <td style={{ textAlign: 'center' }}>
                                        <Star size={14} color="var(--warning)" style={{ opacity: (mem.importance || 1) / 3 }} />
                                    </td>
                                    <td className="topic-cell">{mem.topic || 'Untitled'}</td>
                                    <td>{mem.key_finding || '—'}</td>
                                    <td style={{ textAlign: 'center' }}>
                                        <span className={`importance-badge ${getImportanceClass(mem.importance || 1)}`}>
                                            {mem.importance || 1}
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            ) : (
                <div className="empty-state" style={{ marginTop: 40 }}>
                    <div className="empty-icon">
                        <Brain size={28} />
                    </div>
                    <h3>{search ? 'No matching memories' : 'No memories yet'}</h3>
                    <p>
                        {search
                            ? 'Try a different search term'
                            : 'Research memories are automatically saved after each query — ask a research question to start building memory'}
                    </p>
                </div>
            )}
        </div>
    )
}

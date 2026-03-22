import { NavLink, useNavigate } from 'react-router-dom'
import { Search, FileText, Share2, Brain, Clock, LogOut, Shield } from 'lucide-react'
import { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'

const ADMIN_EMAIL = "aryankale1410@gmail.com"

export default function Sidebar({ history = [], activeHistoryIndex, onHistoryClick }) {
    const [historySearch, setHistorySearch] = useState('')
    const { user, logout } = useAuth()
    const navigate = useNavigate()

    const filteredHistory = history.filter((h) =>
        h.question.toLowerCase().includes(historySearch.toLowerCase())
    )

    const handleEntryClick = (entry, realIndex) => {
        onHistoryClick(entry, realIndex)
        navigate('/app')
    }

    const handleLogout = async () => {
        await logout()
        navigate('/')
    }

    return (
        <aside className="sidebar">
            <div className="sidebar-brand">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px', padding: '12px 0 12px 0' }}>
                    <img src="/logo.png" alt="PaperMind Logo" style={{ height: 28, width: 'auto', objectFit: 'contain' }} />
                    <span className="logo-gradient" style={{ fontSize: '1.4rem', fontWeight: 800, letterSpacing: '-0.02em' }}>PaperMind</span>
                </div>
                <p style={{ marginTop: 0 }}>Multi-Agent RAG System</p>
            </div>

            <nav className="sidebar-nav">
                <NavLink to="/app" end className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                    <Search />
                    Research
                </NavLink>
                <NavLink to="/app/papers" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                    <FileText />
                    Papers
                </NavLink>
                <NavLink to="/app/graph" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                    <Share2 />
                    Knowledge Graph
                </NavLink>
                <NavLink to="/app/memory" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                    <Brain />
                    Research Memory
                </NavLink>

                {user?.email === ADMIN_EMAIL && (
                    <NavLink to="/app/admin" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} style={{ marginTop: '16px', color: 'var(--accent-primary)' }}>
                        <Shield />
                        Admin Dashboard
                    </NavLink>
                )}
            </nav>

            {/* ── Research History ─────────────── */}
            <div className="sidebar-history">
                <div className="sidebar-history-header">
                    <span><Clock size={13} /> Your Research</span>
                    <span className="history-count">{history.length}</span>
                </div>

                {history.length > 3 && (
                    <div className="sidebar-history-search">
                        <Search size={12} />
                        <input
                            type="text"
                            placeholder="Search..."
                            value={historySearch}
                            onChange={(e) => setHistorySearch(e.target.value)}
                        />
                    </div>
                )}

                <div className="sidebar-history-list">
                    {filteredHistory.length > 0 ? (
                        [...filteredHistory].reverse().map((entry, i) => {
                            const realIndex = filteredHistory.length - 1 - i
                            return (
                                <div
                                    key={i}
                                    className={`sidebar-history-item ${activeHistoryIndex === realIndex ? 'active' : ''}`}
                                    onClick={() => handleEntryClick(entry, realIndex)}
                                    title={entry.question}
                                >
                                    <span className="sidebar-history-text">{entry.question}</span>
                                </div>
                            )
                        })
                    ) : (
                        <div className="sidebar-history-empty">
                            {historySearch ? 'No matches' : 'No research yet'}
                        </div>
                    )}
                </div>
            </div>

            <div className="sidebar-footer">
                <div className="sidebar-user">
                    <div className="sidebar-user-info">
                        <span className="sidebar-user-name">
                            {user?.displayName || user?.email?.split('@')[0] || 'User'}
                        </span>
                        <span className="sidebar-user-email">{user?.email}</span>
                    </div>
                    <button className="sidebar-logout" onClick={handleLogout} title="Sign out">
                        <LogOut size={15} />
                    </button>
                </div>
            </div>
        </aside>
    )
}

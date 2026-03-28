import { NavLink, useNavigate } from 'react-router-dom'
import { Search, FileText, Share2, Brain, Clock, LogOut, Shield, MoreVertical, Trash2 } from 'lucide-react'
import { useState, useRef, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'

const ADMIN_EMAIL = "aryankale1410@gmail.com"

export default function Sidebar({ history = [], activeHistoryIndex, onHistoryClick, onCloseMobile, onDeleteHistory }) {
    const [historySearch, setHistorySearch] = useState('')
    const [menuOpenId, setMenuOpenId] = useState(null)
    const { user, logout } = useAuth()
    const navigate = useNavigate()
    const menuRef = useRef(null)

    // Close menu when clicking outside
    useEffect(() => {
        function handleClickOutside(event) {
            if (menuRef.current && !menuRef.current.contains(event.target)) {
                setMenuOpenId(null)
            }
        }
        document.addEventListener("mousedown", handleClickOutside)
        return () => document.removeEventListener("mousedown", handleClickOutside)
    }, [])

    const filteredHistory = history.filter((h) =>
        h.question.toLowerCase().includes(historySearch.toLowerCase())
    )

    const handleEntryClick = (entry, realIndex) => {
        onHistoryClick(entry, realIndex)
        navigate('/app')
        if (onCloseMobile) onCloseMobile()
    }

    const handleLogout = async () => {
        await logout()
        navigate('/')
    }

    return (
        <aside className="sidebar">
            <div className="sidebar-brand" onClick={() => navigate('/')} style={{ cursor: 'pointer' }} title="Go to Landing Page">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px', padding: '12px 0 12px 0' }}>
                    <img src="/logo.png" alt="PaperMind Logo" style={{ height: 28, width: 'auto', objectFit: 'contain' }} />
                    <span className="logo-gradient" style={{ fontSize: '1.4rem', fontWeight: 800, letterSpacing: '-0.02em' }}>PaperMind</span>
                </div>
                <p style={{ marginTop: 0 }}>Multi-Agent RAG System</p>
            </div>

            <nav className="sidebar-nav">
                <NavLink to="/app" end onClick={() => onCloseMobile && onCloseMobile()} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                    <Search />
                    Research
                </NavLink>
                <NavLink to="/app/papers" onClick={() => onCloseMobile && onCloseMobile()} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                    <FileText />
                    Papers
                </NavLink>
                <NavLink to="/app/graph" onClick={() => onCloseMobile && onCloseMobile()} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                    <Share2 />
                    Knowledge Graph
                </NavLink>
                <NavLink to="/app/memory" onClick={() => onCloseMobile && onCloseMobile()} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                    <Brain />
                    Research Memory
                </NavLink>

                {user?.email === ADMIN_EMAIL && (
                    <NavLink to="/app/admin" onClick={() => onCloseMobile && onCloseMobile()} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} style={{ marginTop: '16px', color: 'var(--accent-primary)' }}>
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
                                    style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', position: 'relative' }}
                                >
                                    <span className="sidebar-history-text" style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                        {entry.question}
                                    </span>
                                    <button
                                        className="history-menu-btn"
                                        onClick={(e) => {
                                            e.stopPropagation()
                                            setMenuOpenId(menuOpenId === entry.id ? null : entry.id)
                                        }}
                                        style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', padding: '4px' }}
                                    >
                                        <MoreVertical size={14} />
                                    </button>

                                    {menuOpenId === entry.id && (
                                        <div
                                            ref={menuRef}
                                            style={{
                                                position: 'absolute',
                                                right: '24px',
                                                top: '100%',
                                                background: 'var(--bg-secondary)',
                                                border: '1px solid var(--border)',
                                                borderRadius: '6px',
                                                padding: '4px',
                                                zIndex: 100,
                                                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
                                            }}
                                        >
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation()
                                                    if (window.confirm("Delete this research query from history?")) {
                                                        onDeleteHistory(entry.id)
                                                    }
                                                    setMenuOpenId(null)
                                                }}
                                                style={{
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    gap: '8px',
                                                    background: 'none',
                                                    border: 'none',
                                                    color: 'var(--danger)',
                                                    cursor: 'pointer',
                                                    padding: '8px 12px',
                                                    width: '100%',
                                                    textAlign: 'left',
                                                    fontSize: '0.85rem',
                                                    borderRadius: '4px'
                                                }}
                                                onMouseEnter={(e) => e.target.style.background = 'rgba(239, 68, 68, 0.1)'}
                                                onMouseLeave={(e) => e.target.style.background = 'none'}
                                            >
                                                <Trash2 size={12} />
                                                Delete
                                            </button>
                                        </div>
                                    )}
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

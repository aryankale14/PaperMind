import React, { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { Shield, ChevronDown, ChevronRight, FileText, Search, Clock, Users, Trash2 } from 'lucide-react'

export default function AdminPage() {
    const { getToken, user } = useAuth()
    const [stats, setStats] = useState([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [expandedUser, setExpandedUser] = useState(null)
    const [isDeleting, setIsDeleting] = useState(false)

    useEffect(() => {
        const fetchStats = async () => {
            try {
                const token = await getToken()
                const API_BASE_URL = import.meta.env.VITE_API_URL || '';
                const res = await fetch(`${API_BASE_URL}/api/admin/users`, {
                    headers: { Authorization: `Bearer ${token}` }
                })

                if (!res.ok) {
                    throw new Error('You do not have permission to view this page.')
                }

                const data = await res.json()
                setStats(data.users)
            } catch (err) {
                setError(err.message)
            } finally {
                setLoading(false)
            }
        }

        fetchStats()
    }, [getToken])

    const handleDeleteUser = async (e, userId) => {
        e.stopPropagation(); // prevent row expansion
        if (!window.confirm("Are you sure you want to delete this user and all their data? This action cannot be undone.")) {
            return;
        }

        setIsDeleting(true);
        try {
            const token = await getToken();
            const API_BASE_URL = import.meta.env.VITE_API_URL || '';
            const res = await fetch(`${API_BASE_URL}/api/admin/users/${userId}`, {
                method: 'DELETE',
                headers: { Authorization: `Bearer ${token}` }
            });

            if (!res.ok) {
                const data = await res.json();
                throw new Error(data.detail || 'Failed to delete user.');
            }

            // Remove user from stats
            setStats(prev => prev.filter(u => u.id !== userId));
            if (expandedUser === userId) {
                setExpandedUser(null);
            }
        } catch (err) {
            alert(`Error: ${err.message}`);
        } finally {
            setIsDeleting(false);
        }
    };

    if (loading) {
        return (
            <div className="admin-page-centered">
                <div className="spinner" style={{ width: 40, height: 40 }}></div>
                <p style={{ marginTop: 16, color: 'var(--text-secondary)' }}>Loading admin data...</p>
            </div>
        )
    }

    if (error) {
        return (
            <div className="admin-page-centered">
                <Shield size={48} color="var(--danger)" style={{ marginBottom: 16 }} />
                <h2>Access Denied</h2>
                <p style={{ color: 'var(--text-secondary)', marginTop: 8 }}>{error}</p>
            </div>
        )
    }

    const totalUsers = stats.length
    const totalPapers = stats.reduce((acc, u) => acc + u.papers.length, 0)
    const totalQueries = stats.reduce((acc, u) => acc + u.queries.length, 0)

    return (
        <div className="page-container admin-page">
            <header className="page-header">
                <div>
                    <h1 className="page-title">
                        <Shield className="page-icon" color="var(--accent-primary)" />
                        Admin Dashboard
                    </h1>
                    <p className="page-subtitle">Monitor usage, signups, and platform activity data.</p>
                </div>
            </header>

            <div className="admin-metrics-grid">
                <div className="admin-metric-card">
                    <div className="metric-icon"><Users size={20} /></div>
                    <div className="metric-data">
                        <span className="metric-value">{totalUsers}</span>
                        <span className="metric-label">Total Users</span>
                    </div>
                </div>
                <div className="admin-metric-card">
                    <div className="metric-icon"><FileText size={20} /></div>
                    <div className="metric-data">
                        <span className="metric-value">{totalPapers}</span>
                        <span className="metric-label">Uploaded Papers</span>
                    </div>
                </div>
                <div className="admin-metric-card">
                    <div className="metric-icon"><Search size={20} /></div>
                    <div className="metric-data">
                        <span className="metric-value">{totalQueries}</span>
                        <span className="metric-label">Total Queries</span>
                    </div>
                </div>
            </div>

            <div className="admin-table-container">
                <table className="admin-table">
                    <thead>
                        <tr>
                            <th style={{ width: '40px' }}></th>
                            <th>User Name</th>
                            <th>Email Address</th>
                            <th>Joined Date</th>
                            <th>Papers</th>
                            <th>Queries</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {stats.map((u) => {
                            const isExpanded = expandedUser === u.id
                            const date = new Date(u.created_at).toLocaleDateString()

                            return (
                                <React.Fragment key={u.id}>
                                    <tr
                                        className={`admin-row ${isExpanded ? 'expanded' : ''}`}
                                        onClick={() => setExpandedUser(isExpanded ? null : u.id)}
                                    >
                                        <td className="admin-expander">
                                            {isExpanded ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
                                        </td>
                                        <td className="admin-cell-main" data-label="User Name">{u.display_name}</td>
                                        <td className="admin-cell-muted" data-label="Email">{u.email}</td>
                                        <td className="admin-cell-muted" data-label="Joined Date">{date}</td>
                                        <td data-label="Papers">
                                            <span className="admin-badge">{u.papers.length}</span>
                                        </td>
                                        <td data-label="Queries">
                                            <span className="admin-badge accent">{u.queries.length}</span>
                                        </td>
                                        <td data-label="Actions">
                                            <button 
                                                className="admin-delete-btn" 
                                                title="Delete User"
                                                onClick={(e) => handleDeleteUser(e, u.id)}
                                                disabled={isDeleting}
                                                style={{ background: 'none', border: 'none', color: 'var(--danger)', cursor: 'pointer', padding: '4px' }}
                                            >
                                                <Trash2 size={16} />
                                            </button>
                                        </td>
                                    </tr>

                                    {isExpanded && (
                                        <tr className="admin-details-row">
                                            <td colSpan={7}>
                                                <div className="admin-details-content">

                                                    <div className="admin-details-section">
                                                        <h4><FileText size={14} /> Uploaded Papers</h4>
                                                        {u.papers.length === 0 ? (
                                                            <p className="admin-empty">No papers uploaded.</p>
                                                        ) : (
                                                            <ul className="admin-list">
                                                                {u.papers.map(p => (
                                                                    <li key={p.id}>{p.title}</li>
                                                                ))}
                                                            </ul>
                                                        )}
                                                    </div>

                                                    <div className="admin-details-section">
                                                        <h4><Clock size={14} /> Research Queries</h4>
                                                        {u.queries.length === 0 ? (
                                                            <p className="admin-empty">No queries run.</p>
                                                        ) : (
                                                            <ul className="admin-list">
                                                                {u.queries.map((q, idx) => (
                                                                    <li key={idx}>
                                                                        <span className="query-text">"{q.question}"</span>
                                                                        <span className="query-date">
                                                                            {new Date(q.timestamp).toLocaleString()}
                                                                        </span>
                                                                    </li>
                                                                ))}
                                                            </ul>
                                                        )}
                                                    </div>

                                                </div>
                                            </td>
                                        </tr>
                                    )}
                                </React.Fragment>
                            )
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    )
}

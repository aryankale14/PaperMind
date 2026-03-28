import { useState, useEffect, useCallback } from 'react'
import { Routes, Route, Navigate, useNavigate } from 'react-router-dom'
import { Menu, X, LogOut } from 'lucide-react'
import { AuthProvider, useAuth } from './contexts/AuthContext'
import Sidebar from './components/Sidebar'
import ResearchPage from './pages/ResearchPage'
import PapersPage from './pages/PapersPage'
import GraphPage from './pages/GraphPage'
import MemoryPage from './pages/MemoryPage'
import AdminPage from './pages/AdminPage'
import LoginPage from './pages/LoginPage'
import LandingPage from './pages/LandingPage'

function ProtectedApp() {
    const { user, loading, getToken, logout } = useAuth()
    const [result, setResult] = useState(null)
    const [history, setHistory] = useState([])
    const [activeHistoryIndex, setActiveHistoryIndex] = useState(null)
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

    // Load history from backend on login
    useEffect(() => {
        if (!user) return
            ; (async () => {
                try {
                    const token = await getToken()
                    const API_BASE_URL = import.meta.env.VITE_API_URL || '';
                    const res = await fetch(`${API_BASE_URL}/api/history`, {
                        headers: { Authorization: `Bearer ${token}` },
                    })
                    const data = await res.json()
                    setHistory(data.history || [])
                } catch {
                    setHistory([])
                }
            })()
    }, [user, getToken])

    const addToHistory = useCallback((entry) => {
        setHistory(prev => [...prev, entry])
        setActiveHistoryIndex(null)
    }, [])

    const deleteHistoryItem = useCallback(async (historyId) => {
        try {
            const token = await getToken()
            const API_BASE_URL = import.meta.env.VITE_API_URL || '';
            const res = await fetch(`${API_BASE_URL}/api/history/${historyId}`, {
                method: 'DELETE',
                headers: { Authorization: `Bearer ${token}` },
            })
            if (res.ok) {
                setHistory(prev => prev.filter(h => h.id !== historyId))
                if (result && result.id === historyId) {
                    setResult(null)
                }
            }
        } catch (err) {
            console.error('Failed to delete history item:', err)
        }
    }, [getToken, result])

    const handleHistoryClick = useCallback((entry, index) => {
        setResult({
            answer: entry.answer,
            mode: entry.mode,
            plan: entry.plan,
            sources: entry.sources,
            grounded: true,
        })
        setActiveHistoryIndex(index)
    }, [])

    if (loading) {
        return (
            <div className="login-page">
                <div className="spinner" style={{ width: 32, height: 32 }} />
            </div>
        )
    }

    if (!user) {
        return <Navigate to="/login" />
    }

    return (
        <div className="app-layout">
            {/* Mobile Header */}
            <div className="mobile-app-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <button className="mobile-icon-btn" onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}>
                        {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
                    </button>
                    <div className="mobile-brand">
                        <img src="/logo.png" alt="PaperMind Logo" style={{ height: 24, width: 'auto' }} />
                        <span className="logo-gradient" style={{ fontSize: '1.2rem', fontWeight: 800 }}>PaperMind</span>
                    </div>
                </div>
                <button className="mobile-icon-btn" onClick={() => logout()} title="Sign out">
                    <LogOut size={20} />
                </button>
            </div>

            {/* Mobile Sidebar Overlay */}
            {isMobileMenuOpen && (
                <div className="mobile-sidebar-overlay" onClick={() => setIsMobileMenuOpen(false)} />
            )}

            <div className={`sidebar-container ${isMobileMenuOpen ? 'open' : ''}`}>
                <Sidebar
                    history={history}
                    activeHistoryIndex={activeHistoryIndex}
                    onHistoryClick={handleHistoryClick}
                    onCloseMobile={() => setIsMobileMenuOpen(false)}
                    onDeleteHistory={deleteHistoryItem}
                />
            </div>

            <main className="main-content">
                <Routes>
                    <Route path="/" element={
                        <ResearchPage
                            result={result}
                            setResult={setResult}
                            history={history}
                            addToHistory={addToHistory}
                            setActiveHistoryIndex={setActiveHistoryIndex}
                        />
                    } />
                    <Route path="/papers" element={<PapersPage />} />
                    <Route path="/graph" element={<GraphPage />} />
                    <Route path="/memory" element={<MemoryPage />} />
                    <Route path="/admin" element={<AdminPage />} />
                    <Route path="*" element={<Navigate to="/app" />} />
                </Routes>
            </main>
        </div>
    )
}

export default function App() {
    return (
        <AuthProvider>
            <Routes>
                <Route path="/" element={<LandingPage />} />
                <Route path="/login" element={<LoginPage />} />
                <Route path="/app/*" element={<ProtectedApp />} />
                <Route path="*" element={<Navigate to="/" />} />
            </Routes>
        </AuthProvider>
    )
}

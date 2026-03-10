import { useState, useEffect, useCallback } from 'react'
import { Routes, Route, Navigate, useNavigate } from 'react-router-dom'
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
    const { user, loading, getToken } = useAuth()
    const [result, setResult] = useState(null)
    const [history, setHistory] = useState([])
    const [activeHistoryIndex, setActiveHistoryIndex] = useState(null)

    // Load history from backend on login
    useEffect(() => {
        if (!user) return
            ; (async () => {
                try {
                    const token = await getToken()
                    const res = await fetch('/api/history', {
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
            <Sidebar
                history={history}
                activeHistoryIndex={activeHistoryIndex}
                onHistoryClick={handleHistoryClick}
            />
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

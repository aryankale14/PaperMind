import { useState, useEffect, useCallback } from 'react'
import { Upload, FileText, Trash2, AlertTriangle, Plus, RefreshCw, Search } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'

export default function PapersPage() {
    const [papers, setPapers] = useState([])
    const [uploading, setUploading] = useState(false)
    const [uploadMsg, setUploadMsg] = useState('')
    const [dragover, setDragover] = useState(false)
    const [showResetModal, setShowResetModal] = useState(false)
    const [resetting, setResetting] = useState(false)
    const [search, setSearch] = useState('')
    const { getToken } = useAuth()

    const fetchPapers = useCallback(async () => {
        try {
            const token = await getToken()
            const res = await fetch('/api/papers', {
                headers: { Authorization: `Bearer ${token}` },
            })
            const data = await res.json()
            setPapers(data.papers || [])
        } catch {
            setPapers([])
        }
    }, [getToken])

    useEffect(() => { fetchPapers() }, [fetchPapers])

    const handleUpload = async (file) => {
        if (!file || !file.name.toLowerCase().endsWith('.pdf')) {
            setUploadMsg('Only PDF files are supported')
            return
        }
        setUploading(true)
        setUploadMsg(`Uploading ${file.name}...`)

        const formData = new FormData()
        formData.append('file', file)

        try {
            const token = await getToken()
            const res = await fetch('/api/upload', {
                method: 'POST',
                headers: { Authorization: `Bearer ${token}` },
                body: formData,
            })
            const data = await res.json()

            if (data.status === 'uploaded_and_indexed') {
                setUploadMsg(`✅ ${file.name} uploaded and indexed successfully!`)
            } else {
                setUploadMsg(`⚠️ Uploaded but indexing had an issue: ${data.index_error || 'unknown'}`)
            }
            fetchPapers()
        } catch (err) {
            setUploadMsg(`❌ Upload failed: ${err.message}`)
        } finally {
            setUploading(false)
            setTimeout(() => setUploadMsg(''), 5000)
        }
    }

    const handleDrop = (e) => {
        e.preventDefault()
        setDragover(false)
        const file = e.dataTransfer.files[0]
        if (file) handleUpload(file)
    }

    const handleFileInput = (e) => {
        const file = e.target.files[0]
        if (file) handleUpload(file)
    }

    const handleReset = async () => {
        setResetting(true)
        try {
            const token = await getToken()
            const res = await fetch('/api/reset', {
                method: 'POST',
                headers: { Authorization: `Bearer ${token}` },
            })
            const data = await res.json()
            if (data.status === 'reset_complete') {
                setPapers([])
                setShowResetModal(false)
            } else {
                alert('Partial reset: ' + JSON.stringify(data.errors))
                setShowResetModal(false)
                fetchPapers()
            }
        } catch (err) {
            alert('Reset failed: ' + err.message)
        } finally {
            setResetting(false)
        }
    }

    const filteredPapers = papers.filter((p) =>
        p.name.toLowerCase().includes(search.toLowerCase())
    )

    return (
        <div className="page-container fade-in">
            <div className="header-actions">
                <div className="page-header" style={{ marginBottom: 0 }}>
                    <h2>Research Papers</h2>
                    <p>Upload PDFs to build your research corpus. Manage your research session data.</p>
                </div>
                <div className="session-actions">
                    <button className="btn btn-danger" onClick={() => setShowResetModal(true)} id="new-session-btn">
                        <RefreshCw size={15} />
                        New Session
                    </button>
                    <button className="btn btn-danger" onClick={() => setShowResetModal(true)} id="delete-session-btn">
                        <Trash2 size={15} />
                        Delete Session
                    </button>
                </div>
            </div>

            {/* Upload Zone */}
            <div
                className={`upload-zone ${dragover ? 'dragover' : ''}`}
                style={{ marginTop: 28 }}
                onDragOver={(e) => { e.preventDefault(); setDragover(true) }}
                onDragLeave={() => setDragover(false)}
                onDrop={handleDrop}
                onClick={() => !uploading && document.getElementById('file-upload-input').click()}
                id="upload-zone"
            >
                <div className="upload-icon">
                    <Upload size={22} />
                </div>
                <h3>{uploading ? 'Processing...' : 'Drop a PDF here or click to upload'}</h3>
                <p>Research papers will be chunked, embedded, and indexed automatically</p>
                {uploadMsg && <div className="upload-progress">{uploadMsg}</div>}
                <input
                    id="file-upload-input"
                    type="file"
                    accept=".pdf"
                    style={{ display: 'none' }}
                    onChange={handleFileInput}
                    disabled={uploading}
                />
            </div>

            {/* Search */}
            {papers.length > 0 && (
                <div className="search-box" style={{ marginTop: 24 }}>
                    <Search size={16} />
                    <input
                        type="text"
                        placeholder="Search papers..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        id="papers-search"
                    />
                </div>
            )}

            {/* Papers Grid */}
            {filteredPapers.length > 0 ? (
                <div className="papers-grid">
                    {filteredPapers.map((paper, i) => (
                        <div key={i} className="paper-card slide-up" style={{ animationDelay: `${i * 0.05}s` }}>
                            <div className="paper-icon">
                                <FileText size={20} />
                            </div>
                            <h3>{paper.name}</h3>
                            <p className="paper-meta">{paper.filename} · {paper.size_kb} KB</p>
                        </div>
                    ))}
                </div>
            ) : (
                <div className="empty-state" style={{ marginTop: 20 }}>
                    <div className="empty-icon">
                        <FileText size={28} />
                    </div>
                    <h3>No papers yet</h3>
                    <p>Upload research PDFs above to start building your knowledge base</p>
                </div>
            )}

            {/* Reset Confirmation Modal */}
            {showResetModal && (
                <div className="modal-overlay" onClick={() => !resetting && setShowResetModal(false)}>
                    <div className="modal modal-danger slide-up" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-icon">
                            <AlertTriangle size={28} />
                        </div>
                        <h3>Delete All Session Data?</h3>
                        <p>This action is <strong>permanent</strong> and cannot be undone. The following data will be completely erased:</p>
                        <ul className="warning-list">
                            <li>All uploaded research papers (PDFs)</li>
                            <li>FAISS vector embeddings index</li>
                            <li>Entire research memory bank</li>
                            <li>Knowledge graph (nodes & relationships)</li>
                            <li>Research history log</li>
                        </ul>
                        <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                            You will need to re-upload papers and rebuild the index to use the copilot again.
                        </p>
                        <div className="modal-actions" style={{ marginTop: 20 }}>
                            <button
                                className="btn btn-ghost"
                                onClick={() => setShowResetModal(false)}
                                disabled={resetting}
                            >
                                Cancel
                            </button>
                            <button
                                className="btn btn-danger-solid"
                                onClick={handleReset}
                                disabled={resetting}
                                id="confirm-reset-btn"
                            >
                                {resetting ? (
                                    <>
                                        <div className="spinner" style={{ width: 14, height: 14, borderTopColor: 'white' }} />
                                        Deleting...
                                    </>
                                ) : (
                                    <>
                                        <Trash2 size={14} />
                                        I understand, delete everything
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

import React from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { Database, Search, Target, BrainCircuit, Globe, ArrowRight, ChevronDown, Beaker, Library, Code2 } from 'lucide-react'

export default function LandingPage() {
    const { user } = useAuth()
    const navigate = useNavigate()

    const handleGetStarted = () => {
        if (user) {
            navigate('/app')
        } else {
            navigate('/login')
        }
    }

    return (
        <div className="landing-page">
            {/* Header / Nav */}
            <nav className="landing-nav">
                <div className="landing-logo" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <img src="/logo.png" alt="PaperMind Logo" style={{ height: 35, width: 'auto', objectFit: 'contain' }} />
                    <span className="logo-gradient" style={{ fontSize: '1.55rem', fontWeight: 800, letterSpacing: '-0.02em' }}>PaperMind</span>
                </div>
                <div className="landing-nav-actions">
                    {user ? (
                        <button
                            onClick={() => navigate('/app')}
                            className="landing-btn-primary"
                        >
                            Open Dashboard
                        </button>
                    ) : (
                        <button
                            onClick={handleGetStarted}
                            className="landing-btn-secondary"
                        >
                            Sign In
                        </button>
                    )}
                </div>
            </nav>

            {/* Hero Section */}
            <main className="landing-hero">
                <div className="landing-badge">
                    <span className="landing-badge-dot"></span>
                    Announcing Multi-Agent RAG Support
                </div>

                <h1 className="landing-title">
                    Supercharge your <span className="text-gradient">academic research</span>.
                </h1>

                <p className="landing-subtitle">
                    Upload your PDFs and let our autonomous AI agents extract, synthesize, and evaluate vast amounts of literature instantly. Powered by pgvector and Gemini.
                </p>

                <div className="landing-hero-actions">
                    <button
                        onClick={handleGetStarted}
                        className="landing-btn-hero"
                    >
                        {user ? 'Go to Dashboard' : 'Start Researching for Free'}
                        <ArrowRight size={18} />
                    </button>
                </div>

                <div className="landing-scroll-indicator">
                    <span className="text-slate-500 text-sm mb-2">Scroll to explore</span>
                    <ChevronDown size={24} className="text-slate-600 animate-bounce-slow" />
                </div>
            </main>

            {/* Features Banners */}
            <section className="landing-features-section">
                <div className="landing-features-container">
                    <h2 className="landing-features-title">How it transforms your workflow</h2>

                    <div className="landing-features-grid">
                        <FeatureCard
                            icon={<Database color="var(--accent-primary)" size={24} />}
                            title="Unlimited Paper Tracking"
                            description="Drag and drop your PDFs. We instantly chunk, embed, and index them into a secure pgvector cloud database."
                        />
                        <FeatureCard
                            icon={<Search color="#7c7cff" size={24} />}
                            title="Deep Multi-Hop Searching"
                            description="Ask complex questions. Our multi-agent system plans sub-queries, evaluates evidence, and builds comprehensive answers."
                        />
                        <FeatureCard
                            icon={<Globe color="#a855f7" size={24} />}
                            title="Persistent Knowledge Graph"
                            description="Visualize connections between authors, domains, and papers. Turn raw literature into a navigable map of knowledge."
                        />
                    </div>
                </div>
            </section>

            {/* Target Audience Section */}
            <section className="landing-audience-section">
                <div className="landing-features-container">
                    <h2 className="landing-features-title">Engineered for deep work</h2>

                    <div className="landing-audience-grid">
                        <AudienceCard
                            icon={<Library size={24} className="text-blue-400" />}
                            title="Ph.D. Researchers"
                            description="Stop drowning in hundreds of unread PDFs. Instantly query your entire literature review library to find exact contradictions, methodologies, and supporting evidence."
                        />
                        <AudienceCard
                            icon={<Beaker size={24} className="text-emerald-400" />}
                            title="R&D Scientists"
                            description="Accelerate discovery. Cross-reference whitepapers, lab reports, and technical manuals to synthesize state-of-the-art findings in seconds."
                        />
                        <AudienceCard
                            icon={<Code2 size={24} className="text-indigo-400" />}
                            title="Technical Analysts"
                            description="Analyze complex multi-document specifications without missing critical details. The knowledge graph tracks exactly where every piece of information originated."
                        />
                    </div>
                </div>
            </section>

            {/* Footer */}
            <footer className="landing-footer">
                <p>© {new Date().getFullYear()} PaperMind. Built for the academic community.</p>
            </footer>
        </div>
    )
}

function FeatureCard({ icon, title, description }) {
    return (
        <div className="landing-feature-card">
            <div className="landing-feature-icon">
                {icon}
            </div>
            <h3 className="landing-feature-title">{title}</h3>
            <p className="landing-feature-desc">
                {description}
            </p>
        </div>
    )
}

function AudienceCard({ icon, title, description }) {
    return (
        <div className="landing-audience-card">
            <div className="landing-audience-header">
                {icon}
                <h3 className="text-lg font-semibold text-white">{title}</h3>
            </div>
            <p className="text-slate-400 leading-relaxed text-sm">
                {description}
            </p>
        </div>
    )
}

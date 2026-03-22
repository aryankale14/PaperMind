import React from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import {
    Database, Search, Globe, ArrowRight, ChevronDown,
    Beaker, Library, Code2, Shield, BarChart3,
    Brain, GitBranch, Layers, CheckCircle2,
    Mail, User, MessageSquare,
    Sparkles, Target, BookOpen, Network, Cpu
} from 'lucide-react'



/* ─── Floating Particle Background ─── */
function ParticleField() {
    return (
        <div className="landing-particles">
            {Array.from({ length: 30 }).map((_, i) => (
                <div
                    key={i}
                    className="landing-particle"
                    style={{
                        left: `${Math.random() * 100}%`,
                        top: `${Math.random() * 100}%`,
                        animationDelay: `${Math.random() * 6}s`,
                        animationDuration: `${4 + Math.random() * 6}s`,
                        width: `${2 + Math.random() * 3}px`,
                        height: `${2 + Math.random() * 3}px`,
                    }}
                />
            ))}
        </div>
    )
}

export default function LandingPage() {
    const { user } = useAuth()
    const navigate = useNavigate()

    const handleGetStarted = () => {
        navigate(user ? '/app' : '/login')
    }

    return (
        <div className="landing-page">
            <ParticleField />

            {/* ────── Navigation ────── */}
            <nav className="landing-nav">
                <div className="landing-logo" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <img src="/logo.png" alt="PaperMind Logo" style={{ height: 35, width: 'auto', objectFit: 'contain' }} />
                    <span className="logo-gradient" style={{ fontSize: '1.55rem', fontWeight: 800, letterSpacing: '-0.02em' }}>PaperMind</span>
                </div>
                <div className="landing-nav-links">
                    <a href="#features" className="landing-nav-link">Features</a>
                    <a href="#how-it-works" className="landing-nav-link">How It Works</a>
                    <a href="#contact" className="landing-nav-link">Contact</a>
                </div>
                <div className="landing-nav-actions">
                    {user ? (
                        <button onClick={() => navigate('/app')} className="landing-btn-primary">
                            Open Dashboard
                        </button>
                    ) : (
                        <>
                            <button onClick={handleGetStarted} className="landing-btn-secondary">Sign In</button>
                            <button onClick={handleGetStarted} className="landing-btn-primary">Get Started Free</button>
                        </>
                    )}
                </div>
            </nav>

            {/* ────── Hero ────── */}
            <main className="landing-hero">
                <div className="landing-badge">
                    <span className="landing-badge-dot" />
                    Powered by Multi-Agent RAG + Gemini AI
                </div>

                <h1 className="landing-title">
                    Research at the<br />
                    <span className="text-gradient">speed of thought.</span>
                </h1>

                <p className="landing-subtitle">
                    Upload your PDFs and let autonomous AI agents extract, synthesize, and evaluate 
                    vast amounts of literature in seconds — not weeks.
                </p>

                <div className="landing-hero-actions">
                    <button onClick={handleGetStarted} className="landing-btn-hero">
                        {user ? 'Go to Dashboard' : 'Start Researching — It\'s Free'}
                        <ArrowRight size={18} />
                    </button>
                    <a href="#how-it-works" className="landing-btn-ghost">
                        See How It Works
                        <ChevronDown size={16} />
                    </a>
                </div>

                <div className="landing-hero-visual">
                    <div className="landing-hero-glow" />
                    <div className="landing-hero-mockup">
                        <div className="mockup-bar">
                            <span className="mockup-dot red" />
                            <span className="mockup-dot yellow" />
                            <span className="mockup-dot green" />
                            <span className="mockup-url">papermind.app/research</span>
                        </div>
                        <div className="mockup-content">
                            <div className="mockup-sidebar">
                                <div className="mockup-sidebar-item active" />
                                <div className="mockup-sidebar-item" />
                                <div className="mockup-sidebar-item" />
                                <div className="mockup-sidebar-item" />
                            </div>
                            <div className="mockup-main">
                                <div className="mockup-line w80" />
                                <div className="mockup-line w60" />
                                <div className="mockup-block" />
                                <div className="mockup-line w90" />
                                <div className="mockup-line w70" />
                            </div>
                        </div>
                    </div>
                </div>
            </main>



            {/* ────── Features ────── */}
            <section className="landing-features-section" id="features">
                <div className="landing-features-container">
                    <div className="landing-section-label">
                        <Sparkles size={14} />
                        <span>Core Capabilities</span>
                    </div>
                    <h2 className="landing-features-title">Everything you need to<br />dominate your literature review</h2>

                    <div className="landing-features-grid">
                        <FeatureCard
                            icon={<Database color="#4d94ff" size={24} />}
                            title="Smart Paper Indexing"
                            description="Drag-and-drop PDFs. We chunk, embed, and index them into pgvector in real-time for lightning-fast semantic search."
                            tag="Storage"
                        />
                        <FeatureCard
                            icon={<Search color="#7c7cff" size={24} />}
                            title="Multi-Hop Deep Search"
                            description="Ask complex questions. Our multi-agent system plans sub-queries, evaluates evidence, and builds comprehensive answers with citations."
                            tag="Search"
                        />
                        <FeatureCard
                            icon={<Globe color="#a855f7" size={24} />}
                            title="Knowledge Graph Explorer"
                            description="Visualize connections between authors, domains, and papers. Navigate your entire research landscape in an interactive graph."
                            tag="Visualization"
                        />
                        <FeatureCard
                            icon={<Brain color="#ec4899" size={24} />}
                            title="Research Memory"
                            description="The system remembers your past queries and builds context over time. Every session gets smarter than the last."
                            tag="Intelligence"
                        />
                        <FeatureCard
                            icon={<Shield color="#34d399" size={24} />}
                            title="Citation Grounding"
                            description="Every answer is grounded with exact paper titles and page numbers. Verify any claim instantly — no hallucinations."
                            tag="Trust"
                        />
                        <FeatureCard
                            icon={<BarChart3 color="#f59e0b" size={24} />}
                            title="Quality Evaluation"
                            description="AI evaluates answer confidence, research coverage, and depth score in real-time, so you know exactly how thorough your results are."
                            tag="Analytics"
                        />
                    </div>
                </div>
            </section>

            {/* ────── How It Works ────── */}
            <section className="landing-pipeline-section" id="how-it-works">
                <div className="landing-features-container">
                    <div className="landing-section-label">
                        <Layers size={14} />
                        <span>Architecture</span>
                    </div>
                    <h2 className="landing-features-title">A 7-agent research pipeline<br />working in harmony</h2>

                    <div className="landing-pipeline-grid">
                        <PipelineStep number="01" title="Complexity Classification" desc="Determines if your query needs single-hop or multi-hop reasoning." icon={<Target size={20} />} />
                        <PipelineStep number="02" title="Query Planning" desc="Breaks complex questions into strategic sub-queries for targeted evidence collection." icon={<GitBranch size={20} />} />
                        <PipelineStep number="03" title="Hybrid Retrieval" desc="Combines pgvector semantic search + BM25 keyword matching for maximum recall." icon={<Search size={20} />} />
                        <PipelineStep number="04" title="Evidence Synthesis" desc="Gemini AI synthesizes evidence from multiple papers into a coherent, cited answer." icon={<BookOpen size={20} />} />
                        <PipelineStep number="05" title="Quality Evaluation" desc="AI evaluator scores factual accuracy, research depth, and citation coverage." icon={<CheckCircle2 size={20} />} />
                        <PipelineStep number="06" title="Knowledge Extraction" desc="Extracts entities and relationships to continuously grow your knowledge graph." icon={<Network size={20} />} />
                        <PipelineStep number="07" title="Memory Storage" desc="Stores research insights into persistent memory for future context enrichment." icon={<Brain size={20} />} />
                    </div>
                </div>
            </section>

            {/* ────── Tech Stack ────── */}
            <section className="landing-tech-section">
                <div className="landing-features-container">
                    <div className="landing-section-label">
                        <Cpu size={14} />
                        <span>Technology</span>
                    </div>
                    <h2 className="landing-features-title">Built with cutting-edge<br />AI infrastructure</h2>

                    <div className="landing-tech-grid">
                        <TechBadge name="Google Gemini" category="LLM" />
                        <TechBadge name="Groq (LLaMA)" category="Agent Brain" />
                        <TechBadge name="pgvector" category="Vector DB" />
                        <TechBadge name="LangChain" category="Orchestration" />
                        <TechBadge name="FastAPI" category="Backend" />
                        <TechBadge name="React" category="Frontend" />
                        <TechBadge name="Firebase Auth" category="Auth" />
                        <TechBadge name="Supabase" category="Database" />
                    </div>
                </div>
            </section>

            {/* ────── Audience ────── */}
            <section className="landing-audience-section">
                <div className="landing-features-container">
                    <div className="landing-section-label">
                        <User size={14} />
                        <span>Who It's For</span>
                    </div>
                    <h2 className="landing-features-title">Engineered for people<br />who do deep work</h2>

                    <div className="landing-audience-grid">
                        <AudienceCard
                            icon={<Library size={24} color="#60a5fa" />}
                            title="Ph.D. Researchers"
                            description="Stop drowning in hundreds of unread PDFs. Query your entire literature library to find exact contradictions, methodologies, and supporting evidence."
                        />
                        <AudienceCard
                            icon={<Beaker size={24} color="#34d399" />}
                            title="R&D Scientists"
                            description="Cross-reference whitepapers, lab reports, and technical manuals to synthesize state-of-the-art findings in seconds, not days."
                        />
                        <AudienceCard
                            icon={<Code2 size={24} color="#818cf8" />}
                            title="Technical Analysts"
                            description="Analyze complex multi-document specifications without missing critical details. The knowledge graph tracks where every piece of information originated."
                        />
                    </div>
                </div>
            </section>



            {/* ────── CTA Banner ────── */}
            <section className="landing-cta-section">
                <div className="landing-cta-inner">
                    <div className="landing-cta-glow" />
                    <h2 className="landing-cta-title">Ready to transform<br />your research workflow?</h2>
                    <p className="landing-cta-subtitle">
                        Join thousands of researchers who are already using PaperMind 
                        to accelerate their academic work.
                    </p>
                    <button onClick={handleGetStarted} className="landing-btn-hero">
                        {user ? 'Go to Dashboard' : 'Get Started for Free'}
                        <ArrowRight size={18} />
                    </button>
                </div>
            </section>

            {/* ────── Contact ────── */}
            <section className="landing-feedback-section" id="contact">
                <div className="landing-features-container">
                    <div className="landing-section-label">
                        <MessageSquare size={14} />
                        <span>Get in Touch</span>
                    </div>
                    <h2 className="landing-features-title">Built by a passionate developer</h2>

                    <div className="landing-contact-card">
                        <div className="landing-feedback-info">
                            <h3 className="landing-feedback-info-title">Let's Connect</h3>
                            <p className="landing-feedback-info-desc">
                                Have a question, feedback, or just want to say hello? 
                                I'd love to hear from you. Feel free to reach out anytime.
                            </p>
                            <div className="landing-feedback-contact">
                                <div className="landing-feedback-contact-item">
                                    <div className="landing-feedback-contact-icon">
                                        <User size={18} />
                                    </div>
                                    <div>
                                        <span className="landing-feedback-contact-label">Developer</span>
                                        <span className="landing-feedback-contact-value">Aaryan Kale</span>
                                    </div>
                                </div>
                                <div className="landing-feedback-contact-item">
                                    <div className="landing-feedback-contact-icon">
                                        <Mail size={18} />
                                    </div>
                                    <div>
                                        <span className="landing-feedback-contact-label">Email</span>
                                        <a href="mailto:aryankale1410@gmail.com" className="landing-feedback-contact-value" style={{ textDecoration: 'none' }}>aryankale1410@gmail.com</a>
                                    </div>
                                </div>
                            </div>
                            <p className="landing-feedback-quote">
                                "Built with ❤️ as a passion project to make academic research more accessible and efficient for everyone."
                            </p>
                        </div>
                    </div>
                </div>
            </section>

            {/* ────── Footer ────── */}
            <footer className="landing-footer-premium">
                <div className="landing-footer-inner">
                    <div className="landing-footer-brand">
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
                            <img src="/logo.png" alt="PaperMind" style={{ height: 28 }} />
                            <span className="logo-gradient" style={{ fontSize: '1.3rem', fontWeight: 800 }}>PaperMind</span>
                        </div>
                        <p className="landing-footer-tagline">AI-powered research intelligence platform for academics, scientists, and analysts.</p>
                    </div>
                    <div className="landing-footer-links-group">
                        <h4>Product</h4>
                        <a href="#features">Features</a>
                        <a href="#how-it-works">How It Works</a>
                    </div>
                    <div className="landing-footer-links-group">
                        <h4>Connect</h4>
                        <a href="#contact">Contact</a>
                        <a href="mailto:aryankale1410@gmail.com">Email</a>
                    </div>
                </div>
                <div className="landing-footer-bottom">
                    <p>© {new Date().getFullYear()} PaperMind. Built by <strong>Aaryan Kale</strong>. All rights reserved.</p>
                </div>
            </footer>
        </div>
    )
}

/* ─── Sub-Components ─── */

function FeatureCard({ icon, title, description, tag }) {
    return (
        <div className="landing-feature-card">
            <div className="landing-feature-card-top">
                <div className="landing-feature-icon">{icon}</div>
                {tag && <span className="landing-feature-tag">{tag}</span>}
            </div>
            <h3 className="landing-feature-title">{title}</h3>
            <p className="landing-feature-desc">{description}</p>
        </div>
    )
}

function PipelineStep({ number, title, desc, icon }) {
    return (
        <div className="landing-pipeline-step">
            <div className="landing-pipeline-number">{number}</div>
            <div className="landing-pipeline-icon">{icon}</div>
            <h3 className="landing-pipeline-step-title">{title}</h3>
            <p className="landing-pipeline-step-desc">{desc}</p>
        </div>
    )
}

function TechBadge({ name, category }) {
    return (
        <div className="landing-tech-badge">
            <span className="landing-tech-name">{name}</span>
            <span className="landing-tech-category">{category}</span>
        </div>
    )
}

function AudienceCard({ icon, title, description }) {
    return (
        <div className="landing-audience-card">
            <div className="landing-audience-header">
                {icon}
                <h3>{title}</h3>
            </div>
            <p className="landing-audience-desc">{description}</p>
        </div>
    )
}


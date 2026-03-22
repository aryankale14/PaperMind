import os

CSS = """

/* ═══════════════════════════════════════════════════════════
   Landing Page Styles
   ═══════════════════════════════════════════════════════════ */

.landing-page {
  background-color: var(--bg-primary);
  min-height: 100vh;
  color: var(--text-primary);
  display: flex;
  flex-direction: column;
}

.landing-nav {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 24px 32px;
  max-width: 1280px;
  margin: 0 auto;
  width: 100%;
  border-bottom: 1px solid var(--border-subtle);
}

.landing-logo {
  display: flex;
  align-items: center;
  gap: 12px;
}

.landing-logo-icon {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  background: var(--accent-gradient);
  display: flex;
  align-items: center;
  justify-content: center;
}

.landing-logo-text {
  color: #fff;
  font-weight: 600;
  font-size: 1.25rem;
  letter-spacing: -0.02em;
}

.landing-nav-actions {
  display: flex;
  align-items: center;
  gap: 16px;
}

.landing-btn-primary {
  background: var(--accent-primary);
  color: #fff;
  font-weight: 500;
  font-size: 0.875rem;
  padding: 10px 20px;
  border-radius: 9999px;
  border: none;
  cursor: pointer;
  transition: all 0.2s;
}
.landing-btn-primary:hover {
  background: #3b82f6;
}

.landing-btn-secondary {
  background: transparent;
  color: #fff;
  font-weight: 500;
  font-size: 0.875rem;
  padding: 10px 20px;
  border-radius: 9999px;
  border: 1px solid var(--border-default);
  cursor: pointer;
  transition: all 0.2s;
}
.landing-btn-secondary:hover {
  background: var(--bg-card-hover);
}

.landing-hero {
  max-width: 1280px;
  margin: 96px auto 128px;
  padding: 0 32px;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}

.landing-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 4px 12px;
  border-radius: 9999px;
  background: var(--accent-glow);
  border: 1px solid rgba(77, 148, 255, 0.2);
  color: var(--accent-primary);
  font-size: 0.875rem;
  font-weight: 500;
  margin-bottom: 32px;
}

.landing-badge-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--accent-primary);
  animation: landing-pulse 2s infinite;
}

@keyframes landing-pulse {
  0% { opacity: 1; }
  50% { opacity: 0.5; }
  100% { opacity: 1; }
}

.landing-title {
  font-size: 4.5rem;
  font-weight: 700;
  color: #fff;
  letter-spacing: -0.02em;
  line-height: 1.1;
  margin-bottom: 32px;
  max-width: 900px;
}

.text-gradient {
  background: var(--accent-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.landing-subtitle {
  font-size: 1.25rem;
  color: var(--text-secondary);
  max-width: 670px;
  margin-bottom: 40px;
  line-height: 1.6;
}

.landing-hero-actions {
  display: flex;
  align-items: center;
  gap: 16px;
}

.landing-btn-hero {
  display: flex;
  align-items: center;
  gap: 8px;
  background: #fff;
  color: #000;
  padding: 16px 32px;
  border-radius: 9999px;
  font-weight: 600;
  font-size: 1rem;
  border: none;
  cursor: pointer;
  transition: all 0.2s;
}
.landing-btn-hero:hover {
  background: #f0f0f0;
}

.landing-features-section {
  border-top: 1px solid var(--border-subtle);
  background: rgba(8, 8, 8, 0.4);
}

.landing-features-container {
  max-width: 1280px;
  margin: 0 auto;
  padding: 96px 32px;
}

.landing-features-title {
  font-size: 1.875rem;
  font-weight: 700;
  text-align: center;
  margin-bottom: 64px;
}

.landing-features-grid {
  display: grid;
  grid-template-columns: repeat(1, 1fr);
  gap: 32px;
}

@media (min-width: 768px) {
  .landing-features-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

.landing-feature-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-xl);
  padding: 32px;
  transition: border-color 0.2s;
}
.landing-feature-card:hover {
  border-color: var(--border-hover);
}

.landing-feature-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.05);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 24px;
}

.landing-feature-title {
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 12px;
}

.landing-feature-desc {
  color: var(--text-secondary);
  line-height: 1.6;
  font-size: 0.875rem;
}

.landing-footer {
  border-top: 1px solid var(--border-subtle);
  padding: 32px;
  text-align: center;
  color: var(--text-muted);
  font-size: 0.875rem;
}
"""

with open(r"c:\Users\AARYAN KALE\OneDrive\Documents\Aaryan\Programs\ai-research-copilot\frontend\src\index.css", "a", encoding="utf-8") as f:
    f.write(CSS)
    
print("CSS injected successfully")

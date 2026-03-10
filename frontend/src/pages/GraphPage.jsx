import { useState, useEffect, useRef, useCallback } from 'react'
import { Share2, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'

export default function GraphPage() {
    const [graph, setGraph] = useState({ nodes: [], edges: [] })
    const [loading, setLoading] = useState(true)
    const canvasRef = useRef(null)
    const containerRef = useRef(null)
    const animFrameRef = useRef(null)
    const nodesRef = useRef([])
    const [hoveredNode, setHoveredNode] = useState(null)
    const [dimensions, setDimensions] = useState({ width: 800, height: 600 })

    // Zoom & pan state
    const [scale, setScale] = useState(1)
    const [offset, setOffset] = useState({ x: 0, y: 0 })
    const dragRef = useRef({ dragging: false, startX: 0, startY: 0, startOffX: 0, startOffY: 0 })
    const { getToken } = useAuth()

    useEffect(() => {
        const fetchGraph = async () => {
            try {
                const token = await getToken()
                const res = await fetch('/api/graph', {
                    headers: { Authorization: `Bearer ${token}` },
                })
                const data = await res.json()
                setGraph(data)
            } catch {
                setGraph({ nodes: [], edges: [] })
            } finally {
                setLoading(false)
            }
        }
        fetchGraph()
    }, [])

    // Observe container size
    useEffect(() => {
        if (!containerRef.current) return
        const observer = new ResizeObserver((entries) => {
            for (const entry of entries) {
                setDimensions({
                    width: entry.contentRect.width,
                    height: entry.contentRect.height,
                })
            }
        })
        observer.observe(containerRef.current)
        return () => observer.disconnect()
    }, [])

    // Initialize node positions — spread out more
    useEffect(() => {
        if (graph.nodes.length === 0) return
        const { width, height } = dimensions
        const cx = width / 2
        const cy = height / 2

        nodesRef.current = graph.nodes.map((name, i) => {
            const angle = (2 * Math.PI * i) / graph.nodes.length
            const radius = Math.min(width, height) * 0.38
            return {
                name,
                x: cx + radius * Math.cos(angle) + (Math.random() - 0.5) * 60,
                y: cy + radius * Math.sin(angle) + (Math.random() - 0.5) * 60,
                vx: 0,
                vy: 0,
            }
        })

        // Reset zoom/pan when graph changes
        setScale(1)
        setOffset({ x: 0, y: 0 })
    }, [graph.nodes, dimensions])

    // Improved force-directed simulation — more spacing
    const simulate = useCallback(() => {
        const nodes = nodesRef.current
        if (nodes.length === 0) return

        const { width, height } = dimensions
        const cx = width / 2
        const cy = height / 2

        for (let iter = 0; iter < 3; iter++) {
            // Stronger repulsion for less congestion
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    let dx = nodes[j].x - nodes[i].x
                    let dy = nodes[j].y - nodes[i].y
                    let dist = Math.sqrt(dx * dx + dy * dy) || 1
                    let force = 3000 / (dist * dist)
                    let fx = (dx / dist) * force
                    let fy = (dy / dist) * force
                    nodes[i].vx -= fx
                    nodes[i].vy -= fy
                    nodes[j].vx += fx
                    nodes[j].vy += fy
                }
            }

            // Attraction (edges) — longer ideal distance
            for (const edge of graph.edges) {
                const si = graph.nodes.indexOf(edge.subject)
                const oi = graph.nodes.indexOf(edge.object)
                if (si < 0 || oi < 0) continue
                let dx = nodes[oi].x - nodes[si].x
                let dy = nodes[oi].y - nodes[si].y
                let dist = Math.sqrt(dx * dx + dy * dy) || 1
                let force = (dist - 200) * 0.008
                let fx = (dx / dist) * force
                let fy = (dy / dist) * force
                nodes[si].vx += fx
                nodes[si].vy += fy
                nodes[oi].vx -= fx
                nodes[oi].vy -= fy
            }

            // Center gravity
            for (const node of nodes) {
                node.vx += (cx - node.x) * 0.001
                node.vy += (cy - node.y) * 0.001
            }

            // Apply velocity with damping
            for (const node of nodes) {
                node.vx *= 0.82
                node.vy *= 0.82
                node.x += node.vx
                node.y += node.vy
                node.x = Math.max(100, Math.min(width - 100, node.x))
                node.y = Math.max(60, Math.min(height - 60, node.y))
            }
        }
    }, [graph, dimensions])

    // Draw loop with zoom/pan
    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')

        const draw = () => {
            simulate()
            const nodes = nodesRef.current
            const { width, height } = dimensions

            canvas.width = width * window.devicePixelRatio
            canvas.height = height * window.devicePixelRatio
            canvas.style.width = width + 'px'
            canvas.style.height = height + 'px'
            ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0)

            ctx.clearRect(0, 0, width, height)

            // Apply zoom & pan
            ctx.save()
            ctx.translate(offset.x, offset.y)
            ctx.scale(scale, scale)

            // Draw edges
            for (const edge of graph.edges) {
                const si = graph.nodes.indexOf(edge.subject)
                const oi = graph.nodes.indexOf(edge.object)
                if (si < 0 || oi < 0 || !nodes[si] || !nodes[oi]) continue

                ctx.beginPath()
                ctx.moveTo(nodes[si].x, nodes[si].y)
                ctx.lineTo(nodes[oi].x, nodes[oi].y)
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)'
                ctx.lineWidth = 1
                ctx.stroke()

                // Edge label
                const mx = (nodes[si].x + nodes[oi].x) / 2
                const my = (nodes[si].y + nodes[oi].y) / 2
                ctx.font = '10px Inter'
                ctx.fillStyle = 'rgba(148, 163, 184, 0.5)'
                ctx.textAlign = 'center'
                ctx.fillText(edge.relation, mx, my - 6)
            }

            // Draw nodes
            for (let i = 0; i < nodes.length; i++) {
                const node = nodes[i]
                const isHovered = hoveredNode === i

                // Glow
                if (isHovered) {
                    ctx.beginPath()
                    ctx.arc(node.x, node.y, 22, 0, Math.PI * 2)
                    ctx.fillStyle = 'rgba(77, 148, 255, 0.12)'
                    ctx.fill()
                }

                // Node circle — bigger
                ctx.beginPath()
                ctx.arc(node.x, node.y, isHovered ? 10 : 7, 0, Math.PI * 2)
                ctx.fillStyle = isHovered ? '#6cb4ff' : '#4d94ff'
                ctx.fill()

                // Label
                ctx.font = `${isHovered ? '13px' : '11px'} Inter`
                ctx.fillStyle = isHovered ? '#f0f2f5' : '#94a3b8'
                ctx.textAlign = 'center'
                ctx.fillText(node.name, node.x, node.y + (isHovered ? 24 : 22))
            }

            ctx.restore()

            animFrameRef.current = requestAnimationFrame(draw)
        }

        draw()
        return () => {
            if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
        }
    }, [graph, dimensions, hoveredNode, simulate, scale, offset])

    // Mouse interaction with zoom/pan awareness
    const handleMouseMove = useCallback((e) => {
        const canvas = canvasRef.current
        if (!canvas) return

        // Handle dragging for pan
        if (dragRef.current.dragging) {
            setOffset({
                x: dragRef.current.startOffX + (e.clientX - dragRef.current.startX),
                y: dragRef.current.startOffY + (e.clientY - dragRef.current.startY),
            })
            return
        }

        const rect = canvas.getBoundingClientRect()
        const mx = (e.clientX - rect.left - offset.x) / scale
        const my = (e.clientY - rect.top - offset.y) / scale
        const nodes = nodesRef.current

        let found = -1
        for (let i = 0; i < nodes.length; i++) {
            const dx = nodes[i].x - mx
            const dy = nodes[i].y - my
            if (Math.sqrt(dx * dx + dy * dy) < 24) {
                found = i
                break
            }
        }
        setHoveredNode(found >= 0 ? found : null)
    }, [scale, offset])

    const handleMouseDown = useCallback((e) => {
        if (hoveredNode !== null) return // don't pan when hovering a node
        dragRef.current = {
            dragging: true,
            startX: e.clientX,
            startY: e.clientY,
            startOffX: offset.x,
            startOffY: offset.y,
        }
    }, [offset, hoveredNode])

    const handleMouseUp = useCallback(() => {
        dragRef.current.dragging = false
    }, [])

    const handleWheel = useCallback((e) => {
        e.preventDefault()
        const delta = e.deltaY > 0 ? 0.9 : 1.1
        setScale(prev => Math.max(0.3, Math.min(3, prev * delta)))
    }, [])

    const resetView = () => {
        setScale(1)
        setOffset({ x: 0, y: 0 })
    }

    if (loading) {
        return (
            <div className="page-container">
                <div className="page-header">
                    <h2>Knowledge Graph</h2>
                </div>
                <div className="empty-state">
                    <div className="spinner" style={{ margin: '0 auto 16px' }} />
                    <p>Loading graph...</p>
                </div>
            </div>
        )
    }

    return (
        <div className="page-container fade-in">
            <div className="page-header">
                <h2>Knowledge Graph</h2>
                <p>Interactive visualization of extracted concept relationships across research papers</p>
            </div>

            <div className="graph-stats">
                <div className="stat-card">
                    <div className="stat-label">Concepts</div>
                    <div className="stat-value">{graph.nodes.length}</div>
                </div>
                <div className="stat-card">
                    <div className="stat-label">Relationships</div>
                    <div className="stat-value">{graph.edges.length}</div>
                </div>
                {graph.nodes.length > 0 && (
                    <div className="graph-controls">
                        <button className="btn btn-ghost" onClick={() => setScale(s => Math.min(3, s * 1.2))} title="Zoom in">
                            <ZoomIn size={15} />
                        </button>
                        <button className="btn btn-ghost" onClick={() => setScale(s => Math.max(0.3, s * 0.8))} title="Zoom out">
                            <ZoomOut size={15} />
                        </button>
                        <button className="btn btn-ghost" onClick={resetView} title="Reset view">
                            <Maximize2 size={15} />
                        </button>
                        <span className="zoom-label">{Math.round(scale * 100)}%</span>
                    </div>
                )}
            </div>

            {graph.nodes.length > 0 ? (
                <div className="graph-container" ref={containerRef}>
                    <canvas
                        ref={canvasRef}
                        style={{ display: 'block', cursor: dragRef.current.dragging ? 'grabbing' : hoveredNode !== null ? 'pointer' : 'grab' }}
                        onMouseMove={handleMouseMove}
                        onMouseDown={handleMouseDown}
                        onMouseUp={handleMouseUp}
                        onMouseLeave={handleMouseUp}
                        onWheel={handleWheel}
                    />
                </div>
            ) : (
                <div className="empty-state" style={{ marginTop: 60 }}>
                    <div className="empty-icon">
                        <Share2 size={28} />
                    </div>
                    <h3>No graph data yet</h3>
                    <p>Ask research questions to automatically extract concept relationships and build the knowledge graph</p>
                </div>
            )}
        </div>
    )
}

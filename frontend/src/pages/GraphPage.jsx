import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
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
    const tickRef = useRef(0)

    // Zoom & pan state
    const [scale, setScale] = useState(1)
    const [offset, setOffset] = useState({ x: 0, y: 0 })
    const dragRef = useRef({ dragging: false, startX: 0, startY: 0, startOffX: 0, startOffY: 0 })
    const { getToken } = useAuth()

    // Precompute connection counts for node sizing
    const connectionCounts = useMemo(() => {
        const counts = {}
        for (const node of graph.nodes) counts[node] = 0
        for (const edge of graph.edges) {
            if (counts[edge.subject] !== undefined) counts[edge.subject]++
            if (counts[edge.object] !== undefined) counts[edge.object]++
        }
        return counts
    }, [graph])

    useEffect(() => {
        const fetchGraph = async () => {
            try {
                const token = await getToken()
                const API_BASE_URL = import.meta.env.VITE_API_URL || '';
                const res = await fetch(`${API_BASE_URL}/api/graph`, {
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

    // Initialize node positions — spread out in a large circle
    useEffect(() => {
        if (graph.nodes.length === 0) return
        const { width, height } = dimensions
        const cx = width / 2
        const cy = height / 2
        const isMobile = width < 500

        nodesRef.current = graph.nodes.map((name, i) => {
            const angle = (2 * Math.PI * i) / graph.nodes.length
            const radius = Math.min(width, height) * (isMobile ? 0.32 : 0.42)
            const jitter = isMobile ? 30 : 90
            return {
                name,
                x: cx + radius * Math.cos(angle) + (Math.random() - 0.5) * jitter,
                y: cy + radius * Math.sin(angle) + (Math.random() - 0.5) * jitter,
                vx: 0,
                vy: 0,
            }
        })

        tickRef.current = 0
        setScale(1)
        setOffset({ x: 0, y: 0 })
    }, [graph.nodes, dimensions])

    // Improved force-directed simulation
    const simulate = useCallback(() => {
        const nodes = nodesRef.current
        if (nodes.length === 0) return

        tickRef.current++

        const { width, height } = dimensions
        const cx = width / 2
        const cy = height / 2

        // Cooling: slow down forces over time for a stable layout
        const cooling = Math.max(0.15, 1 - tickRef.current * 0.003)

        for (let iter = 0; iter < 3; iter++) {
            // Strong repulsion between ALL node pairs
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    let dx = nodes[j].x - nodes[i].x
                    let dy = nodes[j].y - nodes[i].y
                    let dist = Math.sqrt(dx * dx + dy * dy) || 1
                    // Much stronger repulsion to prevent clustering
                    let force = (8000 * cooling) / (dist * dist)
                    let fx = (dx / dist) * force
                    let fy = (dy / dist) * force
                    nodes[i].vx -= fx
                    nodes[i].vy -= fy
                    nodes[j].vx += fx
                    nodes[j].vy += fy
                }
            }

            // Very gentle central gravity (just enough to keep them on screen)
            for (let i = 0; i < nodes.length; i++) {
                nodes[i].vx += (cx - nodes[i].x) * 0.008 * cooling
                nodes[i].vy += (cy - nodes[i].y) * 0.008 * cooling
            }

            // Attraction along edges — connected nodes pull toward each other
            for (const edge of graph.edges) {
                const si = graph.nodes.indexOf(edge.subject)
                const oi = graph.nodes.indexOf(edge.object)
                if (si < 0 || oi < 0) continue
                let dx = nodes[oi].x - nodes[si].x
                let dy = nodes[oi].y - nodes[si].y
                let dist = Math.sqrt(dx * dx + dy * dy) || 1
                let force = (dist - 220) * 0.006 * cooling
                let fx = (dx / dist) * force
                let fy = (dy / dist) * force
                nodes[si].vx += fx
                nodes[si].vy += fy
                nodes[oi].vx -= fx
                nodes[oi].vy -= fy
            }

            // Apply velocity with damping
            const marginX = Math.min(80, width * 0.08)
            const marginY = Math.min(60, height * 0.08)
            for (const node of nodes) {
                node.vx *= 0.85
                node.vy *= 0.85
                node.x += node.vx
                node.y += node.vy
                node.x = Math.max(marginX, Math.min(width - marginX, node.x))
                node.y = Math.max(marginY, Math.min(height - marginY, node.y))
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
            const isMobile = width < 500
            const time = Date.now() * 0.001 // for subtle animations

            canvas.width = width * window.devicePixelRatio
            canvas.height = height * window.devicePixelRatio
            canvas.style.width = '100%'
            canvas.style.height = '100%'
            ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0)

            ctx.clearRect(0, 0, width, height)

            // Apply zoom & pan
            ctx.save()
            ctx.translate(offset.x, offset.y)
            ctx.scale(scale, scale)

            // Build a set of hovered-related edges + neighbor indices
            const hoveredEdges = new Set()
            const neighborIndices = new Set()
            if (hoveredNode !== null) {
                neighborIndices.add(hoveredNode)
                const hoveredName = graph.nodes[hoveredNode]
                graph.edges.forEach((edge, idx) => {
                    if (edge.subject === hoveredName || edge.object === hoveredName) {
                        hoveredEdges.add(idx)
                        neighborIndices.add(graph.nodes.indexOf(edge.subject))
                        neighborIndices.add(graph.nodes.indexOf(edge.object))
                    }
                })
            }

            // ──── Draw Edges ────
            graph.edges.forEach((edge, edgeIdx) => {
                const si = graph.nodes.indexOf(edge.subject)
                const oi = graph.nodes.indexOf(edge.object)
                if (si < 0 || oi < 0 || !nodes[si] || !nodes[oi]) return

                const isHighlighted = hoveredEdges.has(edgeIdx)
                const isDimmed = hoveredNode !== null && !isHighlighted

                // Curved edge using quadratic bezier
                const sx = nodes[si].x, sy = nodes[si].y
                const ex = nodes[oi].x, ey = nodes[oi].y
                const mx = (sx + ex) / 2
                const my = (sy + ey) / 2
                // offset control point perpendicular to the line
                const dx = ex - sx, dy = ey - sy
                const len = Math.sqrt(dx * dx + dy * dy) || 1
                const curvature = Math.min(30, len * 0.12)
                const cpx = mx + (-dy / len) * curvature
                const cpy = my + (dx / len) * curvature

                ctx.beginPath()
                ctx.moveTo(sx, sy)
                ctx.quadraticCurveTo(cpx, cpy, ex, ey)

                if (isHighlighted) {
                    ctx.strokeStyle = 'rgba(120, 180, 255, 0.7)'
                    ctx.lineWidth = 2.5
                    ctx.shadowColor = 'rgba(100, 160, 255, 0.4)'
                    ctx.shadowBlur = 8
                } else if (isDimmed) {
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)'
                    ctx.lineWidth = 0.8
                    ctx.shadowBlur = 0
                } else {
                    ctx.strokeStyle = 'rgba(100, 160, 255, 0.15)'
                    ctx.lineWidth = 1.2
                    ctx.shadowBlur = 0
                }
                ctx.stroke()
                ctx.shadowBlur = 0

                // Edge label (only on desktop, only for highlighted or default)
                if (!isMobile && !isDimmed) {
                    const labelX = cpx
                    const labelY = cpy
                    const label = edge.relation
                    ctx.font = '10px Inter, sans-serif'
                    const textWidth = ctx.measureText(label).width

                    // Dark pill background behind label
                    const px = 5, py = 3
                    ctx.fillStyle = isHighlighted ? 'rgba(30, 40, 65, 0.9)' : 'rgba(15, 20, 35, 0.75)'
                    ctx.beginPath()
                    const rx = labelX - textWidth / 2 - px
                    const ry = labelY - 6 - py
                    const rw = textWidth + px * 2
                    const rh = 12 + py * 2
                    const cornerR = 4
                    ctx.moveTo(rx + cornerR, ry)
                    ctx.lineTo(rx + rw - cornerR, ry)
                    ctx.quadraticCurveTo(rx + rw, ry, rx + rw, ry + cornerR)
                    ctx.lineTo(rx + rw, ry + rh - cornerR)
                    ctx.quadraticCurveTo(rx + rw, ry + rh, rx + rw - cornerR, ry + rh)
                    ctx.lineTo(rx + cornerR, ry + rh)
                    ctx.quadraticCurveTo(rx, ry + rh, rx, ry + rh - cornerR)
                    ctx.lineTo(rx, ry + cornerR)
                    ctx.quadraticCurveTo(rx, ry, rx + cornerR, ry)
                    ctx.fill()

                    ctx.fillStyle = isHighlighted ? 'rgba(180, 210, 255, 0.95)' : 'rgba(148, 163, 184, 0.65)'
                    ctx.textAlign = 'center'
                    ctx.textBaseline = 'middle'
                    ctx.fillText(label, labelX, labelY)
                }
            })

            // ──── Draw Nodes ────
            for (let i = 0; i < nodes.length; i++) {
                const node = nodes[i]
                const isHovered = hoveredNode === i
                const isNeighbor = neighborIndices.has(i)
                const isDimmed = hoveredNode !== null && !isNeighbor

                // Connection-based sizing: more connections = larger node
                const connections = connectionCounts[node.name] || 0
                const sizeMultiplier = 1 + Math.min(connections * 0.15, 0.8)
                const baseRadius = isMobile ? 4 : 6
                let nodeRadius = baseRadius * sizeMultiplier

                if (isHovered) nodeRadius = isMobile ? 9 : 14
                else if (isNeighbor && hoveredNode !== null) nodeRadius = isMobile ? 6 : 10

                // Outer glow ring (breathing animation)
                const breathe = 0.7 + 0.3 * Math.sin(time * 1.5 + i * 0.5)
                if (!isDimmed) {
                    const glowRadius = nodeRadius * (isHovered ? 3.2 : 2.2) * breathe
                    const gradient = ctx.createRadialGradient(node.x, node.y, nodeRadius * 0.5, node.x, node.y, glowRadius)
                    if (isHovered) {
                        gradient.addColorStop(0, 'rgba(100, 180, 255, 0.25)')
                        gradient.addColorStop(1, 'rgba(100, 180, 255, 0)')
                    } else {
                        gradient.addColorStop(0, 'rgba(80, 140, 255, 0.08)')
                        gradient.addColorStop(1, 'rgba(80, 140, 255, 0)')
                    }
                    ctx.beginPath()
                    ctx.arc(node.x, node.y, glowRadius, 0, Math.PI * 2)
                    ctx.fillStyle = gradient
                    ctx.fill()
                }

                // Node circle with gradient fill
                const nodeGrad = ctx.createRadialGradient(
                    node.x - nodeRadius * 0.3, node.y - nodeRadius * 0.3, nodeRadius * 0.1,
                    node.x, node.y, nodeRadius
                )
                if (isDimmed) {
                    nodeGrad.addColorStop(0, 'rgba(80, 110, 160, 0.4)')
                    nodeGrad.addColorStop(1, 'rgba(50, 70, 120, 0.3)')
                } else if (isHovered) {
                    nodeGrad.addColorStop(0, '#8cc8ff')
                    nodeGrad.addColorStop(1, '#4d94ff')
                } else if (isNeighbor && hoveredNode !== null) {
                    nodeGrad.addColorStop(0, '#7ab8ff')
                    nodeGrad.addColorStop(1, '#4080e0')
                } else {
                    nodeGrad.addColorStop(0, '#6aadff')
                    nodeGrad.addColorStop(1, '#3d7ee6')
                }

                ctx.beginPath()
                ctx.arc(node.x, node.y, nodeRadius, 0, Math.PI * 2)
                ctx.fillStyle = nodeGrad
                ctx.fill()

                // Bright ring on hovered/neighbor
                if (isHovered || (isNeighbor && hoveredNode !== null)) {
                    ctx.beginPath()
                    ctx.arc(node.x, node.y, nodeRadius + 1.5, 0, Math.PI * 2)
                    ctx.strokeStyle = isHovered ? 'rgba(140, 200, 255, 0.8)' : 'rgba(100, 160, 255, 0.4)'
                    ctx.lineWidth = isHovered ? 2 : 1
                    ctx.stroke()
                }

                // Label with dark background pill
                if (!isDimmed || isHovered) {
                    let label = node.name
                    if (isMobile && label.length > 16) label = label.substring(0, 14) + '…'
                    else if (!isMobile && label.length > 28) label = label.substring(0, 26) + '…'

                    const fontSize = isMobile
                        ? (isHovered ? 10 : 8)
                        : (isHovered ? 13 : 11)
                    ctx.font = `${isHovered ? '600' : '400'} ${fontSize}px Inter, sans-serif`
                    const textWidth = ctx.measureText(label).width
                    const labelY = node.y + nodeRadius + (isMobile ? 10 : 16)

                    // Dark background pill
                    const px = 6, py = 3
                    ctx.fillStyle = isHovered ? 'rgba(20, 30, 50, 0.9)' : 'rgba(10, 15, 30, 0.7)'
                    const pillX = node.x - textWidth / 2 - px
                    const pillY = labelY - fontSize / 2 - py
                    const pillW = textWidth + px * 2
                    const pillH = fontSize + py * 2
                    const cr = 4
                    ctx.beginPath()
                    ctx.moveTo(pillX + cr, pillY)
                    ctx.lineTo(pillX + pillW - cr, pillY)
                    ctx.quadraticCurveTo(pillX + pillW, pillY, pillX + pillW, pillY + cr)
                    ctx.lineTo(pillX + pillW, pillY + pillH - cr)
                    ctx.quadraticCurveTo(pillX + pillW, pillY + pillH, pillX + pillW - cr, pillY + pillH)
                    ctx.lineTo(pillX + cr, pillY + pillH)
                    ctx.quadraticCurveTo(pillX, pillY + pillH, pillX, pillY + pillH - cr)
                    ctx.lineTo(pillX, pillY + cr)
                    ctx.quadraticCurveTo(pillX, pillY, pillX + cr, pillY)
                    ctx.fill()

                    ctx.fillStyle = isHovered ? '#f0f4ff' : (isDimmed ? 'rgba(148, 163, 184, 0.4)' : 'rgba(190, 205, 225, 0.85)')
                    ctx.textAlign = 'center'
                    ctx.textBaseline = 'middle'
                    ctx.fillText(label, node.x, labelY)
                }
            }

            ctx.restore()

            animFrameRef.current = requestAnimationFrame(draw)
        }

        draw()
        return () => {
            if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current)
        }
    }, [graph, dimensions, hoveredNode, simulate, scale, offset, connectionCounts])

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
            if (Math.sqrt(dx * dx + dy * dy) < 28) {
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
                        style={{ display: 'block', cursor: dragRef.current.dragging ? 'grabbing' : hoveredNode !== null ? 'pointer' : 'grab', touchAction: 'none' }}
                        onMouseMove={handleMouseMove}
                        onMouseDown={handleMouseDown}
                        onMouseUp={handleMouseUp}
                        onMouseLeave={handleMouseUp}
                        onWheel={handleWheel}
                        onTouchStart={(e) => {
                            const t = e.touches[0]
                            dragRef.current = { dragging: true, startX: t.clientX, startY: t.clientY, startOffX: offset.x, startOffY: offset.y }
                        }}
                        onTouchMove={(e) => {
                            if (!dragRef.current.dragging) return
                            const t = e.touches[0]
                            setOffset({ x: dragRef.current.startOffX + (t.clientX - dragRef.current.startX), y: dragRef.current.startOffY + (t.clientY - dragRef.current.startY) })
                        }}
                        onTouchEnd={() => { dragRef.current.dragging = false }}
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

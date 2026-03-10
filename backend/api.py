"""
PaperMind — FastAPI Backend
SSE streaming with Firebase Auth + PostgreSQL multi-tenancy.
"""

import os
import sys
import json
import shutil
import asyncio
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

# ── Ensure backend/ is on the path ──────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from auth import get_current_user
from database import (
    init_db, close_db, upsert_user,
    get_history, add_history, get_memories, get_graph as db_get_graph,
    reset_user_data, check_daily_limit, get_all_users_admin_stats
)

from agents.planner_agent import plan_query
from agents.research_loop import collect_evidence
from agents.research_agent import generate_answer
from retrieval.retriever import build_context
from agents.memory_agent import store_research_memory
from memory.memory_retriever import retrieve_memory
from agents.evaluator_agent import evaluate_answer
from agents.coverage_agent import evaluate_coverage
from agents.complexity_agent import classify_complexity
from agents.depth_agent import evaluate_research_depth
from agents.graph_agent import extract_graph_knowledge
from memory.graph_retriever import find_related_concepts
from agents.hop_agent import determine_next_hop
from ingestion.pdf_loader import load_pdfs
from ingestion.chunker import chunk_documents
from ingestion.embedder import get_embedding_model

# ── Paths ────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _user_papers_dir(user_id):
    return DATA_DIR / user_id / "papers"


# ── App ──────────────────────────────────────────────────────
app = FastAPI(title="PaperMind")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


@app.on_event("shutdown")
def shutdown():
    close_db()


# ── Models ───────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str


# ── Helpers ──────────────────────────────────────────────────
def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ══════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════


# ── GET /api/me ──────────────────────────────────────────────
@app.get("/api/me")
async def get_me(user: dict = Depends(get_current_user)):
    upsert_user(user["uid"], user["email"], user["name"])
    return {"uid": user["uid"], "email": user["email"], "name": user["name"]}


# ── GET /api/admin/users ─────────────────────────────────────
@app.get("/api/admin/users")
async def get_admin_dashboard_data(user: dict = Depends(get_current_user)):
    # Hardcoded admin check as requested
    admin_email = os.getenv("ADMIN_EMAIL", "aryankale1410@gmail.com")
    if user.get("email") != admin_email:
        raise HTTPException(status_code=403, detail="Forbidden: You do not have admin access.")
    
    try:
        stats = get_all_users_admin_stats()
        return {"users": stats}
    except Exception as e:
        print(f"[Admin Error] {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail="Failed to load admin stats.")


# ── GET /api/papers ──────────────────────────────────────────
@app.get("/api/papers")
async def list_papers(user: dict = Depends(get_current_user)):
    from database import get_user_papers
    db_papers = get_user_papers(user["uid"])
    papers = []
    
    for p in db_papers:
        papers.append({
            "name": p["paper_title"].replace("_", " ").replace("-", " ").replace(".pdf", ""),
            "filename": p["paper_title"],
            "size_kb": "stored in cloud"
        })
        
    return {"papers": papers}


# ── POST /api/upload ─────────────────────────────────────────
@app.post("/api/upload")
async def upload_paper(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    uid = user["uid"]
    filename = file.filename
    content = await file.read()

    # 1. Upload to Firebase Storage
    try:
        from firebase_admin import storage
        bucket = storage.bucket()
        blob = bucket.blob(f"users/{uid}/papers/{filename}")
        blob.upload_from_string(content, content_type="application/pdf")
    except Exception as e:
        print(f"Firebase Storage upload failed: {e}")
        # non-fatal, we still want to embed it

    # 2. Extract Text & Embed into pgvector
    try:
        import tempfile
        import os
        from langchain_community.document_loaders import PyPDFLoader
        from database import add_document_chunks

        # write to temp file for PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            # inject metadata
            for doc in documents:
                doc.metadata["paper_title"] = filename
                doc.metadata["paper_id"] = filename

            chunks = chunk_documents(documents)
            embeddings = get_embedding_model()
            
            # insert into pgvector postgres DB instead of FAISS
            add_document_chunks(uid, filename, filename, chunks, embeddings)
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        return {"status": "uploaded", "index_error": str(e), "filename": filename}

    return {"status": "uploaded_and_indexed", "filename": filename}


# ── GET /api/memory ──────────────────────────────────────────
@app.get("/api/memory")
async def get_memory(user: dict = Depends(get_current_user)):
    memories = get_memories(user["uid"])
    return {"memories": memories}


# ── GET /api/graph ───────────────────────────────────────────
@app.get("/api/graph")
async def get_graph(user: dict = Depends(get_current_user)):
    graph = db_get_graph(user["uid"])
    return graph


# ── GET /api/history ─────────────────────────────────────────
@app.get("/api/history")
async def get_history_endpoint(user: dict = Depends(get_current_user)):
    return {"history": get_history(user["uid"])}


# ── POST /api/reset ──────────────────────────────────────────
@app.post("/api/reset")
async def reset_session(user: dict = Depends(get_current_user)):
    """Delete all user data: PDFs in Firebase, memory, graph, chunks and history."""
    errors = []
    uid = user["uid"]

    # Clear papers from Firebase Storage
    try:
        from firebase_admin import storage
        bucket = storage.bucket()
        blobs = bucket.list_blobs(prefix=f"users/{uid}/papers/")
        for blob in blobs:
            blob.delete()
    except Exception as e:
        errors.append(f"Failed to clear Firebase papers: {e}")

    # Clear DB data (this cascades to document_chunks, history, graph, memory)
    try:
        reset_user_data(uid)
    except Exception as e:
        errors.append(f"Failed to clear database: {e}")

    if errors:
        return {"status": "partial_reset", "errors": errors}

    return {"status": "reset_complete", "message": "All session data has been deleted."}


# ── POST /api/query (SSE) ────────────────────────────────────
@app.post("/api/query")
async def query_research(req: QueryRequest, user: dict = Depends(get_current_user)):
    question = req.question
    uid = user["uid"]

    async def event_stream():
        try:
            # ── 0. Enforce Rate Limit ─────────────────────────
            if not check_daily_limit(uid, limit=5):
                yield sse_event("error", {
                    "message": "You have reached your daily limit of 5 deep research queries. Please try again tomorrow! (Premium upgrades coming soon)"
                })
                return

            # ── 1. Classify complexity ────────────────────────
            yield sse_event("stage", {
                "stage": "classifying",
                "detail": "Determining research mode..."
            })
            await asyncio.sleep(0)

            mode_info = classify_complexity(question)
            mode = mode_info.get("mode", "deep")

            yield sse_event("stage", {
                "stage": "classified",
                "detail": f"Mode: {mode.upper()}",
                "mode": mode,
                "reason": mode_info.get("reason", "")
            })
            await asyncio.sleep(0)

            queries = [question]
            docs = []

            if mode == "quick":
                # ── Quick path ────────────────────────────────
                yield sse_event("stage", {
                    "stage": "retrieving",
                    "detail": "Quick retrieval..."
                })
                await asyncio.sleep(0)

                docs = collect_evidence([question], uid)
                context = build_context(docs)

            else:
                # ── 2. Plan sub-queries ───────────────────────
                yield sse_event("stage", {
                    "stage": "planning",
                    "detail": "Breaking query into sub-queries..."
                })
                await asyncio.sleep(0)

                queries = plan_query(question)

                yield sse_event("stage", {
                    "stage": "planned",
                    "detail": f"Generated {len(queries)} sub-queries",
                    "queries": queries
                })
                await asyncio.sleep(0)

                # ── 3. Collect evidence ───────────────────────
                yield sse_event("stage", {
                    "stage": "retrieving",
                    "detail": "Collecting evidence from papers..."
                })
                await asyncio.sleep(0)

                docs = collect_evidence(queries, uid)

                # ── 4. Graph expansion ────────────────────────
                related_concepts = find_related_concepts(uid, question)
                if related_concepts:
                    yield sse_event("stage", {
                        "stage": "graph_expansion",
                        "detail": f"Expanding via {len(related_concepts)} graph concepts",
                        "concepts": related_concepts[:5]
                    })
                    await asyncio.sleep(0)

                    graph_docs = collect_evidence(related_concepts, uid)
                    docs.extend(graph_docs)

                context = build_context(docs)

                # ── 5. Multi-hop loop ─────────────────────────
                max_hops = 2
                hop = 0
                while hop < max_hops:
                    hop_decision = determine_next_hop(question, context)
                    if not hop_decision.get("next_hop"):
                        break
                    new_query = hop_decision.get("new_query")
                    if not new_query:
                        break

                    yield sse_event("stage", {
                        "stage": "hop",
                        "detail": f"Research Hop {hop + 1}: {new_query}",
                        "hop": hop + 1,
                        "query": new_query
                    })
                    await asyncio.sleep(0)

                    new_docs = collect_evidence([new_query], uid)
                    docs.extend(new_docs)
                    context = build_context(docs)
                    hop += 1

                # ── 6. Depth check ────────────────────────────
                yield sse_event("stage", {
                    "stage": "depth_check",
                    "detail": "Evaluating research depth..."
                })
                await asyncio.sleep(0)

                depth_eval = evaluate_research_depth(question, context)
                if depth_eval.get("depth") == "expand":
                    extra_query = depth_eval.get("suggested_query", question)
                    extra_docs = collect_evidence([extra_query], uid)
                    docs.extend(extra_docs)
                    context = build_context(docs)

            # ── 7. Retrieve memory ────────────────────────────
            memory_items = retrieve_memory(uid, question)
            memory_context = "\n".join(
                [f"{m['topic']}: {m['key_finding']}" for m in memory_items]
            )

            # ── 8. Generate answer ────────────────────────────
            yield sse_event("stage", {
                "stage": "generating",
                "detail": "Generating research answer..."
            })
            await asyncio.sleep(0)

            answer = generate_answer(question, context, memory_context)

            # ── 9. Evaluate ───────────────────────────────────
            yield sse_event("stage", {
                "stage": "evaluating",
                "detail": "Evaluating answer quality..."
            })
            await asyncio.sleep(0)

            evaluation = evaluate_answer(question, answer, context)

            if not evaluation.get("grounded", True):
                yield sse_event("stage", {
                    "stage": "improving",
                    "detail": "Improving answer using evaluator feedback..."
                })
                await asyncio.sleep(0)

                improved_query = evaluation.get("improvement_query", question)
                docs = collect_evidence([improved_query], uid)
                context = build_context(docs)

                coverage = evaluate_coverage(question, context)
                if not coverage.get("enough_coverage", True):
                    extra_query = coverage.get("additional_query", question)
                    extra_docs = collect_evidence([extra_query], uid)
                    docs.extend(extra_docs)
                    context = build_context(docs)

                answer = generate_answer(question, context, memory_context)

            # ── 10. Build sources list ────────────────────────
            sources = []
            seen = set()
            for d in docs[:8]:
                if hasattr(d, "metadata"):
                    title = (
                        d.metadata.get("paper_title")
                        or d.metadata.get("source")
                        or d.metadata.get("paper_id")
                        or "Unknown Paper"
                    )
                    page = d.metadata.get("page", "?")
                    key = f"{title}-{page}"
                    if key not in seen:
                        sources.append({"title": title, "page": str(page)})
                        seen.add(key)

            # ── 11. Save memory + graph ───────────────────────
            yield sse_event("stage", {
                "stage": "saving",
                "detail": "Saving research memory & knowledge graph..."
            })
            await asyncio.sleep(0)

            store_research_memory(uid, question, answer)
            extract_graph_knowledge(uid, question, answer)

            # ── 12. Save to history ───────────────────────────
            add_history(uid, question, answer, mode, queries, sources)

            # ── 13. Final answer ──────────────────────────────
            yield sse_event("answer", {
                "answer": answer,
                "sources": sources,
                "mode": mode,
                "plan": queries,
                "grounded": evaluation.get("grounded", True),
            })

        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate" in error_msg or "quota" in error_msg:
                yield sse_event("error", {
                    "message": "Our AI models are currently experiencing extremely high demand from the launch. Please try your query again in a few minutes!"
                })
            else:
                yield sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

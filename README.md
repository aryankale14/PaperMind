# PaperMind - Deep Research Assistant: Detailed Project Documentation

## 1. Introduction and Overview
**PaperMind** is an advanced, AI-powered Deep Research Assistant designed to help users ingest, analyze, and query academic papers and large PDF documents. It goes beyond simple semantic search by implementing a robust **Multi-Agent RAG (Retrieval-Augmented Generation)** pipeline. 

The project is currently live at: **[https://paper-mind-nine.vercel.app/](https://paper-mind-nine.vercel.app/)**

PaperMind is built as a highly scalable multi-tenant application, allowing different users to maintain their isolated document stores, research memories, and knowledge graphs.

## 2. Technical Stack & Architecture

### Frontend
- **Framework:** React + Vite
- **Routing:** React Router v6
- **Styling:** Custom CSS (`index.css`) with responsive design for Desktop and Mobile.
- **Icons:** Lucide-React
- **State Management Context:** Firebase Authentication context (`AuthContext.jsx`)
- **Deployment:** Vercel

### Backend
- **Framework:** FastAPI (Python)
- **Database:** PostgreSQL (with `pgvector` extension for vector storage)
- **LLM Provider:** Google Gemini (`google-generativeai`) and Grok
- **Embedding Model:** Google Generative AI Embeddings (`GoogleGenerativeAIEmbeddings`)
- **Authentication:** Firebase Authentication (via JWT token verification)
- **Storage:** Firebase Storage (for raw PDF files)
- **Deployment:** Render (with a specific note on Free Tier constraints)

---

## 3. Core Features & UI Components

The application is protected by a unified Auth layer, routing unauthenticated users to `/login` or `/` (Landing Page) and authenticated users to `/app/*`.

### 3.1. Mobile-Responsive Layout & Sidebar Nav
- The platform features a responsive sidebar containing links to all core pages and a history visualizer.
- On mobile devices, the sidebar is hidden behind a hamburger menu overlay for better UX.

### 3.2. Authentication (Firebase)
- Users sign in/sign up using Firebase. 
- The `AuthContext.jsx` manages user state and provides a `getToken()` function, ensuring every API request to the backend includes a valid `Bearer` token in the `Authorization` header.

### 3.3. Research Page (`/`)
- The primary interface where users ask research questions.
- **Server-Sent Events (SSE):** Queries are sent to the `/api/query` backend endpoint via an open SSE stream. This allows the backend to send real-time chunks indicating the current "stage" (e.g., *Planning*, *Retrieving*, *Evaluating*, *Hop 1*) before sending the final answer.
- Features a markdown-rendered response space with inline citation/sources components.

### 3.4. Papers Page (`/papers`)
- Users upload `.pdf` documents here. 
- Display of previously uploaded papers.
- **Workflow:** UI sends the file via `FormData` to `/api/upload`.

### 3.5. Graph Page (`/graph`)
- Visualizes the user's specific Knowledge Graph extracted asynchronously during their research queries. Submits a `GET` request to `/api/graph`.

### 3.6. Memory Page (`/memory`)
- Displays the user's Long-Term Memory (LTM) entries. As the AI Agent discovers important facts or concepts, they are stored persistently to personalize and provide context to future queries.

### 3.7. Admin Dashboard (`/admin`)
- Accessible only if the user's email matches the hardcoded `ADMIN_EMAIL` env variable.
- Provides system-wide statistics: total users, all queries, and all uploaded papers.

---

## 4. Backend Workflow & The Multi-Agent System

The backend logic lies primarily in `api.py`. It integrates an intricate, 13-step research loop when users ask a question.

### 4.1. Step-by-Step Deep Research Pipeline (`api.py` -> `/api/query`)

When a user submits a query `POST /api/query`, the following workflow applies:

1. **Rate Limit Check:** The system verifies if the user has exceeded their daily limit of 5 deep research queries. If so, an error is yielded.
2. **Complexity Classification:** The `classify_complexity` agent determines if the query requires a `quick` or `deep` research mode.
3. **If `quick` mode:**
   - The system immediately triggers a "Quick Retrieval" using `collect_evidence` on the initial query.
   - Pushes straight to answer generation.
4. **If `deep` mode (The Core RAG loop):**
   - **Step A - Planning:** The `plan_query` agent breaks down the user's complex question into multiple sub-queries to ensure broad coverage.
   - **Step B - Evidence Collection:** `collect_evidence` retrieves chunks from the vector database for all planned sub-queries.
   - **Step C - Graph Expansion:** The `find_related_concepts` agent queries the user's Knowledge Graph (stored in PostgreSQL) for nodes connected to the topic. It searches for chunks related to these extended topics to improve context.
   - **Step D - Multi-Hop Reasoning Loop:** A loop (max 2 hops) executes. In each hop, the `determine_next_hop` agent assesses what information is still missing from the retrieved context. If more information is needed, a new query is generated, hitting the vector DB again.
   - **Step E - Depth Check:** The `evaluate_research_depth` agent considers if the current context is deep enough. If the verdict is "expand," it suggests one more query to grab additional, highly specific documents.
5. **Memory Retrieval:** The `retrieve_memory` agent checks the user's Long-Term Memory store to append personalized/historical context to the prompt.
6. **Answer Generation:** The LLM (Gemini) takes the massive, consolidated context payload, the user's memory, and the query, then synthesizes the final research answer.
7. **Self-Reflection & Evaluation:** 
   - The `evaluate_answer` agent reviews the generated answer against the retrieved context to verify **groundedness** (halting hallucinations).
   - If not grounded, the system issues an improved query, gets new docs, evaluates coverage via `evaluate_coverage`, and **regenerates the answer**.
8. **Sources Resolution:** The list of unique papers and page numbers used in the generation is compiled.
9. **Memory & Graph Saving:** The `store_research_memory` and `extract_graph_knowledge` agents analyze the final answer in the background and write new findings to the Postgres Memory and Graph tables.
10. **History Save:** The interaction is logged.
11. **Final OutputStream:** The JSON-formatted final answer is sent out via SSE.

### 4.2. Ingestion & Storage Workflow (The FAISS & pgvector Transition)

Let's clarify the exact evolution of vector storage in this project:
1. **The Origin (FAISS):** During early development, the project heavily relied on local `FAISS` indices via `langchain_community.vectorstores.FAISS`. 
2. **The Transition (PostgreSQL + pgvector):** FAISS ran into extreme limitations for a multi-tenant cloud application, as managing varying `.faiss` files per user via memory is highly inefficient and loses persistence on serverless/ephemeral hosts easily. 
   - **Current Live Architecture:** PaperMind now uses **`pgvector`** within a PostgreSQL database. 
   - When a user uploads a PDF (`/api/upload`), it is first backed up to Firebase Storage. 
   - Then, it is loaded using `PyPDFLoader`, chunked, embedded via Google Generative AI, and directly inserted into the PostgreSQL `document_embeddings` table which holds a `VECTOR(3072)` column.
   - **FAISS Remnants:** While FAISS dependencies and commented-out local scripts might still exist in the repository, **it is no longer used in the live deployment flow.**

### 4.3. Hybrid Retrieval & Reranking Constraints

**Cross-Encoder Reranking Status:**
The retriever (`retriever.py`) uses a Hybrid Search strategy:
1. **ANN Vector Search:** Utilizing pgvector's fast cosine distance queries (`<=>`).
2. **Lexical Search (BM25):** Standard keyword search logic via `BM25Retriever`.
3. **Paper Balancing:** The `balance_papers` logic ensures that chunks from a single paper don't overpower the context.
4. **Cross-Encoder Reranking:**
   - The project incorporates a `CrossEncoder` model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) in `reranker.py` designed to re-rank the hybrid results.
   - **Live Deployment Constraint:** Because the backend is hosted on Render's Free Tier (which severely limits RAM to ~512MB), **the Cross-Encoder model is explicitly disabled/commented out in production**. Loading the massive Cross-Encoder transformer causes Out-of-Memory (OOM) crashes on the server.
5. **Section Importance Scoring:** As a fallback to reranking, a lightweight, rule-based custom logic (`section_score`) elevates sections like "Abstract" and "Conclusion" dynamically.

## 5. PostgreSQL Schema Details

The multi-tenant nature of PaperMind is handled via the following schema (in `database.py`):
- `users`: Stores user metadata and `Firebase UID`.
- `document_embeddings`: Stores chunks, vector embeddings (`pgvector`), page numbers, and `paper_id`. Linked explicitly to `user_id` for absolute isolation.
- `research_history`: Raw logs of user chats and plans.
- `research_memory`: LTM key findings associated with user accounts.
- `graph_nodes` & `graph_edges`: Triplets (`subject`, `relation`, `object`) for the knowledge graph per user.

## 6. Deployment Notes
- **Frontend:** Deployed smoothly on Vercel. It successfully connects cross-origin to the backend.
- **Backend:** Deployed on Render. Due to hardware resource capping, memory-intensive operations are avoided in favor of remote managed services (PostgreSQL via Supabase) and lightweight keyword alternatives.

## 7. Summary
PaperMind represents an exceptionally complex orchestration of Agentic patterns, pushing RAG methodologies far beyond naive "embed-and-search" loops. By explicitly adopting Google Gemini, pgvector for reliable cloud vectors, Graph-based context expansion, and multi-hop reasoning algorithms, it guarantees deep, grounded research results.

---

## 8. Local Setup & Installation

To run PaperMind locally, you will need to set up both the FastAPI backend and the React frontend.

### Prerequisites
- Python 3.10+
- Node.js (v18+)
- PostgreSQL database with the `pgvector` extension enabled
- Google Gemini API Key
- Firebase Project (for Authentication and Storage)

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the `backend` directory and add your keys:
   ```env
   DATABASE_URL=postgresql://user:password@localhost:5432/papermind
   GEMINI_API_KEY=your_gemini_api_key
   GROK_API_KEY=your_grok_api_key
   # Add your Firebase Admin SDK credentials path or env vars here
   ```
5. Run the FastAPI server:
   ```bash
   uvicorn api:app --reload
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Create a `.env` file in the `frontend` directory for Firebase config:
   ```env
   VITE_FIREBASE_API_KEY=your_api_key
   VITE_FIREBASE_AUTH_DOMAIN=your_auth_domain
   VITE_FIREBASE_PROJECT_ID=your_project_id
   VITE_FIREBASE_STORAGE_BUCKET=your_storage_bucket
   VITE_FIREBASE_MESSAGING_SENDER_ID=your_messaging_sender_id
   VITE_FIREBASE_APP_ID=your_app_id
   VITE_API_BASE_URL=http://localhost:8000
   ```
4. Start the Vite development server:
   ```bash
   npm run dev
   ```
```

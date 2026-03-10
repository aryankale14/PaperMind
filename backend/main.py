'''from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_documents
from ingestion.embedder import get_embedding_model
from ingestion.vector_store import create_faiss_index


PDF_PATH = "data/papers/research_paper_1.pdf"


def run_ingestion():
    docs = load_pdf(PDF_PATH)
    chunks = chunk_documents(docs)
    embeddings = get_embedding_model()

    create_faiss_index(chunks, embeddings)


if __name__ == "__main__":
    run_ingestion()



from retrieval.retriever import retrieve_documents, build_context


def test_retrieval():
    query = input("Ask a research question: ")

    docs = retrieve_documents(query, k=5)

    print("\n--- Retrieved Documents ---\n")

    for d in docs:
        print(f"Score: {d['score']}")
        print(d["content"][:300])
        print("\n-------------------\n")

    context = build_context(docs)

    print("\n=== BUILT CONTEXT ===\n")
    print(context[:1500])


if __name__ == "__main__":
    test_retrieval()
    


from retrieval.retriever import retrieve_documents, build_context
from agents.research_agent import generate_answer


def ask_copilot():

    query = input("Ask a research question: ")

    docs = retrieve_documents(query, k=5)
    context = build_context(docs)

    print("\n🧠 Generating Answer...\n")

    answer = generate_answer(query, context)

    print(answer)


if __name__ == "__main__":
    ask_copilot()

'''

from agents.planner_agent import plan_query
from agents.research_loop import collect_evidence
from agents.research_agent import generate_answer
from retrieval.retriever import build_context
from agents.memory_agent import store_research_memory
from memory.memory_retriever import retrieve_memory
from agents.evaluator_agent import evaluate_answer
from agents.coverage_agent import evaluate_coverage
from utils.research_trace import ResearchTrace
from agents.complexity_agent import classify_complexity
from agents.depth_agent import evaluate_research_depth
from agents.graph_agent import extract_graph_knowledge
from memory.graph_retriever import find_related_concepts
from agents.hop_agent import determine_next_hop


def ask_copilot():
    trace = ResearchTrace()
    query = input("Ask a research question: ")

    print("\n🧠 Determining research mode...")
    mode_info = classify_complexity(query)
    mode = mode_info.get("mode", "deep")
    print(f"Mode selected: {mode.upper()}")

    if mode == "quick":

        print("\n⚡ Quick Mode Activated")

        docs = collect_evidence([query])
        context = build_context(docs)
        queries = [query]

    else:
        print("\n🧠 Planning research steps...")
        queries = plan_query(query)

        print("Sub-queries:", queries)

        print("\n🔎 Collecting evidence...")
        docs = collect_evidence(queries)

        # ===============================
        # 🧠 GRAPH EXPANSION
        # ===============================
        related_concepts = find_related_concepts(query)

        if related_concepts:

            print("🧠 Graph expansion concepts:", related_concepts)

            graph_docs = collect_evidence(related_concepts)
            docs.extend(graph_docs)
        context = build_context(docs)
        # ===============================
        # 🔁 MULTI-HOP RESEARCH LOOP
        # ===============================

        max_hops = 2
        hop = 0

        while hop < max_hops:

            hop_decision = determine_next_hop(query, context)

            if not hop_decision.get("next_hop"):
                break

            new_query = hop_decision.get("new_query")

            if not new_query:
                break

            print(f"\n🔁 Research Hop {hop+1}: {new_query}")

            new_docs = collect_evidence([new_query])
            docs.extend(new_docs)

            context = build_context(docs)

            hop += 1



# ===============================
# 🧠 RESEARCH DEPTH CHECK (ADD HERE)
# ===============================

    # ===============================
# 🧠 RESEARCH DEPTH CHECK
# ===============================

    if mode == "deep":

        print("\n🧠 Evaluating research depth...")

        depth_eval = evaluate_research_depth(query, context)

        if depth_eval.get("depth") == "expand":

            print("📚 Expanding research automatically...")

            extra_query = depth_eval.get("suggested_query", query)

            extra_docs = collect_evidence([extra_query])
            docs.extend(extra_docs)

            context = build_context(docs)

    trace.set_plan(queries)
    trace.add_sources(docs)
    trace.set_context_preview(context)
 
    memory_items = retrieve_memory(query)

    memory_context = "\n".join(
        [f"{m['topic']}: {m['key_finding']}" for m in memory_items]
    )

    print("\n🧠 Generating Answer...\n")
    answer = generate_answer(query, context, memory_context)

    # ---------- SELF CRITIC ----------
    print("\n🧠 Evaluating answer quality...")
    evaluation = evaluate_answer(query, answer, context)

    if not evaluation.get("grounded", True):

        print("🔁 Improving answer using evaluator feedback...")

        improved_query = evaluation.get("improvement_query", query)

        docs = collect_evidence([improved_query])
        context = build_context(docs)
        trace.set_context_preview(context)

        print("\n🧠 Evaluating research coverage...")
        coverage = evaluate_coverage(query, context)

        if not coverage.get("enough_coverage", True):

            print("📚 Expanding research depth...")

        extra_query = coverage.get("additional_query", query)

        extra_docs = collect_evidence([extra_query])
        docs.extend(extra_docs)

        context = build_context(docs)

        # regenerate ONLY if needed
        answer = generate_answer(query, context, memory_context)

    # ---------- FINAL OUTPUT ----------
    trace.display()
    print("\n✅ Final Answer:\n")
    print(answer)

    print("\n💾 Saving research memory...")
    store_research_memory(query, answer)
    extract_graph_knowledge(query, answer)

if __name__ == "__main__":
    ask_copilot()
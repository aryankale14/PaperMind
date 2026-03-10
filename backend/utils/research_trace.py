class ResearchTrace:

    def __init__(self):
        self.plan = []
        self.sources = []
        self.context_preview = ""
        self.reasoning_notes = ""

    def set_plan(self, queries):
        self.plan = queries

    def add_sources(self, docs):

        sources = []
        seen = set()

        for d in docs[:5]:

            title = "Unknown Paper"
            page = "?"

            if hasattr(d, "metadata"):

                meta = d.metadata

                title = (
                    meta.get("paper_title")
                    or meta.get("source")
                    or meta.get("paper_id")
                    or "Unknown Paper"
                )

                page = meta.get("page", "?")

            source = f"{title} — Page {page}"

            if source not in seen:
                sources.append(source)
                seen.add(source)

        self.sources = sources

    def set_context_preview(self, context):
        if context:
            self.context_preview = context[:500]
        else:
            self.context_preview = "[No context captured]"

    def set_reasoning(self, text):
        self.reasoning_notes = text[:400]

    def display(self):

        print("\n========== 🧠 RESEARCH TRACE ==========\n")

        print("📌 Research Plan:")
        for q in self.plan:
            print(f" - {q}")

        print("\n📚 Sources Used:")
        for s in self.sources:
            print(f" - {s}")

        print("\n🧩 Context Preview:")
        print(self.context_preview)

        print("\n=======================================\n")
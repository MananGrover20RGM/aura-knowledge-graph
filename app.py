import os
import sys
import gradio as gr
from neo4j import AsyncGraphDatabase
from google import genai

# Load environment secrets
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

async def query_point_in_time(as_of_year: int, target_topic: str = "Hotstar") -> str:
    as_of_iso = f"{as_of_year}-07-01T00:00:00Z"
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    cypher = """
    MATCH (src:Entity)-[r:TEMPORAL_EDGE]->(tgt:Entity)
    WHERE (src.name CONTAINS $topic OR tgt.name CONTAINS $topic)
      AND r.valid_from <= datetime($as_of)
      AND (r.valid_to IS NULL OR r.valid_to > datetime($as_of))
    RETURN src.name AS src, 
           r.predicate AS rel, 
           tgt.name AS tgt, 
           toString(r.valid_from) AS valid_from, 
           toString(r.valid_to) AS valid_to,
           r.evidence AS evidence
    """
    async with driver.session(database=NEO4J_DATABASE) as session:
        result = await session.run(cypher, topic=target_topic, as_of=as_of_iso)
        records = await result.data()
    await driver.close()
    
    if not records:
        return f"No active facts found for '{target_topic}' as of July {as_of_year}."

    facts = [f"--- ACTIVE KNOWLEDGE STATE AS OF JULY {as_of_year} ---"]
    for r in records:
        facts.append(
            f"• {r['src']} -[{r['rel']}]-> {r['tgt']} "
            f"(Since: {r['valid_from'][:10]}) | Evidence: {r['evidence']}"
        )
    return "\n".join(facts)

async def handle_ask(query: str):
    if not query.strip():
        return "Please enter a valid question."
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    cypher = """
    MATCH (src:Entity)-[r:TEMPORAL_EDGE]->(tgt:Entity)
    RETURN src.name AS source, 
           r.predicate AS relation, 
           tgt.name AS target, 
           toString(r.valid_from) AS valid_from, 
           toString(r.valid_to) AS valid_to,
           r.evidence AS evidence
    LIMIT 35
    """
    async with driver.session(database=NEO4J_DATABASE) as session:
        result = await session.run(cypher)
        records = await result.data()
    await driver.close()
    
    lines = [
        f"Fact: {r['source']} --[{r['relation']}]--> {r['target']} "
        f"(Valid: {r['valid_from']} to {r['valid_to'] or 'Present'}) | Evidence: {r['evidence']}" 
        for r in records
    ]
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"Answer using strictly this knowledge graph context:\n{chr(10).join(lines)}\n\nQuestion: {query}"
    res = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return res.text.strip()

async def run_ui_timetravel(question: str, year: int, topic: str):
    facts = await query_point_in_time(int(year), topic)
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""
    You are the AURA Temporal Graph Reasoner.
    Answer strictly using the provided point-in-time facts as of year {int(year)}.
    Do NOT include future events that had not occurred yet relative to {int(year)}.
    
    POINT-IN-TIME EVIDENCE:
    {facts}
    
    QUESTION:
    {question}
    """
    res = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return res.text.strip()

with gr.Blocks(title="AURA Bi-Temporal Platform", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AURA: Bi-Temporal Knowledge Graph Platform")
    gr.Markdown("Autonomous knowledge extraction, historical point-in-time reasoning, and Graph RAG.")
    
    with gr.Tab("Time-Travel Copilot"):
        with gr.Row():
            topic_input = gr.Textbox(label="Topic Focus", value="Hotstar")
            year_slider = gr.Slider(minimum=2015, maximum=2026, step=1, value=2021, label="Historical Snapshot Year")
        historical_question = gr.Textbox(
            label="Historical Question", 
            value="Who owns or operates the service, and what is its corporate structure?"
        )
        time_travel_btn = gr.Button("Query Historical Snapshot", variant="primary")
        temporal_output = gr.Markdown(label="Time-Travel Analysis")
        time_travel_btn.click(
            fn=run_ui_timetravel, 
            inputs=[historical_question, year_slider, topic_input], 
            outputs=[temporal_output]
        )
        
    with gr.Tab("Ask Graph Copilot"):
        with gr.Row():
            q_input = gr.Textbox(label="General Question", placeholder="e.g., Explain the merger between Viacom18 and Hotstar.")
        ask_btn = gr.Button("Query Full Knowledge Graph", variant="primary")
        answer_out = gr.Markdown(label="Answer")
        ask_btn.click(fn=handle_ask, inputs=[q_input], outputs=[answer_out])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    # block_thread ensures Render keeps the web server process alive
    demo.launch(server_name="0.0.0.0", server_port=port, prevent_thread_lock=False)

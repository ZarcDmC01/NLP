import os
from dotenv import load_dotenv
load_dotenv()
import gradio as gr
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "nightreign_wiki"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 4

# Chargement au démarrage
print("Chargement du modèle d'embedding...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Connexion à ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)

print("Connexion à Groq...")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

print(f"Prêt ! {collection.count()} chunks disponibles.")


def retrieve(query, top_k=TOP_K):
    embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return docs, metas


def build_context(docs, metas):
    parts = []
    for doc, meta in zip(docs, metas):
        source = f"[{meta['title']} > {meta['section']}]"
        parts.append(f"{source}\n{doc}")
    return "\n\n---\n\n".join(parts)


def answer(question, history):
    if not question.strip():
        return "", history

    docs, metas = retrieve(question)
    context = build_context(docs, metas)

    system_prompt = (
        "You are an expert assistant for Elden Ring Nightreign. "
        "Answer the user's question using ONLY the context provided below from the official wiki. "
        "Be precise, complete, and cite the source sections when relevant. "
        "If the context doesn't contain enough information, say so clearly. "
        "Answer in the same language as the user's question.\n\n"
        f"CONTEXT:\n{context}"
    )

    # Construire l'historique pour Groq à partir du format messages Gradio
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )

    reply = response.choices[0].message.content

    # Ajouter les sources
    seen = set()
    sources = []
    for meta in metas:
        key = meta["url"]
        if key not in seen:
            seen.add(key)
            sources.append(f"- [{meta['title']}]({meta['url']})")

    sources_text = "\n\n**Sources :**\n" + "\n".join(sources)
    full_reply = reply + sources_text

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": full_reply})
    return "", history


with gr.Blocks(title="Nightreign RAG") as demo:
    gr.Markdown("# Elden Ring Nightreign — Wiki Assistant")
    gr.Markdown("Pose tes questions sur Elden Ring Nightreign. Les réponses viennent du wiki officiel.")

    chatbot = gr.Chatbot(height=500, show_label=False)
    with gr.Row():
        txt = gr.Textbox(
            placeholder="Ex: What are the best weapons for the Wylder class?",
            show_label=False,
            scale=5
        )
        btn = gr.Button("Envoyer", scale=1, variant="primary")

    state = gr.State([])

    btn.click(answer, inputs=[txt, state], outputs=[txt, chatbot])
    txt.submit(answer, inputs=[txt, state], outputs=[txt, chatbot])

    gr.Examples(
        examples=[
            "What are the best relics for a Wylder build?",
            "How do I unlock the Forsaken Hollows DLC?",
            "What are the weaknesses of Heolstor the Nightlord?",
            "Comment fonctionne le mode coopératif ?",
            "Quels sont les effets de status dans le jeu ?",
        ],
        inputs=txt
    )

if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Soft())

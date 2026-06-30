import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

DATA_FILE = "wiki_data_v2.json"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "nightreign_wiki"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHUNK_CHARS = 1000   # ~250 tokens, bon équilibre précision/contexte
BATCH_SIZE = 64


def split_text(text, max_chars=MAX_CHUNK_CHARS):
    """Découpe un texte long en sous-chunks par phrase."""
    if len(text) <= max_chars:
        return [text]
    sentences = text.replace(". ", ".|").replace("? ", "?|").replace("! ", "!|").split("|")
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) > max_chars and current:
            chunks.append(current.strip())
            current = sent
        else:
            current += " " + sent
    if current.strip():
        chunks.append(current.strip())
    return chunks


def build_documents(data):
    docs, metas, ids = [], [], []
    doc_id = 0

    for page in data:
        url = page["url"]
        title = page["title"]

        for chunk in page["chunks"]:
            section = chunk["section"]
            text = chunk["text"].strip()
            chunk_type = chunk.get("type", "text")

            if not text or len(text) < 30:
                continue

            # Préfixe contextuel pour améliorer la recherche
            prefix = f"[{title}] {section}: " if section != title else f"[{title}]: "
            full_text = prefix + text

            sub_chunks = split_text(full_text)
            for sub in sub_chunks:
                if len(sub) < 30:
                    continue
                docs.append(sub)
                metas.append({
                    "url": url,
                    "title": title,
                    "section": section,
                    "type": chunk_type
                })
                ids.append(f"doc_{doc_id}")
                doc_id += 1

    return docs, metas, ids


def index():
    print(f"Chargement de {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {len(data)} pages chargées")

    print("Construction des documents...")
    docs, metas, ids = build_documents(data)
    print(f"  {len(docs)} chunks générés")

    print(f"Chargement du modèle d'embedding : {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print("Connexion à ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Supprimer l'ancienne collection si elle existe
    try:
        client.delete_collection(COLLECTION_NAME)
        print("  Ancienne collection supprimée")
    except:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"Indexation par batchs de {BATCH_SIZE}...")
    total = len(docs)
    for i in range(0, total, BATCH_SIZE):
        batch_docs = docs[i:i + BATCH_SIZE]
        batch_metas = metas[i:i + BATCH_SIZE]
        batch_ids = ids[i:i + BATCH_SIZE]

        embeddings = model.encode(batch_docs, show_progress_bar=False).tolist()

        collection.add(
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_metas,
            ids=batch_ids
        )

        done = min(i + BATCH_SIZE, total)
        pct = done / total * 100
        print(f"  [{done}/{total}] {pct:.1f}%", end="\r")

    print(f"\nIndexation terminée ! {total} chunks dans ChromaDB → {CHROMA_DIR}/")


if __name__ == "__main__":
    index()

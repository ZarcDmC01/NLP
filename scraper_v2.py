import json
import time
import os
from urllib.parse import urljoin, urlparse
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

BASE_URL = "https://eldenringnightreign.wiki.fextralife.com"
OUTPUT_FILE = "wiki_data_v2.json"
VISITED_FILE = "visited_urls_v2.json"
DELAY = 1.2

# Points d'entrée pour découvrir les sous-pages individuelles
SEED_URLS = [
    "https://eldenringnightreign.wiki.fextralife.com/Elden+Ring+Nightreign+Wiki",
    "https://eldenringnightreign.wiki.fextralife.com/Weapons",
    "https://eldenringnightreign.wiki.fextralife.com/Armor",
    "https://eldenringnightreign.wiki.fextralife.com/Bosses",
    "https://eldenringnightreign.wiki.fextralife.com/Creatures+and+Enemies",
    "https://eldenringnightreign.wiki.fextralife.com/Relics",
    "https://eldenringnightreign.wiki.fextralife.com/Skills",
    "https://eldenringnightreign.wiki.fextralife.com/NPCs",
    "https://eldenringnightreign.wiki.fextralife.com/Items",
    "https://eldenringnightreign.wiki.fextralife.com/Locations",
    "https://eldenringnightreign.wiki.fextralife.com/Lore",
    "https://eldenringnightreign.wiki.fextralife.com/Nightfarers",
]

SKIP_PREFIXES = [
    "/Special:", "/User:", "/Talk:", "/File:", "/Template:",
    "/Category:", "/Help:", "/?", "/index.php", "/Forum:"
]

SKIP_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".svg", ".pdf", ".zip"]


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_valid_url(url):
    parsed = urlparse(url)
    if parsed.netloc and parsed.netloc != urlparse(BASE_URL).netloc:
        return False
    path = parsed.path
    for prefix in SKIP_PREFIXES:
        if path.startswith(prefix):
            return False
    for ext in SKIP_EXTENSIONS:
        if path.lower().endswith(ext):
            return False
    return True


def table_to_text(table):
    """Convertit une table HTML en texte structuré lisible."""
    rows = []
    headers = []

    # Récupérer les headers
    for th in table.find_all("th"):
        headers.append(th.get_text(" ", strip=True))

    # Récupérer les lignes
    for tr in table.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if cells and any(c for c in cells):
            if headers and len(cells) == len(headers):
                row = " | ".join(f"{h}: {v}" for h, v in zip(headers, cells))
            else:
                row = " | ".join(cells)
            rows.append(row)

    return "\n".join(rows)


def extract_content(html, url):
    soup = BeautifulSoup(html, "lxml")

    # Titre
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)

    # Contenu principal
    content_div = (
        soup.find(id="wiki-content-block")
        or soup.find(class_="wiki-content-block")
        or soup.find(id="content")
        or soup.find("article")
    )

    if not content_div:
        return None

    # Supprimer parasites
    for tag in content_div.find_all(["script", "style", "nav", "footer",
                                      "aside", "div.comments-area",
                                      "div.social-buttons", "div.alertify"]):
        tag.decompose()

    chunks = []
    current_section = title
    buffer = []

    for tag in content_div.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th", "table"]):
        if tag.name == "table":
            # Éviter les tables déjà traitées via td/th
            if tag.find_parent("table"):
                continue
            table_text = table_to_text(tag)
            if table_text:
                chunks.append({
                    "section": current_section,
                    "type": "table",
                    "text": table_text
                })
            continue

        # Éviter le texte dans les tables (déjà géré)
        if tag.find_parent("table"):
            continue

        text = tag.get_text(" ", strip=True)
        if not text:
            continue

        if tag.name in ["h1", "h2", "h3", "h4"]:
            if buffer:
                chunks.append({
                    "section": current_section,
                    "type": "text",
                    "text": " ".join(buffer)
                })
                buffer = []
            current_section = text
        else:
            buffer.append(text)

    if buffer:
        chunks.append({
            "section": current_section,
            "type": "text",
            "text": " ".join(buffer)
        })

    return {"url": url, "title": title, "chunks": chunks}


def extract_links(html, current_url):
    soup = BeautifulSoup(html, "lxml")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(current_url, href)
        parsed = urlparse(full_url)
        clean = parsed._replace(fragment="", query="").geturl()
        if is_valid_url(clean):
            links.add(clean)
    return links


def scrape():
    visited = set(load_json(VISITED_FILE, []))
    all_data = load_json(OUTPUT_FILE, [])

    # Construire la queue avec les seeds non visités
    queue = [u for u in SEED_URLS if u not in visited]
    print(f"Démarrage — {len(visited)} déjà visitées, {len(all_data)} sauvegardées, {len(queue)} en queue")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        # Bloquer images/fonts pour aller plus vite
        page.route("**/*.{png,jpg,jpeg,gif,svg,woff,woff2,ttf}", lambda r: r.abort())

        while queue:
            url = queue.pop(0)
            if url in visited:
                continue

            print(f"  [{len(visited)+1}] {url}")
            visited.add(url)

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=20000)
                # Attendre que le contenu principal soit chargé
                try:
                    page.wait_for_selector("#wiki-content-block", timeout=5000)
                except:
                    pass
                html = page.content()
            except Exception as e:
                print(f"    -> Erreur : {e}")
                continue

            page_data = extract_content(html, url)
            if page_data and page_data["chunks"]:
                all_data.append(page_data)
                tables = sum(1 for c in page_data["chunks"] if c["type"] == "table")
                print(f"    -> '{page_data['title']}' — {len(page_data['chunks'])} sections ({tables} tables)")
            else:
                print(f"    -> Pas de contenu utile")

            new_links = extract_links(html, url)
            added = 0
            for link in new_links:
                if link not in visited and link not in queue:
                    queue.append(link)
                    added += 1

            if len(visited) % 10 == 0:
                save_json(OUTPUT_FILE, all_data)
                save_json(VISITED_FILE, list(visited))
                print(f"  [Checkpoint] {len(all_data)} pages | {len(queue)} en queue")

            time.sleep(DELAY)

        browser.close()

    save_json(OUTPUT_FILE, all_data)
    save_json(VISITED_FILE, list(visited))
    print(f"\nTerminé ! {len(all_data)} pages → {OUTPUT_FILE}")


if __name__ == "__main__":
    scrape()

import requests
from bs4 import BeautifulSoup
import json
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

# 🔹 lista de categorías a recorrer
CATEGORIES = [
    "https://ecolite.com.co/precio/luminarias-interiores/",
    "https://ecolite.com.co/precio/lamparas-inteligentes/",
    "https://ecolite.com.co/precio/iluminacion-exterior/",
    "https://ecolite.com.co/precio/iluminacion-sumergible/",
    "https://ecolite.com.co/precio/iluminacion-decorativa/",
    "https://ecolite.com.co/precio/lamparas-negocios/",
    "https://ecolite.com.co/precio/iluminacion-emergencia/",
    "https://ecolite.com.co/precio/iluminacion-industrial/",
    "https://ecolite.com.co/precio/perfiles-luces-led/",
    "https://ecolite.com.co/precio/lamparas-solares-led/",
    "https://ecolite.com.co/precio/sensores/",
    "https://ecolite.com.co/precio/productos-electricos/",
    "https://ecolite.com.co/precio/energia-solar/",
    "https://ecolite.com.co/precio/camaras-seguridad-solares-hikvision/"
]

OUTPUT = "productos.json"
CHECKPT = "productos_checkpoint.json"
productos = {}

# 🛡 sesión con reintentos automáticos
session = requests.Session()
retries = Retry(total=5, backoff_factor=2,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"])
session.mount("https://", HTTPAdapter(max_retries=retries))
session.headers.update({"User-Agent": "Mozilla/5.0"})

# ========== EXTRACCIÓN DE TAGS / CATEGORÍAS ==========
def scrape_product_meta(product_url: str):
    """Extrae categorías y etiquetas desde la ficha del producto"""
    try:
        res = session.get(product_url, timeout=60)
        if res.status_code != 200:
            return [], []
        soup = BeautifulSoup(res.text, "html.parser")
        
        # categorías
        cat_links = soup.select("span.posted_in a")
        categories = [c.get_text(strip=True) for c in cat_links]

        # etiquetas
        tag_links = soup.select("span.tagged_as a")
        tags = [t.get_text(strip=True) for t in tag_links]

        return categories, tags
    except Exception as e:
        print(f"⚠️ Error en {product_url}: {e}")
        return [], []

# ========== SCRAPER DE CATEGORÍAS ==========
def scrape_category(base_url: str, productos: dict):
    page = 1
    while True:
        url = base_url if page == 1 else f"{base_url}page/{page}/"
        try:
            res = session.get(url, timeout=60)
        except Exception as e:
            print(f"⚠️ Error en {url}: {e}")
            break

        if res.status_code != 200:
            break

        soup = BeautifulSoup(res.text, "html.parser")
        items = soup.select("div.wd-product")
        if not items:
            break

        print(f"🔎 {base_url} - Página {page}: {len(items)} productos")

        # recolectar todos los productos primero
        batch = []
        for prod in items:
            title_el = prod.select_one("h3.wd-entities-title a")
            price_el = prod.select_one("span.price")
            img_el   = prod.select_one("img")
            add_btn  = prod.select_one("a.add_to_cart_button")

            if not (title_el and img_el and add_btn):
                continue

            name = title_el.get_text(strip=True)
            product_url = title_el.get("href")
            price_text = price_el.get_text(strip=True) if price_el else "N/A"

            if "srcset" in img_el.attrs and img_el["srcset"]:
                img_url = img_el["srcset"].split(",")[0].split()[0]
            elif "data-src" in img_el.attrs:
                img_url = img_el["data-src"]
            else:
                img_url = img_el.get("src") or ""

            code = (add_btn.get("data-product_sku") or "").strip().upper()
            if not code:
                continue

            batch.append((code, name, price_text, img_url, product_url, base_url))

        # pedir meta (categorías + tags) en paralelo
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_map = {executor.submit(scrape_product_meta, b[4]): b for b in batch}
            for fut in as_completed(future_map):
                code, name, price_text, img_url, product_url, base_url = future_map[fut]
                try:
                    categories, tags = fut.result()
                except Exception:
                    categories, tags = [], []

                productos[code] = {
                    "code": code,
                    "name": name,
                    "price": price_text,
                    "img_url": img_url,
                    "url": product_url,
                    "category": base_url,
                    "categories": categories,
                    "tags": tags
                }
                print(f"✔️ {code} - {name} ({len(tags)} tags)")

        page += 1
        time.sleep(1.0)

        # 📝 checkpoint cada 5 páginas
        if page % 5 == 0:
            with open(CHECKPT, "w", encoding="utf-8") as f:
                json.dump(productos, f, ensure_ascii=False, indent=2)
            print(f"💾 Checkpoint guardado con {len(productos)} productos")

# 🔄 ejecutar scraping para todas las categorías
for cat in CATEGORIES:
    scrape_category(cat, productos)

# guardar en JSON final
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(productos, f, ensure_ascii=False, indent=2)

print(f"✅ Guardado en {OUTPUT} con {len(productos)} productos únicos de {len(CATEGORIES)} categorías")

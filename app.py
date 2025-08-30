import os
from typing import Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import re, json

# llama-index
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from docx import Document as DocxDocument


from collections import defaultdict
LAST_CODES = defaultdict(list)   # session_id -> [codes]

PHOTO_INTENT = re.compile(r"\b(foto|imagen|muéstrame|muestrame|mostrar|enséñame|ensename|ver producto|ver foto)\b", re.I)




with open("productos.json", "r", encoding="utf-8") as f:
    PRODUCTOS = json.load(f)

# regex para detectar códigos (ej: TLMCR-B, ARQ40, etc.)
code_regex = re.compile(r"\b[A-Z0-9\-]{3,15}\b")



# ================== CONFIG ==================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ================== STATIC FILES ==================
app.mount("/static", StaticFiles(directory="public"), name="static")

@app.get("/")
def home():
    return FileResponse("public/chatbox.html")

# memoria simple
conversations: Dict[str, list] = {}

# ================== PROMPT ==================
SYSTEM_PROMPT = """Eres un asesor de iluminación de Ecolite (Empresa de Cali, Colombia), eres el mejor en lo que haces y eres directo.
Tu objetivo es ayudar al cliente a encontrar la mejor opción de iluminación.
Sugiere luminarias según el ambiente a iluminar (ejemplo: tienda de ropa → TrackLights, oficina → lineales, casa → decorativas/apliques, exterior → postes/balas/pisos).
Usa esto solo como referencia, no como respuesta fija.
Responde CORTO y DIRECTO.

Solo puedes recomendar productos que existan en el catálogo (productos.json).
Siempre incluye el SKU real y exacto de ese catálogo.
Nunca inventes SKUs.
Si no hay un producto adecuado, responde que no lo encuentras en el catálogo.


Reglas:
- Conversa de manera natural y fluida, como una persona real.
- Haz preguntas paso a paso según lo que el cliente diga.
- Usa la información del catálogo como referencia principal para recomendar productos reales.
- No inventes productos que no estén en el catálogo.
- Responde en mensajes cortos y fáciles de leer, como un chat.
"""

# ================== RAG CONFIG ==================
INDEX_PATH = "storage"
DATA_PATH = "data"

def find_products_by_text(text: str, k: int = 1):
    text = text.lower()
    cand = []
    for code, p in PRODUCTOS.items():
        name = p["name"].lower()
        # métrica simple: número de palabras en común
        overlap = sum(1 for w in text.split() if w in name and len(w) > 2)
        if overlap:
            cand.append((overlap, code))
    cand.sort(reverse=True)
    top = [code for _, code in cand[:k]]
    res = []
    for code in top:
        p = PRODUCTOS[code]
        res.append({
            "code": code, "name": p["name"], "price": p["price"],
            "img_url": p["img_url"], "url": p["url"]
        })
    return res


def build_index_from_docx(docx_path: str):
    doc = DocxDocument(docx_path)
    docs = []

    # Extraer texto completo del DOCX
    full_text = "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
    if full_text:
        docs.append(Document(text=full_text))

    # Parsear y crear índice
    parser = SentenceSplitter(chunk_size=512)
    nodes = parser.get_nodes_from_documents(docs)
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=INDEX_PATH)
    return index

index = None

@app.on_event("startup")
def load_index():
    global index
    try:
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_PATH)
        index = load_index_from_storage(storage_context)
        print("✅ Índice cargado desde storage")
    except Exception as e:
        print("⚠️ No se pudo cargar índice, se reconstruirá:", e)
        index = build_index_from_docx("data/catalogo.docx")
        print("📥 Índice creado y guardado en", INDEX_PATH)


def find_by_text(q: str, k: int = 3):
    if not q: return []
    q = q.lower()  
    scored = [] 
    for code, p in PRODUCTOS.items():
        name = p["name"].lower()
        # puntuación sencilla por palabras compartidas (>2 letras)
        score = sum(1 for w in re.findall(r"\w{3,}", q) if w in name)
        if score:
            scored.append((score, code))
    scored.sort(reverse=True)
    out = []
    for _, code in scored[:k]:
        p = PRODUCTOS[code]
        out.append({"code": code, "name": p["name"], "price": p["price"],
                    "img_url": p["img_url"], "url": p["url"]})
    return out



# ================== CHAT ==================
class ChatIn(BaseModel):
    message: str
    lead: dict | None = None

    

@app.post("/chat")
def chat(in_: ChatIn):
    session_id = "default"

    if session_id not in conversations:
        conversations[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    query_engine = index.as_query_engine(similarity_top_k=3)
    results = query_engine.query(in_.message)
    context = str(results)

    user_message = f"Contexto:\n{context}\n\nCliente: {in_.message}"
    conversations[session_id].append({"role": "user", "content": user_message})

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversations[session_id]
    )
    reply = resp.choices[0].message.content

    # === SIEMPRE intentar armar cards ===
    # 1) busca SKUs en mensaje + reply
    texto_total = f"{in_.message} {reply}"
    codes = []
    seen = set()
    for m in code_regex.findall(texto_total):
        c = m.upper()
        if c in PRODUCTOS and c not in seen:
            seen.add(c); codes.append(c)

    productos_detectados = []
    # 2) si hay códigos válidos → usa esos
    for code in codes:
        p = PRODUCTOS[code]
        productos_detectados.append({
            "code": code, "name": p["name"], "price": p["price"],
            "img_url": p["img_url"], "url": p["url"]
        })

    # 3) si no hubo códigos válidos → intenta por texto (del usuario y del reply)
    if not productos_detectados:
        productos_detectados = find_by_text(in_.message, k=3) or find_by_text(reply, k=3)

    # 4) limita y devuelve
    productos_detectados = productos_detectados[:3]  # evita saturar
    conversations[session_id].append({"role":"assistant","content":reply})
    return {"reply": reply, "productos": productos_detectados}
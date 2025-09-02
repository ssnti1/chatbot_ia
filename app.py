import os, re, json, unicodedata, uuid, random
from typing import Dict, Any, List, Set, Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# ===== OpenAI opcional (tool-calling); hay fallback si no hay API =====
OPENAI_OK = True
try:
    from openai import OpenAI
except Exception:
    OPENAI_OK = False

load_dotenv()
client = None
if OPENAI_OK and os.getenv("OPENAI_API_KEY"):
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        OPENAI_OK = False

app = FastAPI(title="Ecolite Chatbot")
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

@app.get("/health")
def health():
    return {"ok": True, "openai": OPENAI_OK}

# ================== DATA ==================
with open("productos.json", "r", encoding="utf-8") as f:
    PRODUCTOS: Dict[str, dict] = json.load(f)

code_regex = re.compile(r"\b[A-Z0-9\-]{3,15}\b", re.I)

# ================== NORMALIZACIÓN/TAGS ==================
def _norm(s: str) -> str:
    if not s: return ""
    s = s.lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s

def parse_presupuesto(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    s = _norm(str(v).strip())
    s = s.replace("cop", "").replace("$", "").strip()
    has_m = ("m" in s) or ("millon" in s) or ("millones" in s)
    has_k = ("k" in s) or ("mil" in s)
    m = re.search(r"(\d+(?:[.,]\d+)*)", s)
    if not m:
        return None
    num_txt = m.group(1).replace(",", ".")
    try:
        val = float(num_txt)
    except ValueError:
        val = float(re.sub(r"[^\d.]", "", num_txt) or 0)
    if has_m and not has_k:
        return int(round(val * 1_000_000))
    if has_k and not has_m:
        return int(round(val * 1_000))
    return int(round(val))

SINONIMOS_BASE = {
    "sumergible": {"sumergible", "subacuatica", "ip68", "piscina", "acuatico"},
    "superficie": {"superficie", "sobreponer", "adherir"},
    "empotrado":  {"empotrado", "encastrar", "embutido", "incrustado"},
    "escritorio": {"escritorio", "desk"},
}

def must_have_from_state(state: dict, last_msg: str) -> set[str]:
    texto = _norm(" ".join([
        str(state.get("espacio") or ""),
        str(state.get("tipo") or ""),
        str(state.get("instalacion") or ""),
        last_msg or ""
    ]))
    must = set()
    if any(x in texto for x in ["piscina", "sumergible", "ip68", "subacuatica"]):
        must.add("sumergible")
    if any(x in texto for x in ["superficie", "sobreponer"]):
        must.add("superficie")
    if any(x in texto for x in ["empotrado", "embutido", "encastrar"]):
        must.add("empotrado")
    if any(x in texto for x in ["escritorio", "desk"]):
        must.add("escritorio")
    return must

def buscar_productos(
    query_text: str,
    k: int = 5,
    must_have: set[str] | None = None,
    presupuesto: int | None = None,
    shuffle: bool = False,
    exclude_codes: Set[str] | None = None,
):
    qn = _norm(query_text)
    must_have = { _norm(x) for x in (must_have or set()) }
    exclude_codes = exclude_codes or set()

    def contiene_must_have(full_text: str) -> bool:
        if not must_have:
            return True
        hay = set()
        for m in must_have:
            grupo = SINONIMOS_BASE.get(m, {m})
            if any(g in full_text for g in grupo):
                hay.add(m)
        return len(hay) == len(must_have)

    candidatos: List[tuple[int, str]] = []
    for code, p in PRODUCTOS.items():
        if code in exclude_codes:
            continue
        name = _norm(str(p.get("name", "")))
        tags = _norm(" ".join([str(t) for t in p.get("tags", [])]))
        cats = _norm(" ".join([str(c) for c in p.get("categories", [])]))
        price = p.get("price")
        full_text = f"{name} {tags} {cats}"

        if not contiene_must_have(full_text):
            continue

        score = 0
        for w in [w for w in qn.split() if len(w) > 2]:
            if w in name: score += 6
            if w in tags: score += 4
            if w in cats: score += 2

        if any(x in qn for x in ["piscina", "sumergible", "ip68", "subacuatica"]):
            if any(x in full_text for x in ["piscina", "sumergible", "ip68", "subacuatica"]):
                score += 5

        if isinstance(price, (int, float)) and isinstance(presupuesto, int):
            score += 1 if price <= presupuesto else -3

        if score > 0:
            candidatos.append((score, code))

    candidatos.sort(reverse=True)
    if shuffle:
        random.shuffle(candidatos)

    out = []
    for _, code in candidatos[:k]:
        p = PRODUCTOS[code]
        out.append({
            "code": code,
            "name": p.get("name"),
            "price": p.get("price"),
            "img_url": p.get("img_url"),
            "url": p.get("url"),
            "tags": p.get("tags", []),
            "categories": p.get("categories", []),
        })
    return out

# ================== ESTADO ==================
conversation_state: Dict[str, dict] = {}
HISTORY: Dict[str, list] = {}

def get_state(session_id="default"):
    if session_id not in conversation_state:
        conversation_state[session_id] = {
            "espacio": None,
            "tipo": None,
            "tamano": None,
            "presupuesto": None,
            "estilo": None,
            "instalacion": None,
            "mostrados": set(),
            "last_space_from_user": None,
        }
    if not isinstance(conversation_state[session_id].get("mostrados"), set):
        conversation_state[session_id]["mostrados"] = set(conversation_state[session_id]["mostrados"] or [])
    return conversation_state[session_id]

def get_history(session_id: str) -> list[dict]:
    h = HISTORY.get(session_id) or []
    return h[-20:]

def push_history(session_id: str, role: str, content: str):
    HISTORY.setdefault(session_id, []).append({"role": role, "content": content})

# ================== TOOLS (para tool-calling) ==================
def tool_search_products(args: dict, state: dict):
    q = args.get("query", "")
    k = int(args.get("k", 5))
    presu = state.get("presupuesto")
    must = must_have_from_state(state, q)
    excl = state.get("mostrados") or set()
    prods = buscar_productos(q, k=k, must_have=must, presupuesto=presu, exclude_codes=excl)
    for p in prods:
        state["mostrados"].add(p["code"])
    return {"items": prods}

def tool_parse_budget(args: dict, state: dict):
    v = args.get("text")
    return {"budget": parse_presupuesto(v)}

TOOLS_SPEC = [
  {
    "type": "function",
    "function": {
      "name": "search_products",
      "description": "Busca productos del catálogo según el texto del usuario.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {"type": "string"},
          "k": {"type": "integer", "default": 5}
        },
        "required": ["query"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "parse_budget",
      "description": "Interpreta presupuesto COP desde texto libre.",
      "parameters": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"]
      }
    }
  }
]

def call_tool(name: str, arguments: dict, state: dict):
    if name == "search_products":
        return tool_search_products(arguments, state)
    if name == "parse_budget":
        return tool_parse_budget(arguments, state)
    return {"error": f"unknown tool {name}"}

# ================== SYSTEM PROMPT ==================
ASSISTANT_SYSTEM = """Eres un asesor de iluminación de Ecolite con tono natural (tipo WhatsApp, 1–2 frases).
No hagas checklist ni flujos rígidos; conversa como persona, muestra empatía y adapta el ritmo.
Puedes hacer preguntas abiertas si falta contexto.
Solo recomiendas productos que EXISTEN en productos.json (no inventes SKUs).
Cuando detectes intención clara de catálogo o comparación, llama search_products con el texto del usuario.
Si el usuario menciona precio, puedes llamar parse_budget.
Si no hay intención de catálogo, responde conversacionalmente (sin productos).
Nunca uses markdown, ni bullets, ni asteriscos.
Mantén respuestas breves y profesionales con acento colombiano.
"""

FEWSHOT = [
  {"role": "user", "content": "hola"},
  {"role": "assistant", "content": "hola, ¿qué espacio quieres iluminar?"},
  {"role": "user", "content": "necesito algo para una piscina grande, presupuesto 500 mil"},
  {"role": "assistant", "content": "de una, busco opciones sumergibles dentro de ese tope."},
]

# ================== LOOP DE TOOL-CALLING ==================
def assistant_turn(messages: List[dict], state: dict) -> dict:
    """
    Devuelve {"reply": str, "productos": list}
    """
    # Fallback local cuando no hay OpenAI: conversa simple + heurística de búsqueda
    if not (OPENAI_OK and client):
        txt = messages[-1]["content"] if messages else ""
        # mini heurística: si menciona categoría/espacio, buscamos
        heur_buscar = any(w in _norm(txt) for w in ["piscina","empotr","superficie","escritorio","reflector","driver","panel","plafon","cinta","tira"])
        productos = buscar_productos(txt, k=5, must_have=must_have_from_state(state, txt), presupuesto=state.get("presupuesto")) if heur_buscar else []
        if productos:
            return {"reply": "listo, te dejo opciones que encajan con lo que dices", "productos": productos}
        return {"reply": "listo, cuéntame qué necesitas iluminar y el estilo que te gusta", "productos": []}

    # 1) el modelo decide si llama tools
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.75,
        messages=[{"role":"system","content":ASSISTANT_SYSTEM}] + FEWSHOT + messages,
        tools=TOOLS_SPEC,
        tool_choice="auto",
    )
    msg = r.choices[0].message

    productos = []
    if getattr(msg, "tool_calls", None):
        tool_msgs = []
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            result = call_tool(name, args, state)
            if name == "search_products":
                productos = result.get("items") or []
            if name == "parse_budget" and result.get("budget") is not None:
                state["presupuesto"] = result["budget"]
            tool_msgs.extend([
                {"role": "assistant", "tool_calls": [tc]},
                {"role": "tool", "tool_call_id": tc.id, "name": name, "content": json.dumps(result, ensure_ascii=False)}
            ])
        # 2) redacción final
        r2 = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.75,
            messages=[{"role":"system","content":ASSISTANT_SYSTEM}] + FEWSHOT + messages + tool_msgs
        )
        final_text = r2.choices[0].message.content.strip()
        return {"reply": final_text, "productos": productos}

    # 3) pura charla si no pidió tools
    text = (msg.content or "").strip() or "dale, cuéntame un poco más para orientarte bien"
    return {"reply": text, "productos": []}

# ================== MODELOS DE ENTRADA ==================
class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    lead: Optional[dict] = None

@app.get("/session")
def new_session():
    return {"session_id": str(uuid.uuid4())}

# ================== SSE opcional para “typing” ==================
@app.get("/stream")
async def stream(request: Request, q: str = "", session_id: str = "default"):
    async def event_gen():
        yield "data: " + json.dumps({"typing": True}) + "\n\n"
        state = get_state(session_id)
        productos = buscar_productos(q, k=3, must_have=must_have_from_state(state, q), presupuesto=state.get("presupuesto"))
        for p in productos:
            state["mostrados"].add(p["code"])
        reply = "te paso unas opciones rápidas"
        payload = {"typing": False, "reply": reply, "productos": productos}
        yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")

# ================== CHAT ENDPOINT ==================
@app.post("/chat")
def chat(in_: ChatIn):
    session_id = in_.session_id or "default"
    state = get_state(session_id)
    user_msg = in_.message.strip()

    push_history(session_id, "user", user_msg)
    turn = assistant_turn(get_history(session_id), state)
    reply = turn["reply"]
    productos = turn.get("productos") or []
    push_history(session_id, "assistant", reply)
    for p in productos:
        state["mostrados"].add(p["code"])
    return {"reply": reply, "productos": productos}

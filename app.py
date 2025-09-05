import os, re, json, unicodedata, uuid, random, time
from collections import defaultdict, deque
from typing import Dict, Any, List, Set, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# ---------------- OpenAI (opcional) ----------------
# Nota: mantenemos la dependencia pero hacemos el uso más robusto
try:
    from openai import OpenAI
    _OPENAI_IMPORTED = True
except Exception:
    OpenAI = None  # type: ignore
    _OPENAI_IMPORTED = False

# ---------------- DOCX (opcional) ----------------
try:
    from docx import Document as DocxDocument
    _DOCX_OK = True
except Exception:
    DocxDocument = None  # type: ignore
    _DOCX_OK = False

# ---------------- SETUP ----------------
load_dotenv()

# Config básica desde env (sin pydantic-settings para no añadir deps)
APP_ENV = os.getenv("APP_ENV", "dev")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
STATIC_DIR = os.getenv("STATIC_DIR", "public")
PRODUCTS_PATH = os.getenv("PRODUCTS_PATH", "productos.json")
DOCX_PATH = os.getenv("DOCX_PATH", "data/catalogo.docx")
CORS_ALLOWED = [o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
MAX_MSG_LEN = int(os.getenv("MAX_MSG_LEN", "800"))

# Cliente OpenAI (si existe clave y lib)
client: Optional[OpenAI] = None
if _OPENAI_IMPORTED and OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None

# ---- estado global (en memoria; ideal migrar a Redis para producción) ----
conversation_state: Dict[str, dict] = {}
rate_buckets: Dict[str, deque] = defaultdict(deque)

# ---------------- APP ----------------
app = FastAPI(title="Chatbot Ecolite API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------- UTILIDADES ----------------
def _norm(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _ip(request: Request) -> str:
    # X-Forwarded-For primero si estás detrás de proxy
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    client_host = request.client.host if request.client else "unknown"
    return client_host or "unknown"


def _rate_limit(key: str):
    now = time.time()
    bucket = rate_buckets[key]
    # limpia >60s
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    bucket.append(now)


def log_event(event: str, **kwargs):
    try:
        payload = {"event": event, **kwargs, "ts": round(time.time(), 3)}
        print(json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass


# ---------------- DATA -------------------
PRODUCTOS: Dict[str, dict] = {}


def _load_productos(path: str) -> Dict[str, dict]:
    if not os.path.exists(path):
        log_event("products_missing", path=path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # validación ligera
            if not isinstance(data, dict):
                return {}
            cleaned = {}
            for code, p in data.items():
                if not isinstance(p, dict):
                    continue
                cleaned[code] = {
                    "name": p.get("name"),
                    "price": p.get("price"),
                    "img_url": p.get("img_url"),
                    "url": p.get("url"),
                    "tags": p.get("tags", []),
                    "categories": p.get("categories", []),
                }
            return cleaned
    except Exception as e:
        log_event("products_load_error", error=str(e))
        return {}


PRODUCTOS = _load_productos(PRODUCTS_PATH)

DOCX_TEXT = ""
if _DOCX_OK and os.path.exists(DOCX_PATH):
    try:
        d = DocxDocument(DOCX_PATH)  # type: ignore
        DOCX_TEXT = "\n".join(p.text for p in d.paragraphs if p.text.strip())
    except Exception:
        DOCX_TEXT = ""


# -------------- PROMPTS ----------------
SYSTEM_PROMPT_CONDUCTOR = """
Eres asesor de iluminación de Ecolite. Responde en 1–2 frases, tono WhatsApp, claro y amable (acento colombiano).
Guía SIEMPRE este flujo, sin decir que existe un flujo:

FORMULARIO (datos)
1) ESPACIO (ambiente/lugar requiere la iluminación)
   - Si falta, pregúntalo: “¿En qué espacio sería la instalación (ej: oficina, pasillo, piscina…)?”

2) ¿QUIERES QUE TE SUGIERA O TIENES UNA LUMINARIA EN ESPECÍFICO?
   - Si ya hay ESPACIO y no hay MODO, pregunta: “¿Quieres que te sugiera o tienes una luminaria en específico?”
   - Si el usuario dice que QUIERE SUGERENCIAS → action=SHOW_SUGGESTIONS (5 productos aleatorios del ESPACIO).
   - Si el usuario dice que TIENE UNA EN ESPECÍFICO o da código/nombre → action=SEARCH_SPECIFIC.

3) SI → QUIERE QUE LE SUGIERA (después de mostrar 5)
   - Pregunta: “¿Te sirve alguna de las luminarias sugeridas o estás buscando algo diferente?”
   - Si contesta que SÍ le sirve alguna → action=COLLECT_SPECS (pide, en ese orden, Temperatura, Vatios y Tipo de instalación: incrustar o sobreponer; una sola pregunta por turno).
   - Si dice que busca diferente → action=ASK_MODE (redirige: “¿Quieres que te sugiera algo distinto o tienes una luminaria en específico?”).

4) SI → TIENE UNA EN ESPECÍFICO
   - action=SEARCH_SPECIFIC (buscar por el ambiente indicado por el usuario o por el nombre/código del producto que menciona).

Reglas:
- NUNCA dispares SHOW_SUGGESTIONS hasta que el usuario responda que QUIERE SUGERENCIAS.
- NUNCA inventes productos. El backend se encarga de mostrar resultados.
- Si el usuario saluda o charla, responde breve y retoma el PASO que falte.
- NO uses markdown ni listas.
- Si el usuario pide “más”, usa action=MORE_SUGGESTIONS y el reply exacto:
  "Te muestro 5 más 👌"
- Evita repetir el espacio (“oficina”, “piscina”) en la misma respuesta si ya quedó claro.
- No agregues preguntas extra en el mismo turno de MORE_SUGGESTIONS.


Devuelve SOLO JSON con:
{
  "reply": string,
  "action": "ASK_SPACE" | "ASK_MODE" | "SHOW_SUGGESTIONS" | "ASK_AFTER_SUGGESTIONS" | "COLLECT_SPECS" | "SEARCH_SPECIFIC" | "MORE_SUGGESTIONS" | "SMALLTALK" | "NONE",
  "space": string|null,
  "mode": "sugerir"|"especifico"|null,
  "spec_field": "temperatura"|"vatios"|"instalacion"|null
}
"""

REPLY_PROMPT = """
Eres asesor de iluminación de Ecolite.
Habla natural, tono WhatsApp (1–2 frases), breve y amable (acento colombiano).
Usa el contexto del usuario si lo hay, pero no inventes productos (eso lo pone el backend).
Si es saludo o charla ligera → responde casual.
Si falta info (espacio, tipo, presupuesto) → pide ese único dato con naturalidad.
Si ya hay productos (el backend te los pasa), comenta corto invitando a verlos.
Nunca uses markdown, viñetas ni texto largo.
"""

# -------------- NORM & GUARDS ----------
# SKU: exige al menos un dígito (evita “hola”)
code_regex   = re.compile(r"\b(?=[A-Z0-9-]{3,15}\b)(?=[A-Z0-9-]*\d)[A-Z0-9-]+\b")
CODE_SOFT_RE = re.compile(r"[A-Z]{2,}\d+[A-Z0-9-]*", re.I)

# Heurísticas para intención
SUGERIR_RE = re.compile(r"\b(sug|sugi|sugie|sugier|sugiere|sugi[eé]reme|sugerir|sugerencia|recomiend|recomienda|sugiere|recomiende|sugiera|recomi[eé]ndame)\w*\b", re.I)
ESPECIFICO_RE = re.compile(r"\b(especific|espec[ií]fica|tengo\s+una|ya\s+tengo|c[oó]digo|codigo|referencia|sku|modelo|nombre)\w*\b", re.I)


def said_suggest(txt: str) -> bool:
    return bool(SUGERIR_RE.search(txt or ""))


def said_specific(txt: str) -> bool:
    return bool(ESPECIFICO_RE.search(txt or ""))


# -------------- RAG LIGERO (señales) -------------
SIGNALS = {
    "piscina": {"piscina", "sumergible", "ip68", "bajo agua", "bajo_agua"},
    "oficina": {"oficina", "panel", "lineal", "downlight"},
    "exterior": {"exterior", "ip65", "ip66", "fachada", "terraza", "jardin", "jardín", "proyector", "aplique", "poste"},
}

DOCX_N = _norm(DOCX_TEXT)
if DOCX_N:
    if "piscina" in DOCX_N:
        SIGNALS["piscina"] |= {"nicho", "pentair", "12v", "rgb"}
    if "oficina" in DOCX_N:
        SIGNALS["oficina"] |= {"ugr", "anti deslumbramiento", "uniformidad"}
    if any(k in DOCX_N for k in ["exterior", "fachada", "terraza", "jardin", "jardín"]):
        SIGNALS["exterior"] |= {"baliza"}


def must_from_text(txt: str) -> Set[str]:
    t = _norm(txt)
    must: Set[str] = set()
    if any(s in t for s in SIGNALS["piscina"]):
        must |= {"piscina", "sumergible"}
    if any(s in t for s in SIGNALS["oficina"]):
        must |= {"oficina"}
    if any(s in t for s in SIGNALS["exterior"]):
        must |= {"ip65"}
    return must


# ---------- BÚSQUEDA ESPECÍFICA (código / nombre) ----------
CODE_SOFT_RE = re.compile(r"[A-Z]{2,}\d+[A-Z0-9\-]*", re.I)


def extraer_codigo(txt: str) -> Optional[str]:
    """
    Devuelve el código más probable dentro del mensaje.
    Requiere al menos un dígito DENTRO de la misma palabra (evita falsos positivos como 'muestrame').
    """
    if not txt:
        return None
    q = txt.upper()
    matches = list(code_regex.finditer(q)) + list(CODE_SOFT_RE.finditer(q))
    # Filtra cualquier match que no tenga dígito en el propio token
    matches = [m for m in matches if any(ch.isdigit() for ch in m.group(0))]
    if not matches:
        return None
    # prioriza por longitud y por estar más al final del texto
    matches.sort(key=lambda m: (len(m.group(0)), m.start()), reverse=True)
    return matches[0].group(0)


def buscar_producto_especifico(query_text: str, space: Optional[str] = None, k: int = 5) -> List[dict]:
    q = (query_text or "").strip()
    qn = _norm(q)
    q_upper = q.upper()

    code_token = extraer_codigo(q_upper)

    code_prefix = None
    if code_token:
        m_pref = re.match(r"([A-Z]+)", code_token)
        code_prefix = m_pref.group(1) if m_pref else None

    name_tokens = [w for w in re.split(r"\s+", qn) if len(w) > 2]

    piscina_mode = (space and _norm(space) == "piscina")

    def score_item(code: str, prod: dict) -> int:
        name = prod.get("name") or ""
        name_n = _norm(name)
        tags_n = _norm(" ".join(map(str, prod.get("tags", []))))
        cats_n = _norm(" ".join(map(str, prod.get("categories", []))))
        full_n = f"{name_n} {tags_n} {cats_n}"

        if piscina_mode and not (("sumergible" in full_n) or ("ip68" in full_n)):
            return 0

        s = 0
        cu = code.upper()

        if code_token:
            if cu == code_token:
                s += 100
            elif cu.startswith(code_token):
                s += 70
            elif code_token in cu:
                s += 45

        if code_prefix:
            if cu.startswith(code_prefix):
                s += 35
            else:
                s -= 10

        for t in name_tokens:
            if t in name_n:
                s += 14
            if t in tags_n:
                s += 7
            if t in cats_n:
                s += 4

        if space:
            sp = _norm(space)
            if sp in full_n:
                s += 8
            if sp == "piscina" and (("sumergible" in full_n) or ("ip68" in full_n)):
                s += 10

        return s

    candidatos: List[tuple[int, str]] = []
    for code, p in PRODUCTOS.items():
        sc = score_item(code, p)
        if sc > 0:
            candidatos.append((sc, code))

    candidatos.sort(reverse=True)

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


# -------------- BUSCADOR (general) ---------------

def buscar_productos(
    query_text: str,
    k: int = 5,
    must_have: Optional[Set[str]] = None,
    presupuesto: Optional[int] = None,
    exclude_codes: Optional[Set[str]] = None,
):
    qn = _norm(query_text)
    words = [w for w in qn.split() if len(w) > 2]
    must_have = {_norm(x) for x in (must_have or set())}
    exclude_codes = exclude_codes or set()

    out: List[tuple[int, str]] = []
    for code, p in PRODUCTOS.items():
        if code in exclude_codes:
            continue
        name = _norm(p.get("name", ""))
        tags = _norm(" ".join([str(t) for t in p.get("tags", [])]))
        cats = _norm(" ".join([str(c) for c in p.get("categories", [])]))
        full = f"{name} {tags} {cats}"
        price = p.get("price")

        ok = True
        for m in must_have:
            if m and m not in full:
                ok = False
                break
        if not ok:
            continue

        hit, score = False, 0
        for w in words:
            if w in name:
                score += 6
                hit = True
            if w in tags:
                score += 4
                hit = True
            if w in cats:
                score += 2
                hit = True

        if hit or must_have:
            for b in {"sumergible", "ip68", "piscina", "rgb", "12v", "panel", "downlight", "lineal", "proyector", "poste", "aplique", "ip65"}:
                if b in full:
                    score += 2

        if isinstance(price, (int, float)) and isinstance(presupuesto, int):
            score += 1 if price <= presupuesto else -2

        if not hit and not must_have:
            continue
        if score > 0:
            out.append((score, code))

    out.sort(reverse=True)
    res = []
    for _, code in out[:k]:
        p = PRODUCTOS[code]
        res.append({
            "code": code,
            "name": p.get("name"),
            "price": p.get("price"),
            "img_url": p.get("img_url"),
            "url": p.get("url"),
            "tags": p.get("tags", []),
            "categories": p.get("categories", []),
        })
    return res


def sugerencias_aleatorias(space: Optional[str], n: int = 5, exclude_codes: Optional[Set[str]] = None):
    exclude_codes = exclude_codes or set()

    def fits(prod: dict, space_k: str) -> bool:
        texto = _norm(
            (prod.get("name") or "")
            + " " + " ".join(map(str, prod.get("tags", [])))
            + " " + " ".join(map(str, prod.get("categories", [])))
        )
        if not space_k:
            return True
        # Piscina: endurece a productos sumergibles o con IP68
        if space_k == "piscina":
            return ("sumergible" in texto) or ("ip68" in texto) or ("piscina" in texto)
        return space_k in texto

    keys = [k for k, p in PRODUCTOS.items() if k not in exclude_codes and fits(p, _norm(space or ""))]
    if not keys:
        keys = [k for k in PRODUCTOS.keys() if k not in exclude_codes]

    random.shuffle(keys)
    out = []
    for code in keys[:n]:
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


# -------------- Parsers de SPECs / presupuesto ----------

def parse_temperatura(s: str) -> Optional[str]:
    t = _norm(s)
    if any(w in t for w in ["calida", "cálida", "warm", "3000", "2700", "3000k", "2700k"]):
        return "cálida"
    if any(w in t for w in ["neutra", "neutral", "4000", "4000k"]):
        return "neutra"
    if any(w in t for w in ["fria", "fría", "cool", "6000", "6500", "6000k", "6500k"]):
        return "fría"
    return None


def parse_vatios(s: str) -> Optional[int]:
    m = re.search(r"(\d{1,3})\s*w", _norm(s))
    return int(m.group(1)) if m else None


def parse_instalacion(s: str) -> Optional[str]:
    t = _norm(s)
    if "incrust" in t or "empotr" in t:
        return "incrustar"
    if "sobrep" in t or "superficie" in t:
        return "sobreponer"
    return None


_BUDGET_RE = re.compile(r"(\$|cop|col\$)?\s*([0-9][0-9\.,\s]*)(k|mil|m|millones)?", re.I)


def parse_presupuesto(s: str) -> Optional[int]:
    t = _norm(s)
    m = _BUDGET_RE.search(t)
    if not m:
        return None
    num = m.group(2).replace(" ", "").replace(".", "").replace(",", "")
    try:
        base = int(num)
    except Exception:
        return None
    suf = (m.group(3) or "").lower()
    if suf in {"k", "mil"}:
        base *= 1000
    elif suf in {"m", "millones"}:
        base *= 1_000_000
    return base


# -------------- Detección de espacio (helper) --------------
SPACE_CANON = {
    "piscina": "piscina", "oficina": "oficina", "pasillo": "pasillo", "bodega": "bodega", "sala": "sala", "cocina": "cocina",
    "baño": "baño", "banio": "baño", "terraza": "terraza", "jardin": "jardin", "jardín": "jardin", "fachada": "fachada",
    "parqueadero": "parqueadero", "retail": "retail", "industrial": "industrial", "dormitorio": "dormitorio", "escalera": "escalera",
}

def detect_space_from_text(text: str) -> Optional[str]:
    t = _norm(text)
    for k in SPACE_CANON.keys():
        if k in t:
            return SPACE_CANON[k]
    return None

# -------------- LLM ORCHESTRATION ----------

def _parse_json_loose(text: str) -> Optional[dict]:
    if not text:
        return None
    s = text.strip()
    # quita fences ```json ... ``` o ``` ... ```
    if s.startswith("```"):
        s = s.strip("` ")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    try:
        return json.loads(s)
    except Exception:
        # intenta extraer el primer bloque {...}
        try:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(s[start : end + 1])
        except Exception:
            return None
    return None


def llm_turn(user_text: str, state: dict) -> dict:
    """Orquestador con fallback si no hay OpenAI o si da JSON inválido."""
    lite_state = {
        "espacio": state.get("espacio"),
        "modo": state.get("modo"),
        "temperatura": state.get("temperatura"),
        "vatios": state.get("vatios"),
        "instalacion": state.get("instalacion"),
        "presupuesto": state.get("presupuesto"),
    }

    if client is None:
        # Fallback sin LLM
        reply = "Listo, ¿en qué espacio sería la instalación (piscina, oficina, etc.)?"
        return {"reply": reply, "action": "ASK_SPACE", "space": None, "mode": None, "spec_field": None}

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT_CONDUCTOR},
        {"role": "user", "content": f"Usuario: {user_text}\nEstado:{json.dumps(lite_state, ensure_ascii=False)}"},
    ]

    try:
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            max_tokens=200,
            messages=msgs,
        )
        raw = (r.choices[0].message.content or "").strip()
        data = _parse_json_loose(raw) or {}
        if not data:
            raise ValueError("invalid_json")
        return {
            "reply": data.get("reply") or "Listo.",
            "action": data.get("action") or "NONE",
            "space": data.get("space"),
            "mode": data.get("mode"),
            "spec_field": data.get("spec_field"),
        }
    except Exception as e:
        log_event("llm_error", error=str(e))
        return {
            "reply": "Listo, ¿en qué espacio sería la instalación (piscina, oficina, etc.)?",
            "action": "ASK_SPACE",
            "space": None,
            "mode": None,
            "spec_field": None,
        }


# -------------- ESTADO / CHAT ----------
class ChatIn(BaseModel):
    message: str
    lead: Optional[dict] = None
    session_id: Optional[str] = None


def get_state(session_id: str):
    if session_id not in conversation_state:
        conversation_state[session_id] = {
            "espacio": None,
            "modo": None,
            "temperatura": None,
            "vatios": None,
            "instalacion": None,
            "tipo": None,
            "presupuesto": None,
            "mostrados": set(),
        }
    if not isinstance(conversation_state[session_id]["mostrados"], set):
        conversation_state[session_id]["mostrados"] = set(conversation_state[session_id]["mostrados"] or [])
    return conversation_state[session_id]


@app.get("/")
def home():
    index_path = os.path.join(STATIC_DIR, "chatbox.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"ok": True, "app": app.title, "version": app.version}


@app.get("/health")
def health():
    return {
        "ok": True,
        "env": APP_ENV,
        "openai": bool(client is not None),
        "products": len(PRODUCTOS),
        "version": app.version,
    }


@app.post("/reset")
def reset(in_: ChatIn):
    sid = (in_.session_id or "default").strip() or "default"
    conversation_state.pop(sid, None)
    return {"ok": True, "session_id": sid}


@app.get("/session")
def new_session():
    return {"session_id": str(uuid.uuid4())}


@app.post("/chat")
async def chat(in_: ChatIn, request: Request):
    # --- Validaciones y rate limit ---
    if not in_.message or not in_.message.strip():
        raise HTTPException(status_code=400, detail="message is required")
    if len(in_.message) > MAX_MSG_LEN:
        raise HTTPException(status_code=400, detail="message too long")

    ip = _ip(request)
    _rate_limit(ip)

    # --- Session ---
    session_id = (in_.session_id or request.headers.get("x-session-id") or "default").strip() or "default"
    st = get_state(session_id)

    user = in_.message.strip()

    # --- Presupuesto si lo menciona ---
    budget = parse_presupuesto(user)
    if isinstance(budget, int):
        st["presupuesto"] = budget

    brain = None

    # --- Detectar cambio de espacio en cualquier momento (ej: "ahora para una piscina") ---
    new_space = detect_space_from_text(user)  # helper agregado
    if new_space:
        if st.get("espacio") != new_space:
            # Reset suave al cambiar de espacio
            st["espacio"] = new_space
            st["modo"] = None
            st["temperatura"] = None
            st["vatios"] = None
            st["instalacion"] = None
            st["mostrados"].clear()

        # Decide la acción según el mismo mensaje
        if said_suggest(user):
            st["modo"] = "sugerir"
            brain = {
                "reply": "Listo, te muestro 5 opciones 👌",
                "action": "SHOW_SUGGESTIONS",
                "space": st["espacio"],
                "mode": "sugerir",
                "spec_field": None,
            }
        elif said_specific(user) or code_regex.search(user) or CODE_SOFT_RE.search(user):
            st["modo"] = "especifico"
            brain = {
                "reply": "Busco esa referencia y te muestro lo que encontré 👌",
                "action": "SEARCH_SPECIFIC",
                "space": st["espacio"],
                "mode": "especifico",
                "spec_field": None,
            }
        else:
            brain = {
                "reply": "¿Quieres que te sugiera o tienes una luminaria en específico?",
                "action": "ASK_MODE",
                "space": st["espacio"],
                "mode": st.get("modo"),
                "spec_field": None,
            }

    # --- Heurística inicial si aún no hay decisión ---
    if brain is None:
        if st.get("espacio") and not st.get("modo"):
            if said_suggest(user):
                st["modo"] = "sugerir"
                brain = {
                    "reply": "Listo, te muestro 5 opciones 👌",
                    "action": "SHOW_SUGGESTIONS",
                    "space": st["espacio"],
                    "mode": "sugerir",
                    "spec_field": None,
                }
            elif said_specific(user) or code_regex.search(user) or CODE_SOFT_RE.search(user):
                st["modo"] = "especifico"
                brain = {
                    "reply": "Busco esa referencia y te muestro lo que encontré 👌",
                    "action": "SEARCH_SPECIFIC",
                    "space": st["espacio"],
                    "mode": "especifico",
                    "spec_field": None,
                }
            else:
                brain = llm_turn(user, st)
        else:
            brain = llm_turn(user, st)

    # --- Actualiza estado con lo que devuelva el LLM (solo si faltaba) ---
    if brain.get("space") and not st.get("espacio"):
        st["espacio"] = _norm(brain["space"]) or None
    if brain.get("mode") and not st.get("modo"):
        st["modo"] = brain["mode"]

    # --- Fallback: detectar espacio directo si aún falta ---
    if not st.get("espacio"):
        t = _norm(user)
        for k in [
            "piscina", "oficina", "pasillo", "bodega", "sala", "cocina",
            "baño", "banio", "terraza", "jardin", "jardín", "fachada",
            "parqueadero", "retail", "industrial", "dormitorio", "escalera",
        ]:
            if k in t:
                st["espacio"] = "baño" if k == "banio" else ("jardin" if k == "jardín" else k)
                if brain.get("action") == "ASK_SPACE":
                    brain = {
                        "reply": "¿Quieres que te sugiera o tienes una luminaria en específico?",
                        "action": "ASK_MODE",
                        "space": st["espacio"],
                        "mode": st.get("modo"),
                        "spec_field": None,
                    }
                break

    # --- Recolección de specs si aplica ---
    if brain.get("action") == "COLLECT_SPECS":
        tval = parse_temperatura(user)
        wval = parse_vatios(user)
        ival = parse_instalacion(user)
        if tval: st["temperatura"] = tval
        if wval: st["vatios"] = wval
        if ival: st["instalacion"] = ival

    # --- Ejecutar acciones y armar productos ---
    productos: List[dict] = []
    action = brain.get("action", "NONE")

    if action == "SHOW_SUGGESTIONS":
        productos = sugerencias_aleatorias(st.get("espacio"), n=5, exclude_codes=st["mostrados"])
        for p in productos:
            st["mostrados"].add(p["code"])  # type: ignore
        # microcopy corta (no repetir espacio)
        brain["reply"] = "Listo, te muestro 5 opciones 👌"

    elif action == "MORE_SUGGESTIONS":
        productos = sugerencias_aleatorias(st.get("espacio"), n=5, exclude_codes=st["mostrados"])
        for p in productos:
            st["mostrados"].add(p["code"])  # type: ignore
        # forzar respuesta mínima
        brain["reply"] = "Te muestro 5 más 👌"

    elif action == "SEARCH_SPECIFIC":
        productos = buscar_producto_especifico(
            query_text=user,
            space=st.get("espacio"),
            k=5,
        )
        if not productos:
            must = must_from_text(st.get("espacio") or "")
            productos = buscar_productos(
                query_text=user,
                k=5,
                must_have=must,
                presupuesto=st.get("presupuesto"),
                exclude_codes=st["mostrados"],
            )
        for p in productos:
            st["mostrados"].add(p["code"])  # type: ignore

        code_tok = extraer_codigo(user)
        brain["reply"] = (
            f"Esto es lo que encontré para {code_tok} 👇"
            if code_tok else
            "Te muestro las coincidencias que encontré 👇"
        )

    # --- Logging mínimo (sin PII sensible) ---
    log_event(
        "chat_turn",
        session_id=session_id,
        ip=ip,
        action=action,
        espacio=st.get("espacio"),
        modo=st.get("modo"),
        specs={"t": st.get("temperatura"), "w": st.get("vatios"), "i": st.get("instalacion")},
        productos=len(productos),
    )

    return {
        "reply": brain.get("reply", "Listo."),
        "productos": productos,
        "session_id": session_id,
    }

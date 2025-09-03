import os, re, json, unicodedata, uuid, random
from typing import Dict, Any, List, Set
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from docx import Document as DocxDocument

# ---------------- SETUP ----------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- estado global (declarado antes de usar en /) ----
conversations: Dict[str, list] = {}
conversation_state: Dict[str, dict] = {}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="public"), name="static")

@app.get("/")
def home():
    conversations.clear(); conversation_state.clear()
    return FileResponse("public/chatbox.html") if os.path.exists("public/chatbox.html") else {"ok": True}

@app.get("/health")
def health(): return {"ok": True}

# -------------- DATA -------------------
with open("productos.json", "r", encoding="utf-8") as f:
    PRODUCTOS: Dict[str, dict] = json.load(f)

DOCX_PATH = "data/catalogo.docx"
DOCX_TEXT = ""
if os.path.exists(DOCX_PATH):
    try:
        d = DocxDocument(DOCX_PATH)
        DOCX_TEXT = "\n".join(p.text for p in d.paragraphs if p.text.strip())
    except Exception:
        DOCX_TEXT = ""


# -------------- PROMPTS ----------------
SYSTEM_PROMPT = """
Eres un asesor de iluminación de Ecolite con tono natural (tipo WhatsApp, 1–2 frases).
No hagas checklist ni flujos rígidos; conversa como persona, muestra empatía y adapta el ritmo.
Puedes hacer preguntas abiertas si falta contexto.
Solo recomiendas productos que EXISTEN en productos.json (no inventes SKUs).
Cuando detectes intención clara de catálogo o comparación, llama search_products con el texto del usuario.
Si el usuario menciona precio, puedes llamar parse_budget.
Si no hay intención de catálogo, responde conversacionalmente (sin productos).
Nunca uses markdown, ni bullets, ni asteriscos.
Mantén respuestas breves y profesionales con acento colombiano.
"""

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

INTENT_PROMPT = """
Clasifica el mensaje del usuario. Devuelve SOLO este JSON:
{
  "intent": "greeting|catalog|more|price|availability|clarify|other",
  "space_hint": string|null,
  "type_hint": string|null
}
greeting: hola/qué tal.
catalog: pide productos, dice “sugiéreme”, menciona potencia/código/marca.
more: pide más opciones.
price: presupuesto/precio.
availability: stock/entrega.
clarify: pide un dato faltante (empotrado/superficie/sumergible, etc.).
other: charla general (no catálogo).
Si dudas, usa "other".
"""


# -------------- NORM & GUARDS ----------
def _norm(s: Any) -> str:
    if s is None: return ""
    s = str(s).lower()
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

# SKU: exige al menos un dígito (evita “hola”)
code_regex = re.compile(r"\b(?=[A-Z0-9\-]{3,15}\b)(?=.*\d)[A-Z0-9\-]+\b")

# --- Heurísticas robustas para capturar intención con typos ---
SUGERIR_RE = re.compile(r"\b(sug|sugi|sugie|sugier|sugiere|sugi[eé]reme|sugerir|sugerencia|recomiend|recomienda|recomi[eé]ndame)\w*\b", re.I)
ESPECIFICO_RE = re.compile(r"\b(especific|espec[ií]fica|tengo\s+una|ya\s+tengo|c[oó]digo|codigo|referencia|sku|modelo|nombre)\w*\b", re.I)

def said_suggest(txt: str) -> bool:
    return bool(SUGERIR_RE.search(txt or ""))

def said_specific(txt: str) -> bool:
    return bool(ESPECIFICO_RE.search(txt or ""))

# -------------- RAG LIGERO (señales) -------------
SIGNALS = {
    "piscina": {"piscina","sumergible","ip68","bajo agua","bajo_agua"},
    "oficina": {"oficina","panel","lineal","downlight"},
    "exterior": {"exterior","ip65","ip66","fachada","terraza","jardin","jardín","proyector","aplique","poste"},
}
DOCX_N = _norm(DOCX_TEXT)
if DOCX_N:
    if "piscina" in DOCX_N:       SIGNALS["piscina"] |= {"nicho","pentair","12v","rgb"}
    if "oficina" in DOCX_N:       SIGNALS["oficina"] |= {"ugr","anti deslumbramiento","uniformidad"}
    if any(k in DOCX_N for k in ["exterior","fachada","terraza","jardin","jardín"]):
        SIGNALS["exterior"] |= {"baliza"}

def must_from_text(txt: str) -> Set[str]:
    t = _norm(txt)
    must = set()
    if any(s in t for s in SIGNALS["piscina"]):  must |= {"piscina","sumergible"}
    if any(s in t for s in SIGNALS["oficina"]):  must |= {"oficina"}
    if any(s in t for s in SIGNALS["exterior"]): must |= {"ip65"}
    return must

# ---------- BÚSQUEDA ESPECÍFICA (código / nombre) ----------
CODE_SOFT_RE = re.compile(r"[A-Z]{2,}\d+[A-Z0-9\-]*", re.I)

# Detectar posibles SKUs en el texto (último y más largo es mejor)
CODE_SOFT_RE = re.compile(r"[A-Z]{2,}\d+[A-Z0-9\-]*", re.I)

def extraer_codigo(txt: str) -> str | None:
    """
    Devuelve el código más probable:
    - Debe tener letras+al menos un dígito.
    - Si hay varios, toma el más largo.
    - Toma SIEMPRE el ÚLTIMO match (para evitar palabras previas como 'quiero').
    """
    if not txt:
        return None
    q = txt.upper()
    matches = list(code_regex.finditer(q)) + list(CODE_SOFT_RE.finditer(q))
    if not matches:
        return None
    # prioriza por longitud y por estar más al final del texto
    matches.sort(key=lambda m: (len(m.group(0)), m.start()), reverse=True)
    return matches[0].group(0)


def buscar_producto_especifico(query_text: str, space: str | None = None, k: int = 5) -> List[dict]:
    """
    Prioriza coincidencias por código exacto o variaciones cercanas.
    Reglas extra:
      - Afinidad por prefijo de código (mismo prefijo del código pedido).
      - Si el espacio es 'piscina' → filtra duro a productos 'sumergible' o 'ip68'.
    """
    q = (query_text or "").strip()
    qn = _norm(q)
    q_upper = q.upper()

    # Token de código "principal"
    code_token = extraer_codigo(q_upper)

    # prefijo alfabético del código (p.ej. ECOPL de ECOPL24W)
    code_prefix = None
    if code_token:
        m_pref = re.match(r"([A-Z]+)", code_token)
        code_prefix = m_pref.group(1) if m_pref else None

    # tokens por nombre
    name_tokens = [w for w in re.split(r"\s+", qn) if len(w) > 2]

    piscina_mode = (space and _norm(space) == "piscina")

    def score_item(code: str, prod: dict) -> int:
        name = prod.get("name") or ""
        name_n = _norm(name)
        tags_n = _norm(" ".join(map(str, prod.get("tags", []))))
        cats_n = _norm(" ".join(map(str, prod.get("categories", []))))
        full_n = f"{name_n} {tags_n} {cats_n}"

        # filtro duro para piscina
        if piscina_mode and not (("sumergible" in full_n) or ("ip68" in full_n)):
            return 0

        s = 0
        cu = code.upper()

        # 1) coincidencia por código
        if code_token:
            if cu == code_token:
                s += 100
            elif cu.startswith(code_token):
                s += 70
            elif code_token in cu:
                s += 45

        # 2) afinidad por prefijo
        if code_prefix:
            if cu.startswith(code_prefix):
                s += 35
            else:
                s -= 10  # penaliza prefijos distintos (evita colarse TWCRGB, MULTIRGBPL, etc.)

        # 3) coincidencia por nombre/tags/categorías
        for t in name_tokens:
            if t in name_n: s += 14
            if t in tags_n: s += 7
            if t in cats_n: s += 4

        # 4) desempate por espacio (suave)
        if space:
            sp = _norm(space)
            if sp in full_n: s += 8
            if sp == "piscina" and (("sumergible" in full_n) or ("ip68" in full_n)):
                s += 10

        return s

    candidatos: List[tuple[int,str]] = []
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
    must_have: Set[str] | None = None,
    presupuesto: int | None = None,
    exclude_codes: Set[str] | None = None,
):
    qn = _norm(query_text)
    words = [w for w in qn.split() if len(w) > 2]
    must_have = { _norm(x) for x in (must_have or set()) }
    exclude_codes = exclude_codes or set()

    out: List[tuple[int,str]] = []
    for code, p in PRODUCTOS.items():
        if code in exclude_codes: continue
        name = _norm(p.get("name",""))
        tags = _norm(" ".join([str(t) for t in p.get("tags", [])]))
        cats = _norm(" ".join([str(c) for c in p.get("categories", [])]))
        full = f"{name} {tags} {cats}"
        price = p.get("price")

        # must_have: todas deben aparecer en el texto del producto
        ok = True
        for m in must_have:
            if m and m not in full:
                ok = False; break
        if not ok: continue

        # scoring: solo si hay match con consulta O hay must_have
        hit, score = False, 0
        for w in words:
            if w in name: score += 6; hit = True
            if w in tags: score += 4; hit = True
            if w in cats: score += 2; hit = True

        if hit or must_have:
            for b in {"sumergible","ip68","piscina","rgb","12v","panel","downlight","lineal","proyector","poste","aplique","ip65"}:
                if b in full: score += 2

        # presupuesto
        if isinstance(price,(int,float)) and isinstance(presupuesto,int):
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

def sugerencias_aleatorias(space: str | None, n: int = 5, exclude_codes: set[str] | None = None):
    """
    Devuelve hasta n productos aleatorios. Si 'space' viene, prioriza productos cuyo texto contenga esa palabra.
    """
    exclude_codes = exclude_codes or set()

    def fits(prod: dict, space_k: str) -> bool:
        if not space_k: return True
        texto = _norm(
            (prod.get("name") or "")
            + " " + " ".join(map(str, prod.get("tags", [])))
            + " " + " ".join(map(str, prod.get("categories", [])))
        )
        return space_k in texto

    keys = [k for k,p in PRODUCTOS.items() if k not in exclude_codes and fits(p, _norm(space or ""))]
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

# -------------- Parsers de SPECs ----------
def parse_temperatura(s: str) -> str | None:
    t = _norm(s)
    if any(w in t for w in ["calida","cálida","warm","3000","2700","3000k","2700k"]): return "cálida"
    if any(w in t for w in ["neutra","neutral","4000","4000k"]): return "neutra"
    if any(w in t for w in ["fria","fría","cool","6000","6500","6000k","6500k"]): return "fría"
    return None

def parse_vatios(s: str) -> int | None:
    m = re.search(r"(\d{1,3})\s*w", _norm(s))
    return int(m.group(1)) if m else None

def parse_instalacion(s: str) -> str | None:
    t = _norm(s)
    if "incrust" in t or "empotr" in t: return "incrustar"
    if "sobrep" in t or "superficie" in t: return "sobreponer"
    return None

# -------------- LLM ORCHESTRATION ----------
def llm_turn(user_text: str, state: dict) -> dict:
    lite_state = {
        "espacio": state.get("espacio"),
        "modo": state.get("modo"),
        "temperatura": state.get("temperatura"),
        "vatios": state.get("vatios"),
        "instalacion": state.get("instalacion"),
    }
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT_CONDUCTOR},
        {"role": "user", "content": f"Usuario: {user_text}\nEstado:{json.dumps(lite_state, ensure_ascii=False)}"}
    ]
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=200,
            messages=msgs,
        )
        raw = (r.choices[0].message.content or "").strip()
        data = json.loads(raw)
        return {
            "reply": data.get("reply") or "Listo.",
            "action": data.get("action") or "NONE",
            "space": data.get("space"),
            "mode": data.get("mode"),
            "spec_field": data.get("spec_field"),
        }
    except Exception:
        return {"reply": "Listo, ¿en qué espacio sería la instalación (piscina, oficina, etc.)?", "action": "ASK_SPACE",
                "space": None, "mode": None, "spec_field": None}

# -------------- ESTADO / CHAT ----------
class ChatIn(BaseModel):
    message: str
    lead: dict | None = None

def get_state(session_id="default"):
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

@app.get("/session")
def new_session(): return {"session_id": str(uuid.uuid4())}

@app.post("/chat")
def chat(in_: ChatIn):
    session_id = "default"
    st = get_state(session_id)
    user = (in_.message or "").strip()

    # ── PASO 0: si YA hay espacio y AÚN no hay modo, captura intención con heurística
    if st.get("espacio") and not st.get("modo"):
        if said_suggest(user):
            st["modo"] = "sugerir"
            brain = {
                "reply": "Listo, te muestro 5 opciones para ese espacio 👌",
                "action": "SHOW_SUGGESTIONS",
                "space": st["espacio"], "mode": "sugerir", "spec_field": None,
            }
        elif said_specific(user) or code_regex.search(user) or CODE_SOFT_RE.search(user):
            st["modo"] = "especifico"
            # si ya viene código/nombre en el mismo mensaje, busca de una
            brain = {
                "reply": "Busco esa referencia y te muestro lo que encontré 👌",
                "action": "SEARCH_SPECIFIC",
                "space": st["espacio"], "mode": "especifico", "spec_field": None,
            }
        else:
            brain = llm_turn(user, st)
    else:
        # flujo normal con el conductor
        brain = llm_turn(user, st)

    # ── Actualiza estado con lo que devuelva el LLM
    if brain.get("space") and not st.get("espacio"):
        st["espacio"] = _norm(brain["space"])
    if brain.get("mode") and not st.get("modo"):
        st["modo"] = brain["mode"]

    # ── Fallback: detectar espacio directo del texto
    detected_space = False
    if not st.get("espacio"):
        t = _norm(user)
        for k in [
            "piscina","oficina","pasillo","bodega","sala","cocina",
            "baño","banio","terraza","jardin","jardín","fachada",
            "parqueadero","retail","industrial","dormitorio","escalera"
        ]:
            if k in t:
                st["espacio"] = "baño" if k == "banio" else ("jardin" if k == "jardín" else k)
                detected_space = True
                break

    # Si acabamos de detectar espacio y el LLM aún pide espacio → forzar ASK_MODE
    if detected_space and brain.get("action") == "ASK_SPACE":
        brain = {
            "reply": "¿Quieres que te sugiera o tienes una luminaria en específico?",
            "action": "ASK_MODE",
            "space": st["espacio"], "mode": st.get("modo"), "spec_field": None,
        }

    # ── Recolección de specs (si aplica)
    if brain.get("action") == "COLLECT_SPECS":
        t = parse_temperatura(user);  w = parse_vatios(user);  i = parse_instalacion(user)
        if t: st["temperatura"] = t
        if w: st["vatios"] = w
        if i: st["instalacion"] = i

    # ── Ejecutar SOLO acciones pedidas
    productos = []
    action = brain.get("action", "NONE")

    if action == "SHOW_SUGGESTIONS":
        productos = sugerencias_aleatorias(st.get("espacio"), n=5, exclude_codes=st["mostrados"])
        for p in productos: st["mostrados"].add(p["code"])

    elif action == "MORE_SUGGESTIONS":
        productos = sugerencias_aleatorias(st.get("espacio"), n=5, exclude_codes=st["mostrados"])
        for p in productos: st["mostrados"].add(p["code"])

    elif action == "SEARCH_SPECIFIC":
        # 1) usa el buscador específico por código/nombre
        productos = buscar_producto_especifico(
            query_text=user,
            space=st.get("espacio"),
            k=5
        )

        # 2) fallback: si no encontró, usa el buscador general con señales suaves
        if not productos:
            must = must_from_text(st.get("espacio") or "")
            productos = buscar_productos(
                query_text=user,
                k=5,
                must_have=must,
                presupuesto=st.get("presupuesto"),
                exclude_codes=st["mostrados"]
            )

        # 3) marca como mostrados
        for p in productos:
            st["mostrados"].add(p["code"])

        # 4) copy de respuesta con el código correcto (si lo hay)
        code_tok = extraer_codigo(user)
        brain["reply"] = f"Esto es lo que encontré para {code_tok} 👇" if code_tok else "Te muestro las coincidencias que encontré 👇"


    return {"reply": brain["reply"], "productos": productos}

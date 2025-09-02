const $ = (q) => document.querySelector(q);
const streamEl = $('#cbStream');
const inputEl = $('#cbInput');
const sendBtn = $('#cbSend');
const panel = $('#cbPanel');
const overlay = $('#cbOverlay');
const fab = $('#cbFab');
const closeBtn = $('#cbClose');
const typing = $('#cbTyping');

let SESSION_ID = "default";
let useSSE = false; // puedes activar SSE cuando quieras

async function initSession() {
  try {
    const r = await fetch('/session');
    const j = await r.json();
    SESSION_ID = j.session_id || 'default';
  } catch (e) {
    SESSION_ID = 'default';
  }
}
initSession();

fab.addEventListener('click', openPanel);
overlay.addEventListener('click', closePanel);
closeBtn.addEventListener('click', closePanel);
sendBtn.addEventListener('click', sendMessage);
inputEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendMessage(); });

function openPanel() {
  overlay.style.display = 'block';
  panel.style.display = 'flex';
  // saludo inicial suave
  pushBot("👋 hola, ¿qué espacio quieres iluminar?");
  inputEl.focus();
}
function closePanel() {
  overlay.style.display = 'none';
  panel.style.display = 'none';
  streamEl.innerHTML = '';
}

function el(t, c, h) { const n = document.createElement(t); if (c) n.className = c; if (h !== undefined) n.innerHTML = h; return n; }
function pushUser(text) {
  const row = el('div', 'msg user');
  row.appendChild(el('div', 'bubble', escapeHTML(text)));
  streamEl.appendChild(row); streamEl.scrollTop = streamEl.scrollHeight;
}
function pushBot(text) {
  const row = el('div', 'msg');
  row.appendChild(el('div', 'bubble', escapeHTML(text)));
  streamEl.appendChild(row); streamEl.scrollTop = streamEl.scrollHeight;
}
function pushCards(items=[]) {
  if (!items.length) return;
  const wrap = el('div', 'cards');
  items.forEach(p => {
    const card = el('div', 'card');
    const img = el('img'); img.src = p.img_url || '/static/placeholder.png';
    const info = el('div', 'info');
    info.appendChild(el('div', 'name', escapeHTML(p.name || p.code)));
    if (typeof p.price === 'number') info.appendChild(el('div', 'price', `$ ${p.price.toLocaleString('es-CO')}`));
    if (p.tags && p.tags.length) info.appendChild(el('div', 'tags', escapeHTML(p.tags.slice(0,4).join(' · '))));
    const cta = el('div', 'cta');
    const ver = el('a', 'btn', 'Ver producto'); ver.href = p.url || '#'; ver.target = '_blank';
    const copiar = el('a', 'btn', 'Copiar código'); copiar.href = '#'; copiar.onclick = (e) => {
      e.preventDefault(); navigator.clipboard.writeText(p.code || '');
    };
    cta.appendChild(ver); cta.appendChild(copiar);
    info.appendChild(cta);
    card.appendChild(img); card.appendChild(info);
    wrap.appendChild(card);
  });
  const row = el('div', 'msg');
  const bub = el('div', 'bubble'); bub.appendChild(wrap);
  row.appendChild(bub); streamEl.appendChild(row); streamEl.scrollTop = streamEl.scrollHeight;
}

function escapeHTML(s='') {
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;', "'":'&#39;'}[c]));
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;
  pushUser(text);
  inputEl.value = '';

  if (useSSE) {
    typing.style.display = 'flex';
    const es = new EventSource(`/stream?q=${encodeURIComponent(text)}&session_id=${encodeURIComponent(SESSION_ID)}`);
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.typing) return;
        typing.style.display = 'none';
        if (data.reply) pushBot(data.reply);
        if (data.productos?.length) pushCards(data.productos);
        es.close();
      } catch {}
    };
    es.onerror = () => { typing.style.display = 'none'; es.close(); };
    return;
  }

  typing.style.display = 'flex';
  try {
    const r = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type':'application/json' },
      body: JSON.stringify({ message: text, session_id: SESSION_ID })
    });
    const j = await r.json();
    typing.style.display = 'none';
    if (j.reply) pushBot(j.reply);
    if (j.productos?.length) pushCards(j.productos);
  } catch (e) {
    typing.style.display = 'none';
    pushBot("ups, hubo un detalle de conexión. intenta de nuevo");
  }
}
 
// ================== CONFIG ==================
const CONFIG = { brand: 'Ecolite', whatsapp: '573173630363', emailTo: 'ventas@tudominio.com' };

// ================== UI helpers ==================
const $ = (q) => document.querySelector(q);
const stream = $('#cbStream'), input = $('#cbInput'), sendBtn = $('#cbSend');
const panel = $('#cbPanel'), overlay = $('#cbOverlay'), fab = $('#cbFab'), closeBtn = $('#cbClose'), footer = $('.cb-footer');

fab.addEventListener('click', openPanel);
overlay.addEventListener('click', closePanel);
closeBtn.addEventListener('click', closePanel);

function openPanel() {
  overlay.style.display = 'block';
  panel.style.display = 'flex';
  if (state.step === 'form') showForm();
}
function closePanel() {
  overlay.style.display = 'none';
  panel.style.display = 'none';
}

function el(t, c, h) {
  const n = document.createElement(t);
  if (c) n.className = c;
  if (h !== undefined) n.innerHTML = h;
  return n;
}
function pushBot(html) {
  typing(true);
  setTimeout(() => {
    typing(false);
    const row = el('div', 'msg');
    row.appendChild(el('div', 'bubble', html));
    stream.appendChild(row);
    stream.scrollTo({ top: stream.scrollHeight, behavior: 'smooth' });
  }, 1200); // simula que piensa 1.2s
}
function pushMe(html) {
  const row = el('div', 'msg me');
  row.appendChild(el('div', 'bubble', html));
  stream.appendChild(row);
  stream.scrollTo({ top: stream.scrollHeight, behavior: 'smooth' });
}
let typingEl = null;
function typing(on = true) {
  if (on && !typingEl) {
    typingEl = el('div', 'msg');
    typingEl.appendChild(el('div', 'bubble', 'Escribiendo…'));
    stream.appendChild(typingEl);
  }
  if (!on && typingEl) {
    typingEl.remove();
    typingEl = null;
  }
  stream.scrollTo({ top: stream.scrollHeight, behavior: 'smooth' });
}

function pushBotWithProducts(reply, productos = []) {
    if (productos.length) {
    const seen = new Set();
    productos = productos.filter(p => {
      if (seen.has(p.code)) return false;
      seen.add(p.code);
      return true;
    });
  }
  
  let prodsHTML = "";
  if (productos.length) {
    prodsHTML = `
      <div class="prod-inline">
        ${productos.map(p => `
          <div class="prod-item">
            <img src="${p.img_url}" alt="${p.name}">
            <div class="prod-info">
              <div class="prod-name">${p.name}</div>
              <div class="prod-price">${p.price}</div>
              <a class="prod-link" href="${p.url}" target="_blank">Ver producto</a>
            </div>
          </div>
        `).join("")}
      </div>`;
  }


const html = `${prodsHTML}${reply ? `<div class="mt8">${reply}</div>` : ""}`;
pushBot(html);
}

(function injectProductStyles() {
  if (document.getElementById('prod-inline-styles')) return;
  const css = `
  .bubble .prod-inline{display:flex;flex-direction:column;gap:10px;margin-bottom:8px;}
  .bubble .prod-item{display:flex;align-items:center;gap:10px;background:#fafafa;border:1px solid #eee;border-radius:10px;padding:8px;}
  .bubble .prod-item img{width:78px;height:78px;object-fit:contain;background:#f7f7f7;border-radius:8px;}
  .bubble .prod-info{display:flex;flex-direction:column;gap:2px;}
  .bubble .prod-name{font-size:13px;font-weight:600;color:#222;line-height:1.2;}
  .bubble .prod-price{font-size:13px;font-weight:700;color:#0a8a3a;}
  .bubble .prod-link{font-size:12px;color:#0a5bd3;text-decoration:none;}
  .bubble .mt8{margin-top:8px;}
  `;
  const style = document.createElement('style');
  style.id = 'prod-inline-styles';
  style.textContent = css;
  document.head.appendChild(style);
})();



// ================== Estado ==================
let state = { step: 'form', lead: {}, flow: "inicio" };

// ================== Formulario ==================
function showForm() {
  footer.style.display = 'none';
  stream.classList.add('form-mode');
  stream.innerHTML = `
    <div class="form-hero">
      <form id="leadForm" class="form-card">
        <h3>Bienvenido a ${CONFIG.brand}</h3>
        <p style="font-size:13px;color:var(--muted)">Completa tus datos para continuar</p>
        <div class="grid-2">
          <div class="f-field"><label class="f-label">Nombre *</label><input class="f-input" name="nombre" required></div>
          <div class="f-field"><label class="f-label">Apellidos *</label><input class="f-input" name="apellidos" required></div>
          <div class="f-field"><label class="f-label">Email *</label><input class="f-input" type="email" name="correo" required></div>
          <div class="f-field"><label class="f-label">Teléfono *</label><input class="f-input" type="tel" name="telefono" required></div>
          <div class="f-field"><label class="f-label">Ciudad *</label><input class="f-input" name="ciudad" required></div>
        </div>
        <div class="form-actions">
          <button type="submit" class="btn">Continuar</button>
          <button type="button" class="btn secondary" id="f-cancel">Cancelar</button>
        </div>
      </form>
    </div>`;
  const form = $('#leadForm'), cancel = $('#f-cancel');
  form.addEventListener('submit', e => {
    e.preventDefault();
    const fd = new FormData(form);
    state.lead = Object.fromEntries(fd.entries());
    state.step = 'chat';
    state.flow = "inicio";
    stream.classList.remove('form-mode');
    footer.style.display = 'flex';
    stream.innerHTML = '';
    welcome();
  });
  cancel.addEventListener('click', () => closePanel());
}

// ================== Chat Flow ==================
function welcome() {
  state.flow = "inicio";
  pushBot(`👋 Hola <strong>${state.lead.nombre || ''}</strong>, dime qué espacio quieres iluminar (ej: oficina, pasillo, bodega...)`);

}

// ================== WhatsApp helper ==================
function openWA() {
  const L = state.lead;
  let txt = `Hola soy ${L.nombre || ''}`;
  if (L.ciudad) txt += `, resido en ${L.ciudad}`;
  if (L.interes) txt += ` y me interesa ${L.interes}`;
  txt = encodeURIComponent(txt);
  const href = `https://wa.me/${CONFIG.whatsapp}?text=${txt}`;
  window.open(href, '_blank');
  pushBot('Abrí una ventana de WhatsApp ✅');
}

// ================== llamada al backend ==================
async function callLLM(userTxt) {
  typing(true);
  const payload = { message: userTxt, lead: state.lead };
  try {
    const r = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!r.ok) {
      typing(false);
      pushBot("⚠️ Hubo un problema con el servidor.");
      return;
    }
    const data = await r.json();
    typing(false);
    pushBotWithProducts(data.reply || "", data.productos || []);
  } catch (e) {
    typing(false);
    console.error("Error en callLLM:", e);
    pushBot("⚠️ El asistente no respondió, intenta de nuevo.");
  }
}

// ================== Input libre ==================
async function onSend() {
  const txt = input.value.trim();
  if (!txt) return;
  input.value = '';
  pushMe(txt);

  try {
    typing(true);
    const r = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: txt })
    });
    const data = await r.json();
    typing(false);

    // ✅ solo una burbuja con fotos + texto
    pushBotWithProducts(data.reply || "", data.productos || []);

  } catch (e) {
    typing(false);
    pushBot("⚠️ No me pude conectar, intenta de nuevo.");
  }
}



function addProductCard(prod) {
  const card = document.createElement("div");
  card.className = "product-card";
  card.innerHTML = `
    <img src="${prod.img_url}" alt="${prod.name}" style="max-width:120px; border-radius:8px; margin-right:10px;">
    <div>
      <h4 style="margin:0;font-size:14px">${prod.name}</h4>
      <p style="margin:4px 0;color:green"><b>${prod.price}</b></p>
      <a href="${prod.url}" target="_blank">Ver producto</a>
    </div>
  `;
  stream.appendChild(card);
  stream.scrollTo({ top: stream.scrollHeight, behavior: 'smooth' });
}


sendBtn.addEventListener('click', onSend);
input.addEventListener('keydown', e => { if (e.key === 'Enter') onSend(); });


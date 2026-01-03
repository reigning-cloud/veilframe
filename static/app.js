const panels = {
  encode: document.getElementById('encode-panel'),
  decode: document.getElementById('decode-panel'),
};
const savedPanel = localStorage.getItem('activePanel');
const encodeTab = document.querySelector('.mode-btn[data-target="encode-panel"]');
const decodeTab = document.querySelector('.mode-btn[data-target="decode-panel"]');

const modeButtons = document.querySelectorAll('.mode-btn');
const toolStatusEl = document.getElementById('tool-status-list');
const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
const ALLOWED_IMAGE_TYPES = new Set(['image/png', 'image/jpeg']);
const ALLOWED_IMAGE_EXTS = ['.png', '.jpg', '.jpeg'];

function formatMb(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function hasSupportedExtension(name) {
  const lower = (name || '').toLowerCase();
  return ALLOWED_IMAGE_EXTS.some((ext) => lower.endsWith(ext));
}

function isSupportedImage(file) {
  if (!file) return false;
  if (file.type) return ALLOWED_IMAGE_TYPES.has(file.type);
  return hasSupportedExtension(file.name);
}

function validateImageFile(file) {
  if (!file) return 'Please choose an image to upload.';
  if (!isSupportedImage(file)) return 'Unsupported image type. Please use PNG or JPG.';
  if (file.size > MAX_IMAGE_BYTES) {
    return `Image too large (${formatMb(file.size)}). Try under ${formatMb(MAX_IMAGE_BYTES)}.`;
  }
  return null;
}

async function readResponse(res) {
  const text = await res.text();
  if (!text) return { data: null, text: '' };
  try {
    return { data: JSON.parse(text), text };
  } catch (_) {
    return { data: null, text };
  }
}

function responseMessage(res, data, text) {
  const status = `${res.status}${res.statusText ? ` ${res.statusText}` : ''}`;
  const base = `Server response (${status})`;
  if (data && data.error) return `${base}: ${data.error}`;
  if (text) {
    const snippet = text.replace(/\s+/g, ' ').slice(0, 160);
    return `${base}: ${snippet}`;
  }
  return base;
}

function showPanel(targetId, persist = true) {
  if (persist && targetId) {
    localStorage.setItem('activePanel', targetId);
  }
  modeButtons.forEach((b) => b.classList.remove('active'));
  Object.entries(panels).forEach(([key, panel]) => {
    if (!panel) return;
    if (`${key}-panel` === targetId) {
      panel.classList.add('active');
      const tab = document.querySelector(`.mode-btn[data-target="${targetId}"]`);
      if (tab) tab.classList.add('active');
      if (targetId === 'decode-panel') {
        loadToolStatus();
      }
    } else {
      panel.classList.remove('active');
    }
  });
}
modeButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.target;
    showPanel(target, true);
  });
});

const modeRadios = document.querySelectorAll('input[name="encodeMode"]');
const simplePlaneField = document.getElementById('simple-plane-field');
const advancedGrid = document.getElementById('advanced-grid');
const jpegFormatRadio = document.querySelector('input[name="outputFormat"][value="jpeg"]');
const pngFormatRadio = document.querySelector('input[name="outputFormat"][value="png"]');

function syncOutputFormatForMode(advanced) {
  if (!jpegFormatRadio) return;
  jpegFormatRadio.disabled = advanced;
  if (advanced && pngFormatRadio) {
    pngFormatRadio.checked = true;
  }
}
const setModeUI = () => {
  const val = document.querySelector('input[name="encodeMode"]:checked').value;
  const advanced = val === 'advanced';
  if (simplePlaneField) simplePlaneField.style.display = advanced ? 'none' : 'flex';
  if (advancedGrid) advancedGrid.style.display = advanced ? 'grid' : 'none';
  syncOutputFormatForMode(advanced);
};
modeRadios.forEach((radio) => {
  radio.addEventListener('change', () => {
    const val = document.querySelector('input[name="encodeMode"]:checked').value;
    const advanced = val === 'advanced';
    localStorage.setItem('encodeMode', val);
    setModeUI();
  });
});
const savedMode = localStorage.getItem('encodeMode');
if (savedMode === 'advanced') {
  document.querySelector('input[name="encodeMode"][value="advanced"]').checked = true;
}

const carrierInput = document.getElementById('carrier-image');
const carrierFilename = document.getElementById('carrier-filename');

setModeUI();
const initialPanel = savedPanel === 'decode-panel' || savedPanel === 'encode-panel' ? savedPanel : 'encode-panel';
showPanel(initialPanel, false);

if (carrierInput && carrierFilename) {
  carrierInput.addEventListener('change', () => {
    const file = carrierInput.files && carrierInput.files[0] ? carrierInput.files[0] : null;
    const name = file ? `${file.name} (${formatMb(file.size)})` : 'no photo chosen';
    carrierFilename.textContent = name;
  });
}

const analyzeInput = document.getElementById('analyze-image');
const analyzeFilename = document.getElementById('analyze-filename');
if (analyzeInput && analyzeFilename) {
  analyzeInput.addEventListener('change', () => {
    const file = analyzeInput.files && analyzeInput.files[0] ? analyzeInput.files[0] : null;
    const name = file ? `${file.name} (${formatMb(file.size)})` : 'no photo chosen';
    analyzeFilename.textContent = name;
  });
}

function toggleChannelBodies() {
  document.querySelectorAll('#advanced-grid .channel-card').forEach((card) => {
    const enabledToggle = card.querySelector('.ch-enabled');
    if (!enabledToggle) return;
    const enabled = enabledToggle.checked;
    card.classList.toggle('channel-collapsed', !enabled);
  });
}

// Channel controls persistence
document.querySelectorAll('#advanced-grid .channel-card').forEach((card) => {
  const enabled = card.querySelector('.ch-enabled');
  const textArea = card.querySelector('.ch-text');
  const fileInput = card.querySelector('.ch-file');
  if (enabled) {
    enabled.addEventListener('change', () => {
      toggleChannelBodies();
      saveChannelState();
    });
  }
  if (textArea) textArea.addEventListener('input', saveChannelState);
  if (fileInput) {
    fileInput.addEventListener('change', () => {
      const nameEl = card.querySelector('.ch-file-name');
      const name = fileInput.files && fileInput.files[0] ? fileInput.files[0].name : 'no file';
      if (nameEl) nameEl.textContent = name;
      saveChannelState();
    });
    const nameEl = card.querySelector('.ch-file-name');
    const name = fileInput.files && fileInput.files[0] ? fileInput.files[0].name : 'no file';
    if (nameEl) nameEl.textContent = name;
  }
});

function saveChannelState() {
  const state = {};
  document.querySelectorAll('#advanced-grid .channel-card').forEach((card) => {
    const ch = card.dataset.channel;
    const enabledToggle = card.querySelector('.ch-enabled');
    const textField = card.querySelector('.ch-text');
    if (!enabledToggle || !textField) return;
    const enabled = enabledToggle.checked;
    const text = textField.value;
    state[ch] = { enabled, text };
  });
  localStorage.setItem('channelsState', JSON.stringify(state));
}

function loadChannelState() {
  const saved = localStorage.getItem('channelsState');
  if (!saved) return;
  try {
    const state = JSON.parse(saved);
    document.querySelectorAll('#advanced-grid .channel-card').forEach((card) => {
      const ch = card.dataset.channel;
      const cfg = state[ch];
      if (!cfg) return;
      const enabledToggle = card.querySelector('.ch-enabled');
      const textField = card.querySelector('.ch-text');
      if (enabledToggle) enabledToggle.checked = !!cfg.enabled;
      if (textField) textField.value = cfg.text || '';
    });
    toggleChannelBodies();
  } catch (_) {
    /* ignore */
  }
}
loadChannelState();
toggleChannelBodies();

const encodeForm = document.getElementById('encode-form');
const encodeOutput = document.getElementById('encode-output');
encodeForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const carrierFile = carrierInput && carrierInput.files ? carrierInput.files[0] : null;
  const carrierError = validateImageFile(carrierFile);
  if (carrierError) {
    encodeOutput.innerHTML = `<div class="status-line error">${carrierError}</div>`;
    return;
  }
  encodeOutput.innerHTML = '<div class="status-line">Encoding...</div>';

  const fd = new FormData(encodeForm);
  fd.append('mode', 'text'); // simple defaults to text
  const encodeMode = document.querySelector('input[name="encodeMode"]:checked').value;

  try {
    if (encodeMode === 'advanced') {
      const channels = {};
      document.querySelectorAll('#advanced-grid .channel-card').forEach((card) => {
        const ch = card.dataset.channel;
        const enabledToggle = card.querySelector('.ch-enabled');
        const textField = card.querySelector('.ch-text');
        const fileInput = card.querySelector('.ch-file');
        if (!enabledToggle || !textField) return;
        const enabled = enabledToggle.checked;
        const text = textField.value;
        const fileObj = fileInput && fileInput.files.length ? fileInput.files[0] : null;
        const type = fileObj ? 'file' : 'text';
        channels[ch] = { enabled, type, text };
        if (fileObj) {
          fd.append(`file_${ch}`, fileObj);
        }
      });
      fd.append('channels', JSON.stringify(channels));
    } else {
      fd.append('text', document.querySelector('textarea[name="text"]').value || '');
    }

    const res = await fetch('/api/encode', { method: 'POST', body: fd });
    const { data, text } = await readResponse(res);
    if (!res.ok) {
      throw new Error(responseMessage(res, data, text));
    }
    if (!data) {
      throw new Error(responseMessage(res, data, text));
    }
    if (data.error) {
      throw new Error(data.error);
    }
    renderEncodeResult(data);
  } catch (err) {
    encodeOutput.innerHTML = `<div class="status-line error">${err.message}</div>`;
  }
});

function renderEncodeResult(data) {
  const html = `
    <div class="result-grid">
      <div class="result-card">
        <h3>Encoded image</h3>
        <img src="${data.data_url}" alt="encoded" style="width:100%;border-radius:10px;border:1px solid rgba(255,255,255,0.1);background:rgba(255,255,255,0.02);">
        <div class="downloads" style="margin-top:10px;">
          <a href="${data.data_url}" download="${data.filename}">Download ${data.filename}</a>
        </div>
      </div>
    </div>
  `;
  encodeOutput.innerHTML = html;
}

const decodeForm = document.getElementById('decode-form');
const decodeOutput = document.getElementById('decode-output');
const outguessToggle = document.querySelector('input[name="deep"]');
const outguessPasswordField = document.getElementById('outguess-password-field');

function syncOutguessPassword() {
  if (!outguessPasswordField) return;
  const enabled = outguessToggle && outguessToggle.checked;
  outguessPasswordField.classList.toggle('visible', !!enabled);
}
if (outguessToggle) {
  outguessToggle.addEventListener('change', syncOutguessPassword);
  syncOutguessPassword();
}
decodeForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const analyzeFile = analyzeInput && analyzeInput.files ? analyzeInput.files[0] : null;
  const analyzeError = validateImageFile(analyzeFile);
  if (analyzeError) {
    decodeOutput.innerHTML = `<div class="status-line error">${analyzeError}</div>`;
    return;
  }
  decodeOutput.innerHTML = '<div class="status-line">Running analyzers...</div>';
  const fd = new FormData(decodeForm);
  showPanel('decode-panel');

  try {
    const res = await fetch('/api/decode', { method: 'POST', body: fd });
    const { data, text } = await readResponse(res);
    if (!res.ok) {
      throw new Error(responseMessage(res, data, text));
    }
    if (!data) {
      throw new Error(responseMessage(res, data, text));
    }
    if (data.error) {
      throw new Error(data.error);
    }
    renderDecodeResult(data);
  } catch (err) {
    decodeOutput.innerHTML = `<div class="status-line error">${err.message}</div>`;
  }
});

function renderDecodeResult(data) {
  const { results = {}, artifacts = { images: [], archives: [] } } = data;
  const priority = ["simple_rgb", "red_plane", "green_plane", "blue_plane", "alpha_plane"];
  const stringsKey = "strings";
  const restOrder = ["simple_zlib", "binwalk", "foremost", "exiftool", "steghide", "outguess", "zsteg", "decomposer"];

  const cardsPriority = priority
    .filter((k) => results[k])
    .map((k) => renderTool(k, results[k]))
    .join('');

  const cardsRest = restOrder
    .filter((k) => results[k] && !priority.includes(k) && k !== stringsKey)
    .map((k) => renderTool(k, results[k]))
    .join('');

  const cardsStrings = results[stringsKey] ? renderTool(stringsKey, results[stringsKey], true) : '';

  const gallery = artifacts.images
    .map((img) => `<div><img src="${img.data_url}" alt="${img.name}"><div class="status-line">${img.name}</div></div>`)
    .join('');

  const downloads = artifacts.archives
    .map((file) => `<a href="${file.data_url}" download="${file.name}">${file.name}</a>`)
    .join('');

  decodeOutput.innerHTML = `
    <div class="result-grid priority-grid">${cardsPriority || ''}</div>
    ${gallery ? `<h3 class="gallery-title">Bit-plane gallery</h3><div class="gallery">${gallery}</div>` : ''}
    <div class="result-grid">${cardsRest || ''}</div>
    ${downloads ? `<div class="downloads" style="margin-top:12px;">${downloads}</div>` : ''}
    ${cardsStrings ? `<div class="result-grid strings-block">${cardsStrings}</div>` : ''}
  `;
}

function renderTool(tool, payload, wide = false) {
  if (!payload || typeof payload !== 'object') {
    return '';
  }
  const status = payload.status || 'unknown';
  const tagClass = status === 'ok' ? 'ok' : status === 'error' ? 'error' : '';
  const hasOutput = Object.prototype.hasOwnProperty.call(payload, 'output');
  const content = formatPayload(
    hasOutput ? payload.output : (payload.error || payload.reason || payload)
  );
  const style = wide ? 'style="grid-column: 1 / -1;"' : '';

  return `
    <div class="result-card" ${style}>
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
        <h3>${tool}</h3>
        <span class="tag ${tagClass}">${status}</span>
      </div>
      ${content}
    </div>
  `;
}

function formatPayload(val) {
  if (Array.isArray(val)) {
    return `<pre>${val.join('\n')}</pre>`;
  }
  if (val && typeof val === 'object') {
    return `<pre>${JSON.stringify(val, null, 2)}</pre>`;
  }
  return `<pre>${val}</pre>`;
}

// Tooling status
async function loadToolStatus() {
  if (!toolStatusEl) return;
  toolStatusEl.innerHTML = '<div class="status-line">loading tools...</div>';
  try {
    const res = await fetch('/api/tools');
    const data = await res.json();
    const tools = data.tools || {};
    const entries = Object.entries(tools);
    if (!entries.length) {
      toolStatusEl.innerHTML = '<div class="status-line">no tools detected.</div>';
      return;
    }
    const html = entries
      .map(([name, info]) => {
        const ok = info && info.available;
        const icon = ok ? '✅' : '❌';
        const cls = ok ? 'ok' : 'missing';
        const path = info && info.path ? `<span class="tool-path">${info.path}</span>` : '';
        return `<div class="tool-pill"><div class="tool-top"><span class="tool-icon ${cls}">${icon}</span><span class="tool-name">${name}</span></div>${path}</div>`;
      })
      .join('');
    toolStatusEl.innerHTML = html || '<div class="status-line">No tools detected.</div>';
  } catch (err) {
    toolStatusEl.innerHTML = `<div class="status-line">Tool status unavailable</div>`;
  }
}

loadToolStatus();

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

const STYLE_MAP = {
  A: '𝐀',
  B: '𝖡',
  C: '𝖢',
  D: '𝖣',
  E: '𝐄',
  F: '𝖥',
  G: '𝖦',
  H: '𝖧',
  I: '𝐈',
  J: '𝖩',
  K: '𝖪',
  L: '𝖫',
  M: '𝖬',
  N: '𝖭',
  O: '𝐎',
  P: '𝖯',
  Q: '𝖰',
  R: '𝖱',
  S: '𝖲',
  T: '𝖳',
  U: '𝐔',
  V: '𝖵',
  W: '𝖶',
  X: '𝖷',
  Y: '𝖸',
  Z: '𝖹',
  a: '𝐚',
  b: '𝖻',
  c: '𝖼',
  d: '𝖽',
  e: '𝐞',
  f: '𝖿',
  g: '𝗀',
  h: '𝗁',
  i: '𝐢',
  j: '𝗃',
  k: '𝗄',
  l: '𝗅',
  m: '𝗆',
  n: '𝗇',
  o: '𝐨',
  p: '𝗉',
  q: '𝗊',
  r: '𝗋',
  s: '𝗌',
  t: '𝗍',
  u: '𝐮',
  v: '𝗏',
  w: '𝗐',
  x: '𝗑',
  y: '𝗒',
  z: '𝗓',
};

function stylizeText(text) {
  return String(text || '').replace(/[A-Za-z]/g, (ch) => STYLE_MAP[ch] || ch);
}

function formatMb(bytes) {
  return stylizeText(`${(bytes / (1024 * 1024)).toFixed(1)} MB`);
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
  if (!file) return stylizeText('Please choose an image to upload.');
  if (!isSupportedImage(file)) return stylizeText('Unsupported image type. Please use PNG or JPG.');
  if (file.size > MAX_IMAGE_BYTES) {
    return stylizeText(
      `Image too large (${formatMb(file.size)}). Try under ${formatMb(MAX_IMAGE_BYTES)}.`
    );
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
  const base = stylizeText(`Server response (${status})`);
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

const encodeMethodSelect = document.getElementById('encode-method');
const simplePlaneField = document.getElementById('simple-plane-field');
const advancedGrid = document.getElementById('advanced-grid');
const jpegFormatRadio = document.querySelector('input[name="outputFormat"][value="jpeg"]');
const pngFormatRadio = document.querySelector('input[name="outputFormat"][value="png"]');
const methodPanels = document.querySelectorAll('[data-encode-method]');
const methodOptionsField = document.getElementById('encode-method-options');
const payloadModeRadios = document.querySelectorAll('input[name="payloadMode"]');
const payloadTextPanel = document.getElementById('payload-text-panel');
const payloadFilePanel = document.getElementById('payload-file-panel');
const payloadFileInput = document.getElementById('payload-file-input');
const payloadFileName = document.getElementById('payload-file-name');
const payloadTextArea = document.querySelector('#payload-text-panel textarea[name="text"]');

function syncOutputFormatForMethod(method) {
  if (!jpegFormatRadio || !pngFormatRadio) return;
  let force = '';
  if (['advanced_lsb', 'palette', 'png_chunks'].includes(method)) {
    force = 'png';
  }
  if (['f5', 'dct'].includes(method)) {
    force = 'jpeg';
  }
  if (!force) {
    jpegFormatRadio.disabled = false;
    pngFormatRadio.disabled = false;
    return;
  }
  if (force === 'png') {
    pngFormatRadio.checked = true;
    pngFormatRadio.disabled = false;
    jpegFormatRadio.disabled = true;
  } else {
    jpegFormatRadio.checked = true;
    jpegFormatRadio.disabled = false;
    pngFormatRadio.disabled = true;
  }
}

function getPayloadMode() {
  const selected = document.querySelector('input[name="payloadMode"]:checked');
  return selected ? selected.value : 'text';
}

function setPayloadModeUI() {
  const mode = getPayloadMode();
  const useFile = mode === 'file';
  if (payloadTextPanel) payloadTextPanel.classList.toggle('hidden', useFile);
  if (payloadFilePanel) payloadFilePanel.classList.toggle('hidden', !useFile);
  if (payloadTextArea) payloadTextArea.disabled = useFile;
  if (payloadFileInput) payloadFileInput.disabled = !useFile;
}

function setEncodeMethodUI() {
  const method = encodeMethodSelect ? encodeMethodSelect.value : 'simple_lsb';
  const advanced = method === 'advanced_lsb';
  if (simplePlaneField) simplePlaneField.style.display = advanced ? 'none' : 'flex';
  if (advancedGrid) advancedGrid.style.display = advanced ? 'grid' : 'none';
  let hasActivePanel = false;
  methodPanels.forEach((panel) => {
    const active = panel.dataset.encodeMethod === method;
    panel.classList.toggle('hidden', !active);
    panel.querySelectorAll('input, select, textarea').forEach((el) => {
      el.disabled = !active;
    });
    if (active) hasActivePanel = true;
  });
  if (methodOptionsField) {
    methodOptionsField.style.display = hasActivePanel ? 'flex' : 'none';
  }
  syncOutputFormatForMethod(method);
  setPayloadModeUI();
  localStorage.setItem('encodeMethod', method);
}

const savedMethod = localStorage.getItem('encodeMethod');
const legacyMode = localStorage.getItem('encodeMode');
if (encodeMethodSelect) {
  if (savedMethod) {
    encodeMethodSelect.value = savedMethod;
  } else if (legacyMode === 'advanced') {
    encodeMethodSelect.value = 'advanced_lsb';
  }
}
if (encodeMethodSelect) {
  encodeMethodSelect.addEventListener('change', setEncodeMethodUI);
}
payloadModeRadios.forEach((radio) => {
  radio.addEventListener('change', setPayloadModeUI);
});

const carrierInput = document.getElementById('carrier-image');
const carrierFilename = document.getElementById('carrier-filename');
setEncodeMethodUI();
const initialPanel = savedPanel === 'decode-panel' || savedPanel === 'encode-panel' ? savedPanel : 'encode-panel';
showPanel(initialPanel, false);

const analyzeInput = document.getElementById('analyze-image');
const analyzeFilename = document.getElementById('analyze-filename');

function bindFileLabel(inputEl, labelEl, emptyLabel) {
  if (!inputEl || !labelEl) return;
  const update = () => {
    const file = inputEl.files && inputEl.files[0] ? inputEl.files[0] : null;
    const name = file ? `${file.name} (${formatMb(file.size)})` : stylizeText(emptyLabel);
    labelEl.textContent = name;
  };
  inputEl.addEventListener('change', update);
  update();
}

bindFileLabel(carrierInput, carrierFilename, 'no photo chosen');
bindFileLabel(analyzeInput, analyzeFilename, 'no photo chosen');
bindFileLabel(payloadFileInput, payloadFileName, 'no file');

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
      const name = fileInput.files && fileInput.files[0] ? fileInput.files[0].name : stylizeText('no file');
      if (nameEl) nameEl.textContent = name;
      saveChannelState();
    });
    const nameEl = card.querySelector('.ch-file-name');
    const name = fileInput.files && fileInput.files[0] ? fileInput.files[0].name : stylizeText('no file');
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
  encodeOutput.innerHTML = `<div class="status-line">${stylizeText('Encoding...')}</div>`;

  const encodeMethod = encodeMethodSelect ? encodeMethodSelect.value : 'simple_lsb';
  const payloadMode = getPayloadMode();
  const payloadFile = payloadFileInput && payloadFileInput.files ? payloadFileInput.files[0] : null;
  const payloadText = payloadTextArea ? payloadTextArea.value.trim() : '';
  if (encodeMethod !== 'advanced_lsb') {
    if (payloadMode === 'file' && !payloadFile) {
      encodeOutput.innerHTML = `<div class="status-line error">${stylizeText('Choose a payload file first.')}</div>`;
      return;
    }
    if (payloadMode === 'text' && !payloadText) {
      encodeOutput.innerHTML = `<div class="status-line error">${stylizeText('Enter a payload message first.')}</div>`;
      return;
    }
  }

  const fd = new FormData(encodeForm);
  fd.append('encodeMethod', encodeMethod);

  try {
    if (encodeMethod === 'advanced_lsb') {
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
    } else if (encodeMethod === 'simple_lsb') {
      fd.append('mode', payloadMode === 'file' ? 'zlib' : 'text');
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
const deepToggle = document.querySelector('input[name="deep"]');
const outguessPasswordField = document.getElementById('outguess-password-field');
const spreadToggle = document.querySelector('input[name="spreadSpectrum"]');
const unicodeToggle = document.querySelector('input[name="unicodeSweep"]');
const unicodeOptions = document.getElementById('unicode-options');

function syncOutguessPassword() {
  if (!outguessPasswordField) return;
  const enabled = deepToggle && deepToggle.checked;
  const needsPassword = spreadToggle && spreadToggle.checked;
  outguessPasswordField.classList.toggle('visible', !!(enabled || needsPassword));
}
if (deepToggle) {
  deepToggle.addEventListener('change', syncOutguessPassword);
  syncOutguessPassword();
}
if (spreadToggle) {
  spreadToggle.addEventListener('change', syncOutguessPassword);
  syncOutguessPassword();
}

function syncUnicodeOptions() {
  if (!unicodeOptions) return;
  const enabled = unicodeToggle && unicodeToggle.checked;
  unicodeOptions.classList.toggle('visible', !!enabled);
  unicodeOptions.querySelectorAll('input, select').forEach((el) => {
    el.disabled = !enabled;
  });
}
if (unicodeToggle) {
  unicodeToggle.addEventListener('change', syncUnicodeOptions);
  syncUnicodeOptions();
}
decodeForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const analyzeFile = analyzeInput && analyzeInput.files ? analyzeInput.files[0] : null;
  const analyzeError = validateImageFile(analyzeFile);
  if (analyzeError) {
    decodeOutput.innerHTML = `<div class="status-line error">${analyzeError}</div>`;
    return;
  }
  decodeOutput.innerHTML = `<div class="status-line">${stylizeText('Running analyzers...')}</div>`;
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
  const planeKeys = ["simple_rgb", "red_plane", "green_plane", "blue_plane", "alpha_plane"];
  const stringsKey = "strings";
  const decodeOptionKeys = [
    "auto_detect",
    "lsb",
    "pvd",
    "dct",
    "f5",
    "spread_spectrum",
    "palette",
    "chroma",
    "png_chunks",
  ];
  const unicodeDecodeKey = "invisible_unicode_decode";
  const autoDetectKey = "auto_detect";
  const restOrder = [
    "advanced_lsb",
    "simple_lsb",
    "simple_zlib",
    "stegg",
    "zero_width",
    "invisible_unicode",
    "randomizer_decode",
    "xor_flag_sweep",
    "binwalk",
    "foremost",
    "exiftool",
    "steghide",
    "outguess",
    "zsteg",
    "decomposer",
    "plane_carver",
    "identify",
    "convert",
    "jpeginfo",
    "jpegtran",
    "cjpeg",
    "djpeg",
    "jpegsnoop",
    "jhead",
    "exiv2",
    "exifprobe",
    "pngcheck",
    "optipng",
    "pngcrush",
    "pngtools",
    "stegdetect",
    "jsteg",
    "stegbreak",
    "stegseek",
    "stegcracker",
    "fcrackzip",
    "bulk_extractor",
    "scalpel",
    "testdisk",
    "photorec",
    "stegoveritas",
    "zbarimg",
    "qrencode",
    "tesseract",
    "ffprobe",
    "ffmpeg",
    "mediainfo",
    "sox",
    "pdfinfo",
    "pdftotext",
    "pdfimages",
    "qpdf",
    "radare2",
    "rizin",
    "hexyl",
    "bvi",
    "xxd",
    "rg",
    "tshark",
    "wireshark",
    "sleuthkit",
    "volatility",
    "stegsolve",
    "openstego",
  ];

  const cardsPlanes = planeKeys
    .filter((k) => results[k])
    .map((k) => renderTool(k, results[k]))
    .join('');

  const unicodeDecodeCard = results[unicodeDecodeKey]
    ? renderTool(unicodeDecodeKey, results[unicodeDecodeKey])
    : '';

  const autoDetectCard = results[autoDetectKey]
    ? renderTool(autoDetectKey, results[autoDetectKey])
    : '';

  const candidates = (results[autoDetectKey]?.details?.candidates || []).slice();
  candidates.sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
  const rankedKeys = [];
  const rankedSet = new Set();
  candidates.forEach((candidate) => {
    const key = candidate.option_id;
    if (!key || rankedSet.has(key)) return;
    if (!results[key]) return;
    rankedSet.add(key);
    rankedKeys.push(key);
  });

  const cardsTop = rankedKeys.map((k) => renderTool(k, results[k])).join('');

  const cardsOtherDecode = decodeOptionKeys
    .filter((k) => k !== autoDetectKey && results[k] && !rankedSet.has(k))
    .map((k) => renderTool(k, results[k]))
    .join('');

  const cardsRest = restOrder
    .filter(
      (k) =>
        results[k] &&
        !planeKeys.includes(k) &&
        !decodeOptionKeys.includes(k) &&
        k !== stringsKey &&
        k !== unicodeDecodeKey
    )
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
    <div class="result-grid priority-grid">${cardsPlanes || ''}</div>
    ${unicodeDecodeCard ? `<div class="result-grid">${unicodeDecodeCard}</div>` : ''}
    ${autoDetectCard ? `<div class="result-grid">${autoDetectCard}</div>` : ''}
    ${cardsTop ? `<div class="result-grid">${cardsTop}</div>` : ''}
    ${cardsOtherDecode ? `<div class="result-grid">${cardsOtherDecode}</div>` : ''}
    ${gallery ? `<h3 class="gallery-title">${stylizeText('Bit-plane gallery')}</h3><div class="gallery">${gallery}</div>` : ''}
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
  const tagClass =
    status === 'ok' ? 'ok' : status === 'error' ? 'error' : status === 'no_signal' || status === 'skipped' ? 'warn' : '';
  const displayTool = payload.label ? payload.label : stylizeText(tool);
  const displayStatus = stylizeText(status);
  const modeBadge = payload.mode ? `<span class="tag mode ${payload.mode}">${stylizeText(payload.mode)}</span>` : '';
  const isSchema = Object.prototype.hasOwnProperty.call(payload, 'summary');
  const hasOutput = Object.prototype.hasOwnProperty.call(payload, 'output');
  const content = isSchema
    ? formatSchemaPayload(payload)
    : formatPayload(hasOutput ? payload.output : (payload.error || payload.reason || payload));
  const style = wide ? 'style="grid-column: 1 / -1;"' : '';

  return `
    <div class="result-card" ${style}>
      <div class="result-card-head">
        <h3>${displayTool}</h3>
        <div class="tag-row">
          ${modeBadge}
          <span class="tag ${tagClass}">${displayStatus}</span>
        </div>
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

function formatSchemaPayload(payload) {
  const lines = [];
  if (payload.summary) {
    lines.push(`<div class="status-line">${payload.summary}</div>`);
  }
  if (typeof payload.confidence === 'number') {
    lines.push(`<div class="status-line">confidence: ${payload.confidence}</div>`);
  }
  if (payload.details && Object.keys(payload.details).length) {
    lines.push(`<pre>${JSON.stringify(payload.details, null, 2)}</pre>`);
  }
  if (payload.artifacts && payload.artifacts.length) {
    lines.push(`<pre>${JSON.stringify(payload.artifacts, null, 2)}</pre>`);
  }
  if (!lines.length) {
    lines.push('<pre>no details</pre>');
  }
  return lines.join('');
}

// Tooling status
async function loadToolStatus() {
  if (!toolStatusEl) return;
  toolStatusEl.innerHTML = `<div class="status-line">${stylizeText('loading tools...')}</div>`;
  try {
    const res = await fetch('/api/tools');
    const data = await res.json();
    const tools = data.tools || {};
    const entries = Object.entries(tools);
    if (!entries.length) {
      toolStatusEl.innerHTML = `<div class="status-line">${stylizeText('no tools detected.')}</div>`;
      return;
    }
    const html = entries
      .map(([name, info]) => {
        const ok = info && info.available;
        const icon = ok ? '✅' : '❌';
        const cls = ok ? 'ok' : 'missing';
        const displayName = stylizeText(name);
        const mode = info && info.mode ? info.mode : 'auto';
        const modeBadge = `<span class="tool-mode ${mode}">${stylizeText(mode)}</span>`;
        const path = info && info.path ? `<span class="tool-path">${info.path}</span>` : '';
        return `<div class="tool-pill"><div class="tool-top"><span class="tool-icon ${cls}">${icon}</span><span class="tool-name">${displayName}</span>${modeBadge}</div>${path}</div>`;
      })
      .join('');
    toolStatusEl.innerHTML = html || `<div class="status-line">${stylizeText('No tools detected.')}</div>`;
  } catch (err) {
    toolStatusEl.innerHTML = `<div class="status-line">Tool status unavailable</div>`;
  }
}

loadToolStatus();

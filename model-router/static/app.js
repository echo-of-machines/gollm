/* GoLLM GUI — vanilla JS, no dependencies */
'use strict';

// ── SVG icons (inline, no external deps) ──────────────────────────────────────
const ICO = {
  load:    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>',
  stop:    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/></svg>',
  trash:   '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>',
  play:    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>',
  restart: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>',
  cancel:  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',
};

// ── Tab routing with sliding indicator ─────────────────────────────────────────
const tabIndicator = document.getElementById('tab-indicator');

function moveIndicator(btn) {
  if (!tabIndicator || !btn) return;
  tabIndicator.style.left  = btn.offsetLeft + 'px';
  tabIndicator.style.width = btn.offsetWidth + 'px';
}

// Set initial indicator position
requestAnimationFrame(() => {
  const activeBtn = document.querySelector('.tab-btn.active');
  if (activeBtn && tabIndicator) {
    tabIndicator.style.transition = 'none';
    moveIndicator(activeBtn);
    requestAnimationFrame(() => {
      tabIndicator.style.transition = '';
    });
  }
});

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
    moveIndicator(btn);
    if (btn.dataset.tab === 'jobs')     refreshJobs();
    if (btn.dataset.tab === 'services') refreshServices();
    if (btn.dataset.tab === 'status')   refreshConfig();
  });
});

// ── API helpers ───────────────────────────────────────────────────────────────
async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const err = new Error(data.error?.message || data.error || `HTTP ${r.status}`);
    err.status = r.status;
    err.body = data;
    throw err;
  }
  return data;
}
const GET    = p       => api('GET',    p);
const POST   = (p, b)  => api('POST',  p, b);
const PUT    = (p, b)  => api('PUT',   p, b);
const DELETE = (p, b)  => api('DELETE', p, b);

// ── Toast ─────────────────────────────────────────────────────────────────────
let toastTimer;
function toast(msg, type = 'ok') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = `toast show ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 3500);
}

// ── Status chip helper ────────────────────────────────────────────────────────
function chip(label, cls) {
  return `<span class="chip chip-${cls}">${label}</span>`;
}
function containerChip(status, healthy, downloading) {
  if (status === 'running') {
    if (healthy)              return chip('ready',       'ready');
    if (downloading)          return chip('downloading', 'downloading');
    return                           chip('loading',     'loading');
  }
  if (status === 'exited')    return chip('stopped',     'exited');
  if (status === 'not_found') return chip('not installed','not_found');
  return chip(status, 'pending');
}
function jobChip(status) {
  const map = { done: 'done', failed: 'failed', running: 'running-job',
                pending: 'pending', cancelled: 'cancelled' };
  return chip(status, map[status] || 'pending');
}

function actionBtn(label, icon, onclick, cls = 'btn-ghost btn-sm') {
  return `<button class="${cls}" onclick="${onclick}">${ICO[icon] || ''} ${label}</button>`;
}

// ── System / header ───────────────────────────────────────────────────────────
async function refreshSystem() {
  try {
    const s = await GET('/router/system');
    _lastSystem = s;
    document.getElementById('conn-dot').className = 'conn-dot ok';

    const badge = document.getElementById('active-badge');
    if (s.swap_pending) {
      badge.className = 'badge badge-swapping';
      badge.innerHTML = 'loading';
    } else if (s.active_set) {
      badge.className = 'badge badge-set';
      badge.innerHTML = s.active_set;
    } else if (s.active_model) {
      badge.className = 'badge badge-active';
      badge.innerHTML = s.active_model;
    } else {
      badge.className = 'badge badge-inactive';
      badge.innerHTML = 'idle';
    }

    const pct = s.ram_total_gb > 0 ? (s.ram_used_gb / s.ram_total_gb) * 100 : 0;
    const fill = document.getElementById('ram-bar-fill');
    fill.style.width = pct.toFixed(1) + '%';
    fill.className = 'ram-bar-fill' + (pct > 90 ? ' crit' : pct > 75 ? ' warn' : '');
    document.getElementById('ram-bar-label').textContent =
      `RAM ${s.ram_used_gb.toFixed(0)} / ${s.ram_total_gb.toFixed(0)} GB`;
  } catch {
    document.getElementById('conn-dot').className = 'conn-dot err';
  }
}

// ── Models tab ────────────────────────────────────────────────────────────────
let _swapping = false;
let _swappingKey = null;  // which model is being loaded
function isSwapping() { return _swapping || (_lastSystem && _lastSystem.swap_pending); }

async function refreshModels() {
  const el = document.getElementById('models-list');
  if (el && !el.children.length) el.innerHTML = '<div class="loading">Fetching data</div>';
  try {
    const { models } = await GET('/router/models');
    const el = document.getElementById('models-list');
    if (!models.length) { el.innerHTML = '<p class="empty">No models configured yet.</p>'; return; }
    const swapping = isSwapping();
    el.innerHTML = models.map(m => {
      const isActive = m.active || m.in_active_set;
      const maxAliases = 4;
      const shownAliases = (m.aliases || []).filter(a => a !== m.key).slice(0, maxAliases);
      const extraCount = Math.max(0, (m.aliases || []).length - 1 - maxAliases);
      const isLoadingThis = swapping && _swappingKey === m.key;
      const loadDisabled = swapping ? 'disabled' : '';
      const loadLabel = isLoadingThis ? 'Loading...' : 'Load';
      return `
      <div class="card ${isActive ? 'active-card' : ''}">
        <div class="card-header">
          <div>
            <div class="card-title">${esc(m.key)}</div>
            <div class="card-subtitle">${esc(m.model_path)}</div>
          </div>
          <div class="card-chips">
            ${isLoadingThis && m.container_status === 'running' && !m.active
              ? chip('starting', 'running-job')
              : containerChip(m.container_status, m.healthy, m.downloading)}
            ${m.active ? chip('active', 'active') : m.in_active_set ? chip('in set', 'active') : ''}
          </div>
        </div>
        <div class="card-meta">
          <span class="meta-item"><span class="meta-icon">svc</span> ${esc(m.service)}</span>
          <span class="meta-item"><span class="meta-icon">ram</span> ${m.ram_required_gb} GB</span>
        </div>
        ${shownAliases.length ? `
        <div class="aliases">
          <span class="aliases-label">aliases</span>
          ${shownAliases.map(a => `<span class="alias-chip">${esc(a)}</span>`).join('')}
          ${extraCount > 0 ? `<span class="alias-overflow">+${extraCount}</span>` : ''}
        </div>` : ''}
        <div class="card-actions">
          ${m.container_status === 'running' ? `
            ${m.active ? '' : `<button class="btn-ghost btn-sm" onclick="swapTo('${esc(m.key)}')" ${loadDisabled}>${ICO.load} ${loadLabel}</button>`}
            ${actionBtn('Stop',  'stop',  `modelAction('${esc(m.key)}','stop')`)}
          ` : `
            <button class="btn-ghost btn-sm" onclick="swapTo('${esc(m.key)}')" ${loadDisabled}>${ICO.load} ${loadLabel}</button>
          `}
          <button class="btn-icon danger" onclick="uninstallModel('${esc(m.key)}')" title="Uninstall">${ICO.trash}</button>
        </div>
      </div>`;
    }).join('');
  } catch (e) {
    document.getElementById('models-list').innerHTML = `<p class="empty">Error: ${esc(e.message)}</p>`;
  }
}

async function swapTo(key) {
  if (isSwapping()) return;
  _swapping = true;
  _swappingKey = key;
  refreshModels();
  try {
    toast(`Loading ${key}...`, 'ok');
    await POST(`/router/swap/${key}`);
    toast(`${key} is active`, 'ok');
  } catch (e) { toast(`Swap failed: ${e.message}`, 'err'); }
  _swapping = false;
  _swappingKey = null;
  refreshModels();
}

async function modelAction(key, action) {
  try {
    await POST(`/router/models/${key}/${action}`);
    toast(`${key}: ${action} OK`, 'ok');
    refreshModels();
  } catch (e) {
    if (e.status === 409) {
      const msg = e.body?.error?.message || e.body?.error?.details?.message || 'A download is in progress.';
      showWarnModal(`Cannot ${action} ${key}`, msg, `Force ${action}`, async () => {
        try {
          await POST(`/router/models/${key}/${action}?force=true`);
          toast(`${key}: ${action} (forced) OK`, 'ok');
          refreshModels();
        } catch (e2) { toast(`Force ${action} failed: ${e2.message}`, 'err'); }
      });
    } else {
      toast(`${action} failed: ${e.message}`, 'err');
    }
  }
}

async function uninstallModel(key) {
  if (!confirm(`Uninstall "${key}"?\nRemoves from config only — model weights are kept on disk.`)) return;
  try {
    await DELETE(`/router/models/${key}/uninstall`);
    toast(`${key} uninstalled`, 'ok');
    refreshModels();
  } catch (e) { toast(`Uninstall failed: ${e.message}`, 'err'); }
}

// ── Install dialog ────────────────────────────────────────────────────────────
// Cache backends from API
let _backends = [];

// ── HF Token management ───────────────────────────────────────────────────────
async function openHfTokenDialog() {
  const dlg = document.getElementById('hf-token-dialog');
  const status = document.getElementById('hf-token-status');
  const input = document.getElementById('hf-token-input');
  input.value = '';
  status.textContent = 'Checking...';
  dlg.showModal();
  try {
    const r = await GET('/router/hf-token');
    if (r.configured) {
      status.innerHTML = `Token configured: <code>${esc(r.masked)}</code>`;
    } else {
      status.textContent = 'No token configured.';
    }
  } catch {
    status.textContent = 'Could not check token status.';
  }
}

async function saveHfToken() {
  const token = document.getElementById('hf-token-input').value.trim();
  if (!token) { toast('Enter a token', 'err'); return; }
  try {
    await POST('/router/hf-token', { token });
    document.getElementById('hf-token-dialog').close();
    toast('HF token saved', 'ok');
    updateHfTokenBtn();
  } catch (e) { toast(`Failed: ${e.message}`, 'err'); }
}

async function removeHfToken() {
  try {
    await DELETE('/router/hf-token');
    document.getElementById('hf-token-dialog').close();
    toast('HF token removed', 'ok');
    updateHfTokenBtn();
  } catch (e) { toast(`Failed: ${e.message}`, 'err'); }
}

async function updateHfTokenBtn() {
  try {
    const r = await GET('/router/hf-token');
    const btn = document.getElementById('hf-token-btn');
    if (r.configured) {
      btn.classList.add('btn-token-set');
    } else {
      btn.classList.remove('btn-token-set');
    }
  } catch {}
}

// Check token status on load
updateHfTokenBtn();

// ── Install dialog ────────────────────────────────────────────────────────────
async function openInstallDialog() {
  document.getElementById('inst-key').value = '';
  document.getElementById('inst-model-path').value = '';
  document.getElementById('inst-image').value = '';
  document.getElementById('inst-command').value = '';
  document.getElementById('inst-download').checked = true;

  // Populate backend dropdown
  const sel = document.getElementById('inst-backend');
  if (!_backends.length) {
    try {
      const r = await GET('/router/backends');
      _backends = r.backends;
    } catch { _backends = [{ key: 'sglang', label: 'SGLang', port: 30000, model_path_label: 'HuggingFace model', model_path_placeholder: '' }]; }
  }
  sel.innerHTML = _backends.map(b => `<option value="${esc(b.key)}">${esc(b.label)}</option>`).join('');
  onBackendChange();

  document.getElementById('install-dialog').showModal();
}

function onBackendChange() {
  const sel = document.getElementById('inst-backend');
  const b = _backends.find(x => x.key === sel.value) || _backends[0];
  if (!b) return;
  const label = document.getElementById('inst-model-path-label');
  const input = document.getElementById('inst-model-path');
  label.childNodes[0].textContent = b.model_path_label + ' ';
  input.placeholder = b.model_path_placeholder || '';
  // Show/hide Docker image and command fields for custom backend
  document.getElementById('inst-image-label').style.display = b.custom ? '' : 'none';
  document.getElementById('inst-command-label').style.display = b.custom ? '' : 'none';
}

async function submitInstall() {
  const key       = document.getElementById('inst-key').value.trim();
  const backend   = document.getElementById('inst-backend').value;
  const modelPath = document.getElementById('inst-model-path').value.trim();
  const image     = document.getElementById('inst-image').value.trim();
  const command   = document.getElementById('inst-command').value.trim();
  const download  = document.getElementById('inst-download').checked;

  if (!key)       { toast('Model name is required', 'err'); return; }
  if (!modelPath) { toast('Model path is required', 'err'); return; }
  if (/[^a-zA-Z0-9._-]/.test(key)) { toast('Model name: letters, numbers, dots, dashes only', 'err'); return; }

  const b = _backends.find(x => x.key === backend);
  if (b && b.custom && !image) { toast('Docker image is required for custom backend', 'err'); return; }

  const port = b ? b.port : 30000;
  const service = `${backend}-${key}`;
  const aliases = [key, modelPath];
  const parts = modelPath.split('/');
  if (parts.length > 1) aliases.push(parts[parts.length - 1]);

  const payload = {
    key,
    backend,
    service,
    model_path: modelPath,
    base_url: `http://${service}:${port}`,
    ram_required_gb: 0,
    aliases,
    download,
  };
  if (image) payload.image = image;
  if (command) payload.command = command;

  try {
    const result = await POST('/router/models/install', payload);
    document.getElementById('install-dialog').close();
    toast(`${key} installed${result.download_job_id ? ' — download started' : ''}`, 'ok');
    refreshModels();
    if (result.download_job_id) {
      document.querySelector('[data-tab="jobs"]').click();
    }
  } catch (e) { toast(`Install failed: ${e.message}`, 'err'); }
}

// ── Sets tab ──────────────────────────────────────────────────────────────────
async function refreshSets() {
  const el = document.getElementById('sets-list');
  if (el && !el.children.length) el.innerHTML = '<div class="loading">Fetching data</div>';
  try {
    const { sets } = await GET('/router/sets');
    const el = document.getElementById('sets-list');
    if (!sets.length) {
      el.innerHTML = '<p class="empty">No sets configured. Create one to run multiple models together.</p>';
      return;
    }
    el.innerHTML = sets.map(s => `
      <div class="card ${s.active ? 'active-card set-card' : 'set-card'}">
        <div class="card-header">
          <div>
            <div class="card-title">${esc(s.key)}</div>
            <div class="card-subtitle">${esc(s.description || 'No description')}</div>
          </div>
          <div class="card-chips">
            ${s.active ? chip('active', 'active') : ''}
          </div>
        </div>
        <div class="card-meta">
          <span class="meta-item"><span class="meta-icon">ram</span> ${s.ram_required_gb.toFixed(0)} GB total</span>
          <span class="meta-item"><span class="meta-icon">#</span> ${s.members.length} model${s.members.length !== 1 ? 's' : ''}</span>
        </div>
        <div class="aliases">
          ${s.members.map(m => {
            const running = m.container_status === 'running';
            return `<span class="alias-chip member-chip ${running ? 'member-running' : 'member-stopped'}">${esc(m.key)}</span>`;
          }).join('')}
        </div>
        ${s.aliases && s.aliases.length ? `
        <div class="aliases">${s.aliases.map(a => `<span class="alias-chip">${esc(a)}</span>`).join('')}</div>` : ''}
        <div class="card-actions">
          ${actionBtn('Load set', 'load', `setAction('${esc(s.key)}','start')`)}
          ${actionBtn('Stop all', 'stop', `setAction('${esc(s.key)}','stop')`)}
          <button class="btn-icon danger" onclick="deleteSet('${esc(s.key)}')" title="Delete set">${ICO.trash}</button>
        </div>
      </div>`).join('');
  } catch (e) {
    document.getElementById('sets-list').innerHTML = `<p class="empty">Error: ${esc(e.message)}</p>`;
  }
}

async function openCreateSetDialog() {
  // Reset form
  document.getElementById('set-key').value = '';
  document.getElementById('set-desc').value = '';
  // Populate model checkboxes from registered models
  const container = document.getElementById('set-model-checkboxes');
  container.innerHTML = '<span class="empty" style="padding:.5rem 0">Fetching data...</span>';
  document.getElementById('create-set-dialog').showModal();
  try {
    const { models } = await GET('/router/models');
    if (!models.length) {
      container.innerHTML = '<span class="empty" style="padding:.5rem 0">No models registered. Install models first.</span>';
      return;
    }
    container.innerHTML = models.map(m => `
      <label class="set-model-item">
        <input type="checkbox" value="${esc(m.key)}" onchange="this.closest('.set-model-item').classList.toggle('checked', this.checked)" />
        <div class="smi-info">
          <div class="smi-name">${esc(m.key)}</div>
          <div class="smi-path">${esc(m.model_path)}</div>
        </div>
        <span class="smi-ram">${m.ram_required_gb} GB</span>
      </label>`).join('');
  } catch (e) {
    container.innerHTML = `<span class="empty" style="padding:.5rem 0">Error loading models</span>`;
  }
}

async function submitCreateSet() {
  const key  = document.getElementById('set-key').value.trim();
  const desc = document.getElementById('set-desc').value.trim();
  const checkboxes = document.querySelectorAll('#set-model-checkboxes input[type="checkbox"]:checked');
  const members = Array.from(checkboxes).map(cb => cb.value);

  if (!key) { toast('Set name is required', 'err'); return; }
  if (members.length < 2) { toast('Select at least 2 models for a set', 'err'); return; }
  try {
    await POST('/router/sets', {
      key,
      description: desc || undefined,
      members,
      primary: members[0],
      aliases: [key],
    });
    document.getElementById('create-set-dialog').close();
    toast(`Set "${key}" created`, 'ok');
    refreshSets();
  } catch (e) { toast(`Create set failed: ${e.message}`, 'err'); }
}

async function setAction(key, action) {
  try {
    toast(`Set ${key}: ${action}...`, 'ok');
    await POST(`/router/sets/${key}/${action}`);
    toast(`Set ${key}: ${action} OK`, 'ok');
    refreshSets();
  } catch (e) {
    if (e.status === 409) {
      const msg = e.body?.error?.message || 'A download is in progress.';
      showWarnModal(`Cannot ${action} set ${key}`, msg, `Force ${action}`, async () => {
        try {
          await POST(`/router/sets/${key}/${action}?force=true`);
          toast(`Set ${key}: ${action} (forced) OK`, 'ok');
          refreshSets();
        } catch (e2) { toast(`Force ${action} failed: ${e2.message}`, 'err'); }
      });
    } else {
      toast(`${action} failed: ${e.message}`, 'err');
    }
  }
}

async function deleteSet(key) {
  if (!confirm(`Delete set "${key}"?`)) return;
  try {
    await DELETE(`/router/sets/${key}`);
    toast(`Set "${key}" deleted`, 'ok');
    refreshSets();
  } catch (e) { toast(`Delete failed: ${e.message}`, 'err'); }
}

// ── Jobs tab ──────────────────────────────────────────────────────────────────
async function refreshJobs() {
  const el = document.getElementById('jobs-list');
  if (el && !el.children.length) el.innerHTML = '<div class="loading">Fetching data</div>';
  try {
    const { jobs } = await GET('/router/jobs');
    const el = document.getElementById('jobs-list');
    if (!jobs.length) { el.innerHTML = '<p class="empty">No active jobs. Downloads started from Install will appear here.</p>'; return; }
    el.innerHTML = [...jobs].reverse().map(j => {
      const isActive = j.status === 'running' || j.status === 'pending';
      return `
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">${esc(j.type)}</div>
            <div class="card-subtitle">${esc(j.message || '...')}</div>
          </div>
          <div class="card-chips">
            ${jobChip(j.status)}
          </div>
        </div>
        ${isActive ? `
        <div style="background:var(--surface3);border:1px solid var(--border);border-radius:6px;padding:.4rem .7rem;font-size:.72rem;color:var(--text-dim)">Check terminal below for details</div>
        ` : ''}
        ${j.result ? `<div class="card-meta"><span class="meta-item card-subtitle">${esc(j.result.local_path || JSON.stringify(j.result))}</span></div>` : ''}
        ${isActive ? `
        <div class="card-actions">
          ${actionBtn('Terminate', 'cancel', `terminateJob('${j.id}')`)}
        </div>` : ''}
      </div>`;
    }).join('');
  } catch (e) {
    document.getElementById('jobs-list').innerHTML = `<p class="empty">Error: ${esc(e.message)}</p>`;
  }
}

async function terminateJob(id) {
  if (!confirm('Terminate this job?')) return;
  try { await POST(`/router/jobs/${id}/cancel`); toast('Job terminated', 'ok'); refreshJobs(); }
  catch (e) { toast(e.message, 'err'); }
}

// ── Services tab ──────────────────────────────────────────────────────────────
async function refreshServices() {
  const el = document.getElementById('services-list');
  if (el && !el.children.length) el.innerHTML = '<div class="loading">Fetching data</div>';
  try {
    const { services } = await GET('/router/services');
    const el = document.getElementById('services-list');
    if (!services.length) { el.innerHTML = '<p class="empty">No compose services found.</p>'; return; }
    el.innerHTML = services.map(s => `
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">${esc(s.service)}</div>
            <div class="card-subtitle">${esc(s.image)}</div>
          </div>
          <div class="card-chips">
            ${containerChip(s.status, s.healthy, s.downloading)}
          </div>
        </div>
        <div class="card-meta">
          <span class="meta-item"><span class="meta-icon">ctr</span> ${esc(s.container)}</span>
          ${s.profiles.length ? `<span class="meta-item"><span class="meta-icon">profile</span> ${esc(s.profiles.join(', '))}</span>` : ''}
        </div>
        <div class="card-actions">
          ${actionBtn('Start',   'play',    `svcAction('${esc(s.service)}','start')`)}
          ${actionBtn('Restart', 'restart', `svcAction('${esc(s.service)}','restart')`)}
          ${actionBtn('Stop',    'stop',    `svcAction('${esc(s.service)}','stop')`)}
        </div>
      </div>`).join('');
  } catch (e) {
    document.getElementById('services-list').innerHTML = `<p class="empty">Error: ${esc(e.message)}</p>`;
  }
}

// Cache last system state for RAM display in confirmation
let _lastSystem = null;

// Generic warning modal — replaces native confirm() for consistent UI
function showWarnModal(title, message, btnLabel, onConfirm) {
  const dlg = document.getElementById('confirm-warn-dialog');
  document.getElementById('confirm-warn-title').textContent = title;
  document.getElementById('confirm-warn-message').innerHTML = message;
  const btn = document.getElementById('confirm-warn-btn');
  btn.textContent = btnLabel;
  btn.onclick = () => { dlg.close(); onConfirm(); };
  dlg.showModal();
}

async function svcAction(service, action) {
  // For start actions on sglang services, show OOM warning
  if (action === 'start' && service.startsWith('sglang')) {
    const sys = _lastSystem || await GET('/router/system').catch(() => null);
    const availGB = sys ? sys.ram_available_gb.toFixed(0) : '?';
    const totalGB = sys ? sys.ram_total_gb.toFixed(0) : '?';
    // Find matching model config for RAM requirement
    let modelRAM = '~35-40';
    try {
      const { models } = await GET('/router/models');
      const match = models.find(m => m.service === service);
      if (match) modelRAM = match.ram_required_gb;
    } catch {}

    showWarnModal(
      `Start ${service} directly?`,
      `This service requires <strong>~${modelRAM} GB</strong> RAM. ` +
      `Currently <strong>${availGB} GB</strong> of ${totalGB} GB available.` +
      `<br><br>Starting directly bypasses the router's RAM check and swap logic. ` +
      `If another model is already loaded, this may cause an <strong>OOM crash</strong>.` +
      `<br><br>Use <strong>Load</strong> on the Models tab for safe, managed startup.`,
      'Start anyway',
      () => confirmStartService(service),
    );
    return;
  }
  try {
    await POST(`/router/services/${service}/${action}`);
    toast(`${service}: ${action} OK`, 'ok');
    refreshServices();
  } catch (e) {
    if (e.status === 409) {
      const msg = e.body?.error?.message || 'A download is in progress.';
      showWarnModal(`Cannot ${action} ${service}`, msg, `Force ${action}`, async () => {
        try {
          await POST(`/router/services/${service}/${action}?force=true`);
          toast(`${service}: ${action} (forced) OK`, 'ok');
          refreshServices();
        } catch (e2) { toast(`Force ${action} failed: ${e2.message}`, 'err'); }
      });
    } else {
      toast(`${action} failed: ${e.message}`, 'err');
    }
  }
}

async function confirmStartService(service) {
  try {
    await POST(`/router/services/${service}/start`);
    toast(`${service}: start OK`, 'ok');
    refreshServices();
  } catch (e) { toast(`Start failed: ${e.message}`, 'err'); }
}

// ── Config tab ────────────────────────────────────────────────────────────────
async function refreshConfig() {
  try {
    const [system, models, sets] = await Promise.all([
      GET('/router/system'),
      GET('/router/models'),
      GET('/router/sets'),
    ]);
    const el = document.getElementById('config-info');
    const row = (k, v) => `<tr><td>${k}</td><td>${v}</td></tr>`;
    el.innerHTML = `
      <div class="config-grid">
        <div class="config-card">
          <h4>Router state</h4>
          <table class="config-table">
            ${row('Active model',  system.active_model  || '<span style="color:var(--text-dim)">none</span>')}
            ${row('Active set',    system.active_set    || '<span style="color:var(--text-dim)">none</span>')}
            ${row('Swap pending',  system.swap_pending  ? '<span style="color:var(--yellow)">yes</span>' : 'no')}
            ${row('In-flight',     system.in_flight)}
          </table>
        </div>
        <div class="config-card">
          <h4>Memory</h4>
          <table class="config-table">
            ${row('Total',     system.ram_total_gb.toFixed(1) + ' GB')}
            ${row('Used',      system.ram_used_gb.toFixed(1)  + ' GB')}
            ${row('Available', system.ram_available_gb.toFixed(1) + ' GB')}
            ${row('Usage',     `<span style="color:${system.ram_used_gb/system.ram_total_gb > .9 ? 'var(--red)' : system.ram_used_gb/system.ram_total_gb > .75 ? 'var(--yellow)' : 'var(--green)'}">${(system.ram_used_gb/system.ram_total_gb*100).toFixed(0)}%</span>`)}
          </table>
        </div>
      </div>
      <div class="config-grid" style="margin-top:1rem">
        <div class="config-card">
          <h4>Registered models (${models.models.length})</h4>
          <table class="config-table">
            ${models.models.map(m =>
              row(`<span style="color:var(--text)">${esc(m.key)}</span>`,
                  `${esc(m.model_path)} <span style="color:var(--text-dim)">${m.ram_required_gb} GB</span>`)).join('')}
          </table>
        </div>
        ${sets.sets.length ? `
        <div class="config-card">
          <h4>Sets (${sets.sets.length})</h4>
          <table class="config-table">
            ${sets.sets.map(s =>
              row(`<span style="color:var(--text)">${esc(s.key)}</span>`,
                  `${s.members.map(m => esc(m.key)).join(', ')} <span style="color:var(--text-dim)">${s.ram_required_gb.toFixed(0)} GB</span>`)).join('')}
          </table>
        </div>` : ''}
      </div>`;
  } catch (e) {
    document.getElementById('config-info').innerHTML = `<p class="empty">Error: ${esc(e.message)}</p>`;
  }
}

async function reloadConfig() {
  try {
    await POST('/router/config/reload');
    toast('Config reloaded', 'ok');
    refreshConfig();
    refreshModels();
    refreshSets();
  } catch (e) { toast(`Reload failed: ${e.message}`, 'err'); }
}

// ── Utility ───────────────────────────────────────────────────────────────────
function esc(s) {
  if (typeof s !== 'string') return s;
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// ── Polling loop ──────────────────────────────────────────────────────────────
function getActiveTab() {
  return document.querySelector('.tab-btn.active')?.dataset.tab || 'services';
}

function pollAll() {
  refreshSystem();
  const tab = getActiveTab();
  if (tab === 'models')   refreshModels();
  if (tab === 'sets')     refreshSets();
  if (tab === 'jobs')     refreshJobs();
  if (tab === 'services') refreshServices();
  if (tab === 'status')   refreshConfig();
}

// Header RAM bar always polls at 1s
setInterval(refreshSystem, 1000);

// Tab content polls at 3s
pollAll();
setInterval(pollAll, 3000);

// ── WebSocket log stream ──────────────────────────────────────────────────────
const MAX_LOG_LINES = 500;
let _logWs = null;
let _logConnected = false;

function connectLogWs() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  _logWs = new WebSocket(`${proto}//${location.host}/ws/logs`);

  _logWs.onopen = () => {
    _logConnected = true;
    appendLog('Connected to log stream', 'info');
    populateLogSources();
  };

  _logWs.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.type === 'log') {
        appendLog(msg.line, classifyLine(msg.line));
      } else if (msg.type === 'info') {
        appendLog(msg.message, 'info');
        document.getElementById('log-title').textContent = msg.message.replace('Streaming logs from ', '');
      }
    } catch {}
  };

  _logWs.onclose = () => {
    _logConnected = false;
    appendLog('Disconnected — reconnecting...', 'warn');
    setTimeout(connectLogWs, 3000);
  };

  _logWs.onerror = () => {
    _logWs.close();
  };
}

function classifyLine(line) {
  const lower = line.toLowerCase();
  if (lower.includes('error') || lower.includes('exception') || lower.includes('traceback') || lower.includes('fatal'))
    return 'err';
  if (lower.includes('warning') || lower.includes('warn'))
    return 'warn';
  if (lower.includes('info'))
    return 'info';
  return '';
}

let _logAutoScroll = true;

function appendLog(text, cls) {
  const body = document.getElementById('log-body');
  if (!body) return;
  const div = document.createElement('div');
  div.className = `log-line${cls ? ` log-line-${cls}` : ''}`;
  div.textContent = text;
  body.appendChild(div);

  // Trim old lines
  while (body.children.length > MAX_LOG_LINES) {
    body.removeChild(body.firstChild);
  }

  if (_logAutoScroll) body.scrollTop = body.scrollHeight;
  updateScrollBtn();
}

function updateScrollBtn() {
  const body = document.getElementById('log-body');
  const btn = document.getElementById('log-scroll-btn');
  if (!body || !btn) return;
  const atBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 40;
  _logAutoScroll = atBottom;
  btn.style.display = atBottom ? 'none' : 'flex';
}

function scrollToBottom() {
  const body = document.getElementById('log-body');
  if (!body) return;
  body.scrollTop = body.scrollHeight;
  _logAutoScroll = true;
  updateScrollBtn();
}

// Attach scroll listener after DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('log-body')?.addEventListener('scroll', updateScrollBtn);
  });
} else {
  document.getElementById('log-body')?.addEventListener('scroll', updateScrollBtn);
}

function clearLogs() {
  document.getElementById('log-body').innerHTML = '';
}

function toggleLogPanel() {
  document.getElementById('log-panel').classList.toggle('collapsed');
}

async function populateLogSources() {
  const sel = document.getElementById('log-source');
  try {
    const { services } = await GET('/router/services');
    sel.innerHTML = services.map(s =>
      `<option value="${esc(s.container)}">${esc(s.service)}</option>`
    ).join('');
    // Select active model's container if available
    if (_lastSystem && _lastSystem.active_model) {
      try {
        const { models } = await GET('/router/models');
        const active = models.find(m => m.active);
        if (active) {
          const svc = services.find(s => s.service === active.service);
          if (svc) sel.value = svc.container;
        }
      } catch {}
    }
  } catch {}
}

function switchLogSource() {
  const sel = document.getElementById('log-source');
  if (_logWs && _logConnected && sel.value) {
    clearLogs();
    _logWs.send(JSON.stringify({ container: sel.value }));
  }
}

// ── Log panel drag resize ─────────────────────────────────────────────────────
function initLogDrag() {
  const panel = document.getElementById('log-panel');
  const handle = document.getElementById('log-drag-handle');
  if (!panel || !handle) return;

  // Restore saved height
  const saved = localStorage.getItem('log-panel-height');
  if (saved) panel.style.height = saved + 'px';

  let startY, startH;

  handle.addEventListener('mousedown', (e) => {
    e.preventDefault();
    startY = e.clientY;
    startH = panel.offsetHeight;
    panel.classList.add('dragging');
    document.addEventListener('mousemove', onDrag);
    document.addEventListener('mouseup', onDragEnd);
  });

  function onDrag(e) {
    const newH = Math.min(Math.max(startH + (startY - e.clientY), 80), window.innerHeight * 0.8);
    panel.style.height = newH + 'px';
  }

  function onDragEnd() {
    panel.classList.remove('dragging');
    document.removeEventListener('mousemove', onDrag);
    document.removeEventListener('mouseup', onDragEnd);
    localStorage.setItem('log-panel-height', panel.offsetHeight);
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initLogDrag);
} else {
  initLogDrag();
}

// Start WebSocket after DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', connectLogWs);
} else {
  connectLogWs();
}

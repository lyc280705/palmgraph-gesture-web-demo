const state = {
  actionDefs: [],
  config: null,
  gestureLabels: {},
  modelOptions: [],
  activeModelId: '',
};

const mappingTemplate = document.getElementById('mappingTemplate');
const mappingList = document.getElementById('mappingList');
const actionLog = document.getElementById('actionLog');

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function formatBytes(sizeBytes) {
  const value = Number(sizeBytes || 0);
  if (!Number.isFinite(value) || value <= 0) {
    return '-';
  }
  if (value >= 1024 * 1024) {
    return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  }
  if (value >= 1024) {
    return `${Math.round(value / 1024)} KB`;
  }
  return `${Math.round(value)} B`;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `请求失败: ${response.status}`);
  }
  return response.json();
}

function actionTitle(actionId) {
  const found = state.actionDefs.find((item) => item.id === actionId);
  return found ? found.title : actionId;
}

function renderParams(host, actionId, binding) {
  host.innerHTML = '';
  const def = state.actionDefs.find((item) => item.id === actionId);
  if (!def || !def.fields.length) {
    return;
  }
  const grid = document.createElement('div');
  grid.className = 'param-grid';
  for (const field of def.fields) {
    const label = document.createElement('label');
    if (field.multiline) {
      label.dataset.wide = 'true';
    }
    const title = document.createElement('span');
    title.textContent = field.label;
    label.appendChild(title);

    const control = field.multiline ? document.createElement('textarea') : document.createElement('input');
    control.className = 'param-input';
    control.dataset.fieldName = field.name;
    control.placeholder = field.placeholder || '';
    control.value = binding.params?.[field.name] || '';
    label.appendChild(control);
    grid.appendChild(label);
  }
  host.appendChild(grid);
}

function collectBindings() {
  const bindings = {};
  document.querySelectorAll('.mapping-card').forEach((card) => {
    bindings[card.dataset.gesture] = readBindingFromCard(card);
  });
  return bindings;
}

function readBindingFromCard(card) {
  const actionId = card.querySelector('.action-select').value;
  const enabled = card.querySelector('.binding-enabled').checked;
  const params = {};
  card.querySelectorAll('.param-input').forEach((input) => {
    params[input.dataset.fieldName] = input.value;
  });
  return { enabled, action_id: actionId, params };
}

function bindingPreview(binding) {
  const params = binding.params || {};
  const extras = Object.entries(params)
    .filter(([, value]) => value)
    .map(([key, value]) => `${key}: ${value}`)
    .join(' | ');
  return extras ? `${actionTitle(binding.action_id)} | ${extras}` : actionTitle(binding.action_id);
}

function createMappingCard(gesture, binding) {
  const fragment = mappingTemplate.content.cloneNode(true);
  const card = fragment.querySelector('.mapping-card');
  card.dataset.gesture = gesture;
  card.querySelector('.gesture-cn').textContent = state.gestureLabels[gesture] || gesture;
  card.querySelector('.gesture-en').textContent = gesture;

  const enabledInput = card.querySelector('.binding-enabled');
  enabledInput.checked = Boolean(binding.enabled);

  const actionSelect = card.querySelector('.action-select');
  for (const def of state.actionDefs) {
    const option = document.createElement('option');
    option.value = def.id;
    option.textContent = def.title;
    actionSelect.appendChild(option);
  }
  actionSelect.value = binding.action_id;

  const paramsHost = card.querySelector('.params-host');
  renderParams(paramsHost, binding.action_id, binding);

  const preview = card.querySelector('.mapping-preview');
  preview.textContent = bindingPreview(binding);

  actionSelect.addEventListener('change', () => {
    const nextBinding = { ...readBindingFromCard(card), action_id: actionSelect.value };
    renderParams(paramsHost, actionSelect.value, nextBinding);
    preview.textContent = bindingPreview(readBindingFromCard(card));
  });

  paramsHost.addEventListener('input', () => {
    preview.textContent = bindingPreview(readBindingFromCard(card));
  });

  enabledInput.addEventListener('change', () => {
    preview.textContent = enabledInput.checked ? bindingPreview(readBindingFromCard(card)) : '当前手势已禁用';
  });

  card.querySelector('.test-button').addEventListener('click', async () => {
    const originalText = preview.textContent;
    preview.textContent = '正在执行测试动作...';
    try {
      const result = await fetchJson('/api/test-binding', {
        method: 'POST',
        body: JSON.stringify({ gesture, binding: readBindingFromCard(card) }),
      });
      preview.textContent = result.event.success ? `测试成功: ${result.event.detail}` : `测试失败: ${result.event.detail}`;
      await refreshState();
    } catch (error) {
      preview.textContent = `测试失败: ${error.message}`;
    }
    setTimeout(() => {
      preview.textContent = originalText;
    }, 3000);
  });

  return fragment;
}

function renderMappings(config) {
  mappingList.innerHTML = '';
  for (const [gesture, binding] of Object.entries(config.bindings)) {
    mappingList.appendChild(createMappingCard(gesture, binding));
  }
}

function renderLogs(items) {
  if (!items.length) {
    actionLog.innerHTML = '<div class="log-item">还没有执行记录。你可以先点击某个手势卡片里的“测试动作”。</div>';
    return;
  }
  actionLog.innerHTML = items
    .map((item) => {
      const statusClass = item.success ? 'log-ok' : 'log-fail';
      const statusText = item.success ? '成功' : '失败';
      return `
        <article class="log-item">
          <div class="log-topline">
            <div>
              <div class="log-title">${escapeHtml(item.gesture_zh)} → ${escapeHtml(item.action_title)}</div>
              <div class="log-time">${escapeHtml(item.time)} · ${escapeHtml(item.source === 'manual' ? '手动测试' : '手势触发')}</div>
            </div>
            <strong class="${statusClass}">${statusText}</strong>
          </div>
          <div class="log-detail">${escapeHtml(item.detail || '')}</div>
        </article>
      `;
    })
    .join('');
}

function renderModelSwitcher(payload) {
  const select = document.getElementById('modelSelect');
  const button = document.getElementById('switchModelBtn');
  const activeText = document.getElementById('activeModelText');
  const note = document.getElementById('modelSwitchNote');
  const options = payload.model_options || state.modelOptions || [];
  const activeId = payload.active_model_id || state.activeModelId || '';
  const activeTitle = payload.active_model_title || '';

  state.modelOptions = options;
  state.activeModelId = activeId;

  const currentIds = Array.from(select.options).map((item) => item.value).join('|');
  const nextIds = options.map((item) => item.id).join('|');
  if (currentIds !== nextIds) {
    select.innerHTML = '';
    for (const item of options) {
      const option = document.createElement('option');
      option.value = item.id;
      option.textContent = item.file_name ? `${item.title} | ${item.file_name}` : item.title;
      select.appendChild(option);
    }
  }

  if (!options.length) {
    select.innerHTML = '<option value="">当前会话没有可切换模型</option>';
    select.disabled = true;
    button.disabled = true;
    activeText.textContent = '固定模型';
    note.textContent = '当前启动方式没有可切换的备用模型。';
    return;
  }

  const activeOption = options.find((item) => item.id === activeId) || options[0];
  if (document.activeElement !== select) {
    select.value = activeOption.id;
  }
  activeText.textContent = activeTitle || activeOption.title;
  note.textContent = `${activeOption.file_name} · ${formatBytes(activeOption.size_bytes)}`;

  const disabled = options.length <= 1;
  select.disabled = disabled;
  button.disabled = disabled;
}

function updateHeader(stateJson) {
  document.getElementById('modelName').textContent = stateJson.model_display_name || stateJson.model_name || '未知模型';
  document.getElementById('paramCount').textContent = Number(stateJson.param_count || 0).toLocaleString('zh-CN');
  document.getElementById('cameraBadge').textContent = stateJson.camera_ready ? '摄像头在线' : '摄像头未就绪';
  document.getElementById('cameraBadge').className = stateJson.camera_ready ? 'badge badge-live' : 'badge badge-muted';
  document.getElementById('controlBadge').textContent = stateJson.control_enabled ? '控制已开启' : '仅识别不控制';
  document.getElementById('controlBadge').className = stateJson.control_enabled ? 'badge badge-live' : 'badge badge-muted';
  renderModelSwitcher(stateJson);
}

function updateRealtimeState(stateJson) {
  document.getElementById('gestureZh').textContent = stateJson.gesture_zh || '无手势';
  document.getElementById('gestureEn').textContent = stateJson.gesture || 'no_gesture';
  const conf = Number(stateJson.confidence || 0);
  const confEl = document.getElementById('gestureConf');
  confEl.textContent = conf.toFixed(2);
  const threshold = Number(stateJson.confidence_threshold || 0.7);
  confEl.style.color = conf >= threshold ? '' : '#b3442f';
  document.getElementById('fpsText').textContent = Number(stateJson.fps || 0).toFixed(1);
  const isNoGesture = stateJson.gesture === 'no_gesture';
  const controlOff = !stateJson.control_enabled;
  if (controlOff) {
    document.getElementById('bindingTitle').textContent = '仅识别不控制';
  } else if (isNoGesture) {
    document.getElementById('bindingTitle').textContent = '无手势';
  } else {
    document.getElementById('bindingTitle').textContent = stateJson.binding?.action_title || '无绑定';
  }
  document.getElementById('rawGesture').textContent = `${stateJson.raw_gesture_zh || '无手势'} / ${stateJson.raw_gesture || 'no_gesture'}`;
  document.getElementById('detectState').textContent = stateJson.detected ? '已检测到手' : '未检测到手';
  updateHeader(stateJson);
  renderLogs(stateJson.recent_actions || []);
}

async function loadBootstrap() {
  const [actionJson, configJson, stateJson] = await Promise.all([
    fetchJson('/api/actions'),
    fetchJson('/api/config'),
    fetchJson('/api/state'),
  ]);
  state.actionDefs = actionJson.actions;
  state.gestureLabels = configJson.gesture_labels || actionJson.gesture_labels || {};
  state.config = configJson;
  state.modelOptions = stateJson.model_options || configJson.model_options || [];
  state.activeModelId = stateJson.active_model_id || configJson.active_model_id || '';
  document.getElementById('controlEnabled').checked = Boolean(configJson.control_enabled);
  document.getElementById('cooldownInput').value = Number(configJson.cooldown_seconds || 1.2).toFixed(1);
  document.getElementById('confThreshold').value = Number(configJson.confidence_threshold || 0.7).toFixed(2);
  document.getElementById('triggerDelayInput').value = Number(configJson.trigger_delay_seconds || 0.2).toFixed(2);
  renderMappings(configJson);
  updateRealtimeState(stateJson);
}

async function refreshState() {
  const data = await fetchJson('/api/state');
  updateRealtimeState(data);
}

async function saveConfig() {
  const button = document.getElementById('saveConfigBtn');
  const payload = {
    control_enabled: document.getElementById('controlEnabled').checked,
    cooldown_seconds: Number(document.getElementById('cooldownInput').value || 1.2),
    confidence_threshold: Number(document.getElementById('confThreshold').value || 0.7),
    trigger_delay_seconds: Number(document.getElementById('triggerDelayInput').value || 0.2),
    bindings: collectBindings(),
  };
  const original = button.textContent;
  button.textContent = '保存中...';
  try {
    state.config = await fetchJson('/api/config', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    button.textContent = '已保存';
    await refreshState();
  } catch (error) {
    button.textContent = '保存失败';
    alert(error.message);
  }
  setTimeout(() => {
    button.textContent = original;
  }, 1500);
}

async function switchModel() {
  const select = document.getElementById('modelSelect');
  const button = document.getElementById('switchModelBtn');
  const modelId = select.value;
  if (!modelId) {
    return;
  }
  const original = button.textContent;
  button.textContent = '切换中...';
  button.disabled = true;
  try {
    const stateJson = await fetchJson('/api/model', {
      method: 'POST',
      body: JSON.stringify({ model_id: modelId }),
    });
    state.activeModelId = stateJson.active_model_id || modelId;
    state.modelOptions = stateJson.model_options || state.modelOptions;
    updateRealtimeState(stateJson);
    button.textContent = '已切换';
  } catch (error) {
    button.textContent = '切换失败';
    alert(error.message);
  }
  setTimeout(() => {
    button.textContent = original;
    button.disabled = state.modelOptions.length <= 1;
  }, 1500);
}

document.getElementById('saveConfigBtn').addEventListener('click', saveConfig);
document.getElementById('switchModelBtn').addEventListener('click', switchModel);

loadBootstrap()
  .then(() => {
    refreshState();
    setInterval(() => {
      refreshState().catch((error) => {
        console.error(error);
      });
    }, 700);
  })
  .catch((error) => {
    actionLog.innerHTML = `<div class="log-item">页面初始化失败: ${escapeHtml(error.message)}</div>`;
  });
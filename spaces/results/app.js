import {COLORS, finite, validateSnapshot, points, last, extrema, evaluationSeconds, configJSON, demoSnapshot} from './data.mjs';

const $ = id => document.getElementById(id);
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const fmt = (n, digits = 3) => finite(n) ? n.toLocaleString('en-US', {maximumFractionDigits: digits}) : '—';
const compact = n => new Intl.NumberFormat('en-US', {notation:'compact', maximumFractionDigits:1}).format(n);
const duration = n => !finite(n) ? '—' : n >= 3600 ? `${fmt(n / 3600, 1)} h` : n >= 60 ? `${fmt(n / 60, 1)} min` : `${fmt(n, 1)} s`;
const title = run => String(run.label || run.id);
let snapshot = {schema_version: 1, runs: []}, realSnapshot = snapshot, selected = new Set(), synthetic = false, tab = 'overview';
const inExperiment = run => !$('experiment').value || JSON.stringify(run.group || '') === $('experiment').value;
const chosen = () => snapshot.runs.filter(run => inExperiment(run) && selected.has(run.id));
const visibleRuns = () => snapshot.runs.filter(run => inExperiment(run) && `${title(run)} ${run.group || ''} ${run.status || ''}`.toLowerCase().includes($('search').value.toLowerCase()));
const color = run => COLORS[snapshot.runs.indexOf(run) % COLORS.length];
const charts = new Map();
const cursor = {
  id: 'recorded-point-cursor',
  afterDatasetsDraw(chart) {
    const point = chart.getActiveElements()[0];
    if (!point) return;
    const {ctx, chartArea: {top, bottom}} = chart, x = point.element.x;
    ctx.save(); ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom);
    ctx.strokeStyle = '#8ba1ae'; ctx.lineWidth = 1; ctx.stroke(); ctx.restore();
  },
};

function message(text, error = false) { $('message').textContent = text; $('message').classList.toggle('error', error); $('message').hidden = !text; }
function downloadJSON(data, filename) {
  const blob = new Blob([JSON.stringify(data,null,2)],{type:'application/json'});
  const url = URL.createObjectURL(blob), a = document.createElement('a');
  a.href = url; a.download = filename; a.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
}
function setSnapshot(data, isDemo = false) {
  validateSnapshot(data); isDemo = isDemo || data.synthetic === true; snapshot = data; synthetic = isDemo;
  if (!isDemo) realSnapshot = data;
  $('experiment').replaceChildren(new Option('All experiments', ''), ...[...new Set(data.runs.map(run => run.group || ''))].sort().map(group => new Option(group || 'Ungrouped', JSON.stringify(group))));
  $('search').value = '';
  selected = new Set(data.runs.map(run => run.id)); render();
}
function runList() {
  const runs = visibleRuns(); $('run-list').replaceChildren();
  for (const run of runs) {
    const label = document.createElement('label'); label.className = 'run-item';
    const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.value = run.id; checkbox.checked = selected.has(run.id);
    checkbox.addEventListener('change', () => { checkbox.checked ? selected.add(run.id) : selected.delete(run.id); render(); });
    const dot = document.createElement('span'); dot.className = 'dot'; dot.style.background = color(run);
    const text = document.createElement('span'); text.append(document.createTextNode(title(run)));
    const small = document.createElement('small'); small.textContent = [run.group, run.status].filter(Boolean).join(' · ') || 'Status not supplied'; text.append(small);
    label.append(checkbox, dot, text); $('run-list').append(label);
  }
  $('run-count').textContent = `${chosen().length} selected · ${runs.length} shown`;
  $('no-runs').hidden = runs.length > 0;
}
function chart(id, metric, {factor = 1, smooth = false, unit = ''} = {}) {
  const host = $(id), axis = $('axis').value, smoothing = smooth ? Number($('smoothing').value) : 0;
  const series = chosen().map(run => ({run, data: points(run, metric, axis, smoothing).map(([x,y,raw]) => [x,y * factor, raw * factor])})).filter(s => s.data.length);
  if (!series.length) { host.innerHTML = '<div class="no-data">No recorded values for this metric and axis.<br>Try training step, or select another run.</div>'; return; }
  const axisLabel = $('axis').selectedOptions[0].textContent;
  host.innerHTML = `<div class="chart-canvas"><canvas role="img" aria-label="${esc(metric)} by ${esc(axisLabel)}. Hover or tap to inspect recorded points."></canvas></div><div class="chart-legend">` + series.map(s => `<span><i style="background:${color(s.run)}"></i>${esc(title(s.run))}</span>`).join('') + '</div>';
  charts.set(id, new Chart(host.querySelector('canvas'), {
    type: 'line',
    data: {datasets: series.map(({run, data}) => ({
      label: title(run), data: data.map(([x,y,raw]) => ({x,y,raw})),
      borderColor: color(run), backgroundColor: color(run), borderWidth: 2,
      pointRadius: data.length === 1 ? 4 : 0, pointHoverRadius: 4, tension: 0,
    }))},
    plugins: [cursor],
    options: {
      responsive: true, maintainAspectRatio: false, animation: false, parsing: false,
      interaction: {mode: 'nearest', axis: 'x', intersect: false},
      scales: {
        x: {type: 'linear', title: {display: true, text: axisLabel}, grid: {display: false}, ticks: {maxTicksLimit: 6, callback: value => Math.abs(value) >= 1000 ? compact(value) : fmt(value, 2)}},
        y: {grace: '10%', grid: {color: '#edf0f3'}, ticks: {maxTicksLimit: 5, callback: value => `${Math.abs(value) >= 1000 ? compact(value) : fmt(value, 3)}${unit}`}},
      },
      plugins: {
        legend: {display: false},
        tooltip: {
          backgroundColor: '#fff', titleColor: '#182631', bodyColor: '#52616d',
          borderColor: '#e5e9ed', borderWidth: 1, padding: 12, position: 'nearest',
          callbacks: {
            title: items => `${axisLabel}: ${fmt(items[0].parsed.x, 8)}`,
            label: item => `${item.dataset.label} · ${metric}: ${fmt(item.parsed.y, 8)}${unit}`,
            afterLabel: item => smoothing ? `Raw loss: ${fmt(item.raw.raw, 8)} · EMA: ${smoothing}` : '',
          },
        },
      },
    },
  }));
}
function table(host, headers, rows) {
  $(host).innerHTML = `<table><thead><tr>${headers.map(h => `<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.length ? rows.map(r => `<tr${r.different ? ' class="config-diff"' : ''}>${r.cells.map(cell => `<td>${cell}</td>`).join('')}</tr>`).join('') : `<tr><td colspan="${headers.length}">No recorded values for this selection.</td></tr>`}</tbody></table>`;
}
function summary() {
  const runs = chosen(); $('stat-runs').textContent = runs.length;
  for (const [key, id, direction] of [['eval/bpb','bpb','min'],['core/core','core','max']]) {
    const candidates = runs.map(run => ({run, value: extrema(run,key,direction)})).filter(x => finite(x.value)).sort((a,b) => direction === 'min' ? a.value-b.value : b.value-a.value);
    $( `stat-${id}`).textContent = candidates.length ? fmt(candidates[0].value,4) : '—';
    $(`stat-${id}-run`).textContent = candidates.length ? title(candidates[0].run) : 'No evaluation recorded';
  }
  const times = runs.map(evaluationSeconds).filter(finite); $('stat-eval').textContent = times.length ? duration(times.reduce((a,b)=>a+b,0)) : '—';
  const rows = runs.map(run => {
    let source = '';
    try { const url = new URL(run.source_url); if (url.protocol === 'https:') source = ` <a href="${esc(url.href)}" target="_blank" rel="noopener noreferrer" aria-label="Open source for ${esc(title(run))}">↗</a>`; } catch { /* source is optional */ }
    return {cells:[`<span style="color:${color(run)}">●</span> ${esc(title(run))}${source}`, `<span class="pill">${esc(run.status || 'unknown')}</span>`, esc(fmt(last(run,'train/total_tokens'),0)), esc(fmt(last(run,'eval/bpb'),4)), esc(fmt(last(run,'core/core'),4)), esc(fmt(last(run,'train/mfu'),2)), esc(fmt(last(run,'core/max_per_task'),0)), esc(duration(last(run,'train/total_training_time'))), esc(duration(last(run,'_runtime'))), esc(duration(evaluationSeconds(run)))]};
  });
  table('recap-table', ['Run','Status','Training tokens','Latest BPB','Latest CORE','Latest MFU (%)','CORE cap/task','Training time','Logged elapsed','Recorded eval'], rows);
}
function tasks() {
  const runs = chosen(), keys = [...new Set(runs.flatMap(run => run.history.flatMap(row => Object.keys(row).filter(k => /^core\/.+\/accuracy$/.test(k)))))].sort();
  table('tasks-table', ['Task', ...runs.map(title)], keys.map(key => ({cells:[esc(key.slice(5,-9)), ...runs.map(run => esc(fmt(last(run,key),4)))]})));
}
function configs() {
  const runs = chosen(), configs = runs.map(run => run.config || {}), keys = [...new Set(configs.flatMap(c => Object.keys(c)))].sort();
  const rows = keys.map(key => {
    const values = configs.map(config => Object.hasOwn(config,key) ? config[key] : undefined);
    const jsons = values.map(value => value === undefined ? '—' : configJSON(value,2));
    return {different: new Set(jsons).size > 1, cells:[`<span class="config-key">${esc(key)}</span>`, ...values.map((value,index) =>
      value && typeof value === 'object'
        ? `<details class="config-json"><summary>JSON ${Array.isArray(value) ? `array · ${value.length} items` : `object · ${Object.keys(value).length} keys`}</summary><pre tabindex="0" aria-label="${esc(key)} JSON for ${esc(title(runs[index]))}">${esc(jsons[index])}</pre></details>`
        : esc(jsons[index]))]};
  }).filter(row => !$('differences').checked || row.different);
  table('config-table', ['Setting', ...runs.map(title)], rows);
  $('config-downloads').replaceChildren();
  for (const run of runs) {
    const button = document.createElement('button'); button.className = 'button secondary';
    button.textContent = `${title(run)} · config.json ↓`;
    button.setAttribute('aria-label', `Download ${title(run)} configuration JSON`);
    button.addEventListener('click', () => downloadJSON(run.config || {}, `${synthetic ? 'synthetic-' : ''}${run.id.replace(/[^\w.-]+/g,'-')}-config.json`));
    $('config-downloads').append(button);
  }
}
function render() {
  for (const chart of charts.values()) chart.destroy(); charts.clear();
  $('loss-description').textContent = Number($('smoothing').value) ? `EMA · ${$('smoothing').value} · NATS` : 'RAW LOSS · NATS';
  runList(); $('empty').hidden = tab === 'evidence' || snapshot.runs.length > 0; $('workspace').hidden = tab === 'evidence' || !snapshot.runs.length;
  $('demo-banner').hidden = !synthetic; $('download').disabled = !chosen().length;
  $('snapshot-info').textContent = synthetic ? 'Synthetic interface preview' : `${snapshot.runs.length} runs · ${snapshot.exported_at ? `Snapshot ${snapshot.exported_at}` : 'Imported snapshot'}`;
  summary(); configs(); tasks();
  if (tab === 'overview') { chart('loss-chart','train/raw_loss',{smooth:true}); chart('bpb-chart','eval/bpb'); chart('throughput-chart','train/tokens_per_sec'); chart('mfu-chart','train/mfu',{unit:'%'}); }
  if (tab === 'evaluation') { chart('core-chart','core/core'); chart('eval-chart','core/step_time_ms',{factor:0.001}); }
}
for (const button of document.querySelectorAll('[data-tab]')) button.addEventListener('click', () => {
  tab = button.dataset.tab;
  for (const node of document.querySelectorAll('[data-tab]')) { node.classList.toggle('active',node.dataset.tab === tab); node.setAttribute('aria-pressed', node.dataset.tab === tab); }
  for (const node of document.querySelectorAll('.tab-panel')) node.hidden = node.id !== tab;
  const names = {overview:'Experiment overview', evaluation:'Evaluation results', configuration:'Training configuration', evidence:'Experiment archive'};
  $('page-title').replaceChildren(document.createTextNode(names[tab]),Object.assign(document.createElement('span'),{textContent:'.'}));
  $('selection-label').textContent = tab === 'overview' ? 'Learning curves' : tab === 'evaluation' ? 'Evaluation checkpoints' : 'Compare every setting';
  $('smoothing-label').hidden = tab !== 'overview'; $('axis').parentElement.hidden = tab === 'configuration'; render();
});
for (const id of ['axis','smoothing','differences','experiment']) $(id).addEventListener('change', render);
$('import-button').addEventListener('click', () => $('import').click());
$('search').addEventListener('input', runList);
$('select-all').addEventListener('click', () => { for (const run of visibleRuns()) selected.add(run.id); render(); });
$('clear-selection').addEventListener('click', () => { for (const run of visibleRuns()) selected.delete(run.id); render(); });
for (const id of ['demo','empty-demo']) $(id).addEventListener('click', () => { message(''); setSnapshot(demoSnapshot(),true); });
$('clear-demo').addEventListener('click', () => setSnapshot(realSnapshot));
$('import').addEventListener('change', async event => {
  const file = event.target.files[0]; if (!file) return;
  try { if (file.size > 25*1024*1024) throw new Error('Please import a snapshot smaller than 25 MiB.'); const data = JSON.parse(await file.text()); setSnapshot(data); message(`Loaded ${data.runs.length} runs from ${file.name}.`); }
  catch (error) { message(`Could not import results: ${error.message}`,true); }
  event.target.value = '';
});
$('download').addEventListener('click', () => {
  downloadJSON({...snapshot, runs:chosen(), ...(synthetic ? {synthetic:true} : {})}, synthetic ? 'synthetic-preview.json' : 'gpt-lab-results.json');
});
render();
try { const response = await fetch('./results.json', {cache:'no-cache'}); if (!response.ok) throw new Error(`HTTP ${response.status}`); setSnapshot(await response.json()); }
catch { message('No hosted snapshot could be loaded. You can still import a results JSON file.'); }

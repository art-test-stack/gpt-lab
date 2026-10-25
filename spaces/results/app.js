import {COLORS, finite, validateSnapshot, points, last, extrema, evaluationSeconds, flatten, demoSnapshot} from './data.mjs';

const $ = id => document.getElementById(id);
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const fmt = (n, digits = 3) => finite(n) ? n.toLocaleString('en-US', {maximumFractionDigits: digits}) : '—';
const compact = n => new Intl.NumberFormat('en-US', {notation:'compact', maximumFractionDigits:1}).format(n);
const duration = n => !finite(n) ? '—' : n >= 3600 ? `${fmt(n / 3600, 1)} h` : n >= 60 ? `${fmt(n / 60, 1)} min` : `${fmt(n, 1)} s`;
const title = run => String(run.label || run.id);
let snapshot = {schema_version: 1, runs: []}, realSnapshot = snapshot, selected = new Set(), synthetic = false, tab = 'overview';
const chosen = () => snapshot.runs.filter(run => selected.has(run.id));
const color = run => COLORS[snapshot.runs.indexOf(run) % COLORS.length];

function message(text, error = false) { $('message').textContent = text; $('message').classList.toggle('error', error); $('message').hidden = !text; }
function setSnapshot(data, isDemo = false) {
  validateSnapshot(data); isDemo = isDemo || data.synthetic === true; snapshot = data; synthetic = isDemo;
  if (!isDemo) realSnapshot = data;
  selected = new Set(data.runs.map(run => run.id)); render();
}
function runList() {
  const query = $('search').value.toLowerCase(); $('run-list').replaceChildren();
  for (const run of snapshot.runs.filter(run => `${title(run)} ${run.group || ''} ${run.status || ''}`.toLowerCase().includes(query))) {
    const label = document.createElement('label'); label.className = 'run-item';
    const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.checked = selected.has(run.id);
    checkbox.addEventListener('change', () => { checkbox.checked ? selected.add(run.id) : selected.delete(run.id); render(); });
    const dot = document.createElement('span'); dot.className = 'dot'; dot.style.background = color(run);
    const text = document.createElement('span'); text.append(document.createTextNode(title(run)));
    const small = document.createElement('small'); small.textContent = [run.group, run.status].filter(Boolean).join(' · ') || 'Status not supplied'; text.append(small);
    label.append(checkbox, dot, text); $('run-list').append(label);
  }
  $('run-count').textContent = snapshot.runs.length;
}
function chart(id, metric, {factor = 1, smooth = false} = {}) {
  const host = $(id), axis = $('axis').value, smoothing = smooth ? Number($('smoothing').value) : 0;
  const series = chosen().map(run => ({run, data: points(run, metric, axis, smoothing).map(([x,y,raw]) => [x,y * factor, raw * factor])})).filter(s => s.data.length);
  if (!series.length) { host.innerHTML = '<div class="no-data">No recorded values for this metric and axis.<br>Try training step, or select another run.</div>'; return; }
  const W = 620, H = 275, L = 56, R = 18, T = 15, B = 38;
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const s of series) for (const [x,y] of s.data) { xmin = Math.min(xmin,x); xmax = Math.max(xmax,x); ymin = Math.min(ymin,y); ymax = Math.max(ymax,y); }
  if (xmin === xmax) { xmin -= 0.5; xmax += 0.5; }
  const pad = ymax === ymin ? Math.max(0.05, Math.abs(ymin) * 0.05) : (ymax - ymin) * 0.1; ymin -= pad; ymax += pad;
  const X = x => L + (x-xmin)/(xmax-xmin)*(W-L-R), Y = y => T + (ymax-y)/(ymax-ymin)*(H-T-B);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(metric)} by ${esc(axis)}"><title>${esc(metric)}; ${esc(axis)}. Recorded data points have hover labels.</title>`;
  for (let i=0; i<=4; i++) {
    const y = ymin+(ymax-ymin)*i/4, x = xmin+(xmax-xmin)*i/4;
    svg += `<line x1="${L}" y1="${Y(y)}" x2="${W-R}" y2="${Y(y)}" stroke="#edf0f3"/><text x="${L-12}" y="${Y(y)+4}" fill="#91a0ab" font-size="10" text-anchor="end">${esc(Math.abs(y) >= 1000 ? compact(y) : fmt(y, 3))}</text><text x="${X(x)}" y="${H-15}" fill="#91a0ab" font-size="10" text-anchor="middle">${esc(Math.abs(x) >= 1000 ? compact(x) : fmt(x, 2))}</text>`;
  }
  for (const {run, data} of series) {
    const path = data.map(([x,y],i) => `${i ? 'L' : 'M'}${X(x).toFixed(2)},${Y(y).toFixed(2)}`).join(' ');
    svg += `<path d="${path}" fill="none" stroke="${color(run)}" stroke-width="2.3" stroke-linejoin="round"/>`;
    // Limit hover targets only; all recorded points remain in the line and export.
    const stride = Math.max(1, Math.ceil(data.length/300));
    data.forEach(([x,y,raw],i) => { if (i % stride === 0 || i === data.length-1) svg += `<circle cx="${X(x)}" cy="${Y(y)}" r="${data.length === 1 ? 4 : 3}" fill="${color(run)}" fill-opacity="${data.length === 1 ? 1 : 0.05}"><title>${esc(title(run))} · ${esc(axis)}: ${fmt(x)} · ${esc(metric)}: ${fmt(y,5)}${smoothing ? ` · raw: ${fmt(raw,5)}` : ''}</title></circle>`; });
  }
  host.innerHTML = svg + '</svg><div class="chart-legend">' + series.map(s => `<span><i style="background:${color(s.run)}"></i>${esc(title(s.run))}</span>`).join('') + '</div>';
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
    return {cells:[`<span style="color:${color(run)}">●</span> ${esc(title(run))}${source}`, `<span class="pill">${esc(run.status || 'unknown')}</span>`, esc(fmt(last(run,'train/total_tokens'),0)), esc(fmt(last(run,'eval/bpb'),4)), esc(fmt(last(run,'core/core'),4)), esc(fmt(last(run,'core/max_per_task'),0)), esc(duration(last(run,'train/total_training_time'))), esc(duration(last(run,'_runtime'))), esc(duration(evaluationSeconds(run)))]};
  });
  table('recap-table', ['Run','Status','Training tokens','Latest BPB','Latest CORE','CORE cap/task','Training time','Logged elapsed','Recorded eval'], rows);
}
function tasks() {
  const runs = chosen(), keys = [...new Set(runs.flatMap(run => run.history.flatMap(row => Object.keys(row).filter(k => /^core\/.+\/accuracy$/.test(k)))))].sort();
  table('tasks-table', ['Task', ...runs.map(title)], keys.map(key => ({cells:[esc(key.slice(5,-9)), ...runs.map(run => esc(fmt(last(run,key),4)))]})));
}
function configs() {
  const runs = chosen(), configs = runs.map(run => flatten(run.config || {})), keys = [...new Set(configs.flatMap(c => Object.keys(c)))].sort();
  const stringify = v => v === undefined ? '—' : JSON.stringify(v);
  const rows = keys.map(key => { const values = configs.map(c => stringify(c[key])); return {different: new Set(values).size > 1, cells:[`<span class="config-key">${esc(key)}</span>`, ...values.map(esc)]}; }).filter(row => !$('differences').checked || row.different);
  table('config-table', ['Setting', ...runs.map(title)], rows);
}
function render() {
  runList(); $('empty').hidden = snapshot.runs.length > 0; $('workspace').hidden = !snapshot.runs.length;
  $('demo-banner').hidden = !synthetic; $('download').disabled = !selected.size;
  $('snapshot-info').textContent = synthetic ? 'Synthetic interface preview' : `${snapshot.runs.length} runs · ${snapshot.exported_at ? `Snapshot ${snapshot.exported_at}` : 'Imported snapshot'}`;
  summary(); configs(); tasks();
  if (tab === 'overview') { chart('loss-chart','train/raw_loss',{smooth:true}); chart('bpb-chart','eval/bpb'); chart('throughput-chart','train/tokens_per_sec'); chart('lr-chart','lrm'); }
  if (tab === 'evaluation') { chart('core-chart','core/core'); chart('eval-chart','core/step_time_ms',{factor:0.001}); }
}
for (const button of document.querySelectorAll('[data-tab]')) button.addEventListener('click', () => {
  tab = button.dataset.tab;
  for (const node of document.querySelectorAll('[data-tab]')) node.classList.toggle('active',node === button);
  for (const node of document.querySelectorAll('.tab-panel')) node.hidden = node.id !== tab;
  const names = {overview:'Experiment overview', evaluation:'Evaluation results', configuration:'Training configuration'};
  $('page-title').replaceChildren(document.createTextNode(names[tab]),Object.assign(document.createElement('span'),{textContent:'.'}));
  $('selection-label').textContent = tab === 'overview' ? 'Learning curves' : tab === 'evaluation' ? 'Evaluation checkpoints' : 'Compare every setting';
  $('smoothing-label').hidden = tab !== 'overview'; $('axis').parentElement.hidden = tab === 'configuration'; render();
});
for (const id of ['axis','smoothing','differences']) $(id).addEventListener('change', render);
$('import-button').addEventListener('click', () => $('import').click());
$('search').addEventListener('input', runList);
$('select-all').addEventListener('click', () => { const query=$('search').value.toLowerCase(); for (const run of snapshot.runs) if (`${title(run)} ${run.group||''} ${run.status||''}`.toLowerCase().includes(query)) selected.add(run.id); render(); });
for (const id of ['demo','empty-demo']) $(id).addEventListener('click', () => { message(''); setSnapshot(demoSnapshot(),true); });
$('clear-demo').addEventListener('click', () => setSnapshot(realSnapshot));
$('import').addEventListener('change', async event => {
  const file = event.target.files[0]; if (!file) return;
  try { if (file.size > 25*1024*1024) throw new Error('Please import a snapshot smaller than 25 MiB.'); const data = JSON.parse(await file.text()); setSnapshot(data); message(`Loaded ${data.runs.length} runs from ${file.name}.`); }
  catch (error) { message(`Could not import results: ${error.message}`,true); }
  event.target.value = '';
});
$('download').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify({...snapshot, runs:chosen(), ...(synthetic ? {synthetic:true} : {})},null,2)],{type:'application/json'});
  const url = URL.createObjectURL(blob), a=document.createElement('a'); a.href=url; a.download=synthetic ? 'synthetic-preview.json' : 'gpt-lab-results.json'; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
});
render();
try { const response = await fetch('./results.json', {cache:'no-cache'}); if (!response.ok) throw new Error(`HTTP ${response.status}`); setSnapshot(await response.json()); }
catch { message('No hosted snapshot could be loaded. You can still import a results JSON file.'); }

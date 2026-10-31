import {validateArchive, evidencePath, filterArchive, comparisonRows, reportMeasurements, primaryReport, reportSetup, METRICS} from './evidence_data.mjs';

const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const number = value => typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString('en-US', {maximumFractionDigits: 2}) : '—';
const bytes = value => value >= 1024 ** 3 ? `${number(value / 1024 ** 3)} GiB` : value >= 1024 ** 2 ? `${number(value / 1024 ** 2)} MiB` : value >= 1024 ? `${number(value / 1024)} KiB` : `${value || 0} B`;
const date = value => value ? String(value).slice(0, 16).replace('T', ' ') + ' UTC' : 'Date not recorded';
const descriptions = {
  communication_buckets: 'Baseline, bucketed, pipelined and backward communication. Traffic, padding and full-update timing.',
  cuda_graphs: 'Compiled execution versus explicit forward/backward capture. Correctness, throughput and memory.',
  fp8_training: 'BF16 and FP8 execution, kernel dispatch and full-update profiles.',
  ffn_ablation: 'SwiGLU, residual composition and squared-ReLU variants. Architecture and runtime measurements.',
  h100_input_pipeline: 'Production loading versus CPU/GPU caching. Paired timings and memory.',
  optimizer_comparison: 'Muon, NorMuon and available ANVIL variants. Recorded recipes and learning evaluations.',
  other: 'Additional diagnostics and training runs. Original names and evidence are retained.',
};
const failed = run => ['failed','crashed','error'].includes(run.status);
const state = run => `<span class="pill ${failed(run) ? 'status-failed' : run.status === 'unknown' ? 'status-unknown' : ''}">${esc(run.status || 'unknown')}</span>`;
let catalog = null, selected = null, artifactFilter = 'all';
const reports = new Map();
const titles = () => new Map(catalog.experiments.map(e => [e.id, e.title]));

function cards() {
  const total = catalog.runs.reduce((n, r) => n + r.artifacts.length, 0);
  $('archive-summary').innerHTML = `<span><b>${catalog.experiments.length}</b> experiment families</span><span><b>${catalog.runs.length}</b> runs</span><span><b>${total}</b> evidence files</span><span><b>${catalog.runs.filter(failed).length}</b> failed runs retained</span>`;
  $('archive-experiments').innerHTML = catalog.experiments.map((experiment, index) => {
    const runs = catalog.runs.filter(r => r.experiment === experiment.id);
    const traces = runs.reduce((n,r) => n + r.artifacts.filter(a => a.kind === 'trace').length, 0);
    return `<button class="experiment-card ${$('archive-experiment').value === experiment.id ? 'selected' : ''}" data-experiment="${esc(experiment.id)}" aria-pressed="${$('archive-experiment').value === experiment.id}"><span class="eyebrow">EXPERIMENT ${String(index + 1).padStart(2, '0')}</span><h2>${esc(experiment.title)}</h2><p>${esc(descriptions[experiment.id] || 'Saved measurements, configurations and reports.')}</p><span class="experiment-card-meta">${runs.length} runs · ${traces} traces <b>Explore ↗</b></span></button>`;
  }).join('');
  for (const button of $('archive-experiments').querySelectorAll('[data-experiment]')) button.addEventListener('click', () => {
    $('archive-experiment').value = button.dataset.experiment;
    $('archive-status').value = ''; $('archive-search').value = ''; selected = null; render();
    $('archive-runs').scrollIntoView({behavior:'smooth', block:'start'});
  });
}

function render() {
  if (!catalog) return;
  const runs = filterArchive(catalog, {experiment: $('archive-experiment').value, status: $('archive-status').value,
    query: $('archive-search').value, order: $('archive-order').value});
  if (!runs.some(r => r.id === selected)) selected = runs[0]?.id || null;
  cards();
  $('archive-count').textContent = `${runs.length} runs · ${runs.reduce((sum, run) => sum + run.artifacts.length, 0)} evidence files`;
  const names = titles();
  $('archive-runs').innerHTML = runs.map(run => `<button class="archive-run-button ${run.id === selected ? 'selected' : ''}" data-run="${esc(run.id)}" aria-pressed="${run.id === selected}"><span class="eyebrow">${esc(names.get(run.experiment) || run.experiment)}</span><strong>${esc(run.name)}</strong><span>${state(run)} <small>${run.artifacts.length} files · ${date(run.started_at)}</small></span></button>`).join('') || '<p class="panel archive-no-results">No runs match these filters.</p>';
  for (const button of $('archive-runs').querySelectorAll('[data-run]')) button.addEventListener('click', () => { selected = button.dataset.run; artifactFilter = 'all'; render(); });
  detail(runs.find(r => r.id === selected));
}

function artifactLink(run, source) {
  const artifact = run.artifacts.find(a => a.name === source || a.name === `${run.source?.directory || run.name}/${source}`);
  return artifact ? `<a href="${esc(evidencePath(artifact.path))}" target="_blank" rel="noopener noreferrer">${esc(source)} ↗</a>` : esc(source);
}

function comparison(run, extra) {
  const rows = comparisonRows(run, extra);
  if (!rows.length) return '<p class="archive-empty-metrics">No recognized runtime summary. Reports and logs below retain the original results.</p>';
  const columns = METRICS.filter(m => rows.some(r => r.metrics[m.id] !== undefined));
  const max = Math.max(...rows.map(r => r.metrics.throughput || 0));
  const baseline = rows.find(r => r.variant === 'disabled') || rows.find(r => r.variant === 'full-baseline');
  const candidate = rows.find(r => r.variant === 'forward_backward') || rows.find(r => r.variant === 'full-pipeline');
  let observation = '';
  if (baseline?.metrics.throughput > 0 && candidate?.metrics.throughput > 0) {
    const delta = 100 * (candidate.metrics.throughput / baseline.metrics.throughput - 1);
    const memory = Number.isFinite(baseline.metrics.reserved) && Number.isFinite(candidate.metrics.reserved)
      ? ` · ${number(candidate.metrics.reserved - baseline.metrics.reserved)} GiB change in peak reserved / GPU` : '';
    observation = `<div class="archive-observation"><b>Observed difference</b><p>${esc(candidate.variant)} vs ${esc(baseline.variant)}: ${delta > 0 ? '+' : ''}${number(delta)}% throughput${memory}.</p><small>Difference in this recorded pair; no confidence interval or learning-quality conclusion.</small></div>`;
  }
  return `${observation}<h3>Recorded comparison</h3><p class="small">Values belong to this run. Screen, full, profile and CPU-only legs retain their labels; timings may use different protocols. Memory is the maximum recorded per-GPU peak.</p><div class="table-wrap"><table><thead><tr><th>Variant / scope</th>${columns.map(m => `<th>${m.title}${m.unit ? `<br>${m.unit}` : ''}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr><td><strong>${esc(row.variant)}</strong><small>${artifactLink(run,row.source)}</small></td>${columns.map(m => `<td class="value" title="${esc(row.scopes[m.id] || 'Not recorded')}">${number(row.metrics[m.id])}${m.id === 'throughput' && max && row.metrics.throughput ? `<span class="metric-bar" style="width:${Math.max(1, 100 * row.metrics.throughput / max)}%"></span>` : ''}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function detail(run) {
  const host = $('archive-detail');
  if (!run) { host.innerHTML = '<p class="archive-no-results">Select an experiment or change the filters to inspect a run.</p>'; return; }
  const extra = reports.get(run.id)?.measurements || [];
  host.innerHTML = `<div class="archive-detail-heading"><div class="eyebrow">${esc(titles().get(run.experiment))}</div><h2>${esc(run.name)}</h2><p>${state(run)} <span class="small">${date(run.started_at)}</span></p><p class="archive-commits">${(run.commits || []).map(sha => /^[a-f0-9]{40}$/i.test(sha) ? `<a href="https://github.com/art-test-stack/gpt-lab/commit/${sha}" target="_blank" rel="noopener noreferrer">${sha.slice(0, 10)} ↗</a>` : '').join(' · ') || 'Source commit not recorded'}</p></div>
    <div id="run-setup" class="archive-setup"></div>
    <div id="run-comparison">${comparison(run, extra)}</div>
    <p id="run-report-status" class="small" role="status"></p>
    <section class="archive-files"><h3>Evidence</h3><p class="small">Read reports and logs, inspect figures, or download traces for <a href="https://ui.perfetto.dev/" target="_blank" rel="noopener noreferrer">Perfetto</a>. Files load only when opened.</p><label>File type <select id="archive-file-type"><option value="all">All evidence</option>${[...new Set(run.artifacts.map(a => a.kind))].map(kind => `<option value="${esc(kind)}">${esc(kind)}</option>`).join('')}</select></label><div id="archive-files-list"></div></section>
    <details class="archive-raw"><summary>All recorded fields and provenance</summary><p class="small">Original field names, units and contexts are retained.</p><pre tabindex="0">${esc(JSON.stringify({source:run.source, measurements: run.measurements, artifacts:run.artifacts}, null, 2))}</pre></details>`;
  const kinds = [...new Set(run.artifacts.map(a => a.kind))];
  if (!kinds.includes(artifactFilter)) artifactFilter = 'all';
  $('archive-file-type').value = artifactFilter;
  $('archive-file-type').addEventListener('change', () => { artifactFilter = $('archive-file-type').value; files(run); });
  files(run);
  const setup = () => {
    $('run-setup').innerHTML = (reports.get(run.id)?.setup || []).map(item => `<span><small>${esc(item.label)}</small><b>${esc(item.value)}</b></span>`).join('');
  };
  setup();
  const report = primaryReport(run);
  if (report && !reports.has(run.id)) {
    reports.set(run.id, {loading:true});
    $('run-report-status').textContent = 'Reading the saved summary report…';
    fetch(evidencePath(report.path)).then(response => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    }).then(data => {
      reports.set(run.id, {measurements:reportMeasurements(data, report.name.split('/').pop()), setup:reportSetup(data)});
      if (selected === run.id) { setup(); $('run-comparison').innerHTML = comparison(run, reports.get(run.id).measurements); $('run-report-status').textContent = 'Summary loaded from the saved report.'; }
    }).catch(() => { reports.set(run.id, {measurements:[]}); if (selected === run.id) $('run-report-status').textContent = 'Summary report could not be loaded. Indexed measurements and evidence links remain available.'; });
  }
}

function files(run) {
  const artifacts = run.artifacts.filter(a => artifactFilter === 'all' || a.kind === artifactFilter);
  $('archive-files-list').innerHTML = `<ul class="archive-artifacts">${artifacts.map(item => `<li><span class="pill">${esc(item.kind)}</span><a href="${esc(evidencePath(item.path))}" target="_blank" rel="noopener noreferrer" ${item.kind === 'trace' ? 'download' : ''}>${esc(item.name.startsWith(run.name + '/') ? item.name.slice(run.name.length + 1) : item.name)}</a><small>${bytes(item.bytes)}${item.transformed ? ' · normalized/redacted' : ''}${item.parse_error ? ' · incomplete JSON' : ''}</small></li>`).join('')}</ul>`;
}

export async function loadArchive() {
  try {
    const response = await fetch('./experiments.json', {cache:'no-cache'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    catalog = validateArchive(await response.json());
    $('archive-experiment').replaceChildren(new Option('All experiments', ''), ...catalog.experiments.map(item => new Option(item.title, item.id)));
    $('archive-status').replaceChildren(new Option('All statuses', ''), ...[...new Set(catalog.runs.map(run => run.status || 'unknown'))].sort().map(status => new Option(status, status)));
    $('archive-status-message').textContent = `Collected ${date(catalog.exported_at)}. Failed and inconclusive runs are retained. No model weights are included.`;
    if (catalog.errors?.length) $('archive-status-message').textContent += ` ${catalog.errors.length} collection errors are recorded in the index.`;
    $('archive-download').disabled = false;
    render();
  } catch (error) {
    $('archive-status-message').textContent = `Experiment archive unavailable (${error.message}). Training views remain accessible from the navigation.`;
    $('archive-download').disabled = true;
  }
}
for (const id of ['archive-experiment','archive-status','archive-order']) $(id).addEventListener('change', render);
$('archive-search').addEventListener('input', render);
$('archive-download').addEventListener('click', () => {
  if (!catalog) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(catalog,null,2)],{type:'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = 'gpt-lab-experiment-archive.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
loadArchive();

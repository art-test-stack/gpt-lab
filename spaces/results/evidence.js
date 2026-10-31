import {validateArchive, evidencePath, filterArchive} from './evidence_data.mjs';

const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const number = value => typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString('en-US', {maximumFractionDigits: 5}) : '—';
const bytes = value => value >= 1024 ** 2 ? `${number(value / 1024 ** 2)} MiB` : value >= 1024 ? `${number(value / 1024)} KiB` : `${value} B`;
let catalog = null;

function render() {
  if (!catalog) return;
  const runs = filterArchive(catalog, {experiment: $('archive-experiment').value, status: $('archive-status').value,
    query: $('archive-search').value, order: $('archive-order').value});
  $('archive-count').textContent = `${runs.length} runs · ${runs.reduce((sum, run) => sum + run.artifacts.length, 0)} evidence files`;
  const groups = new Map();
  for (const run of runs) {
    if (!groups.has(run.experiment)) groups.set(run.experiment, []);
    groups.get(run.experiment).push(run);
  }
  const titles = new Map(catalog.experiments.map(item => [item.id, item.title]));
  $('archive-runs').innerHTML = [...groups].map(([experiment, entries]) => `<section class="archive-group"><h2>${esc(titles.get(experiment) || experiment)}</h2>${entries.map(run => {
    const metrics = (run.measurements || []).flatMap(item => Object.entries(item.values).map(([key, value]) =>
      `<tr><td>${esc(item.context)}</td><td>${esc(key)}</td><td>${number(value)}</td><td>${esc(item.source)}</td></tr>`));
    return `<article class="panel archive-run"><div class="archive-run-head"><h3>${esc(run.name)}</h3><span class="pill">${esc(run.status || 'unknown')}</span></div>
      <p class="small">${esc(run.started_at || 'Date not recorded')} · ${esc(run.source.environment)}/${esc(run.source.volume)} · ${run.artifacts.length} files</p>
      <p class="archive-commits">${(run.commits || []).map(sha => /^[a-f0-9]{40}$/i.test(sha) ? `<a href="https://github.com/art-test-stack/gpt-lab/commit/${sha}" target="_blank" rel="noopener noreferrer">${sha.slice(0, 10)}</a>` : '').join(' · ')}</p>
      ${metrics.length ? `<details><summary>Recorded measurements (${metrics.length})</summary><p class="small">Field names retain their original units and scope. Different hardware, warmup, and protocols are separate experiments.</p><div class="table-wrap"><table><thead><tr><th>Variant / context</th><th>Recorded field</th><th>Value</th><th>Source</th></tr></thead><tbody>${metrics.join('')}</tbody></table></div></details>` : '<p class="small">No recognized scalar summary; inspect the original reports below.</p>'}
      <details><summary>Reports, configurations, logs and traces (${run.artifacts.length})</summary><ul class="archive-artifacts">${run.artifacts.map(item => `<li><span class="pill">${esc(item.kind)}</span> <a href="${esc(evidencePath(item.path))}" download target="_blank" rel="noopener noreferrer">${esc(item.name)}</a><small>${bytes(item.bytes)}${item.transformed ? ' · normalized/redacted' : ''}${item.parse_error ? ' · incomplete/invalid JSON retained' : ''}</small></li>`).join('')}</ul></details>
      <details><summary>Provenance and checksums</summary><pre tabindex="0">${esc(JSON.stringify({id: run.id, source: run.source, commits: run.commits, artifacts: run.artifacts.map(({name, source_sha256, sha256, transformed}) => ({name, source_sha256, sha256, transformed}))}, null, 2))}</pre></details>
    </article>`;
  }).join('')}</section>`).join('') || '<p class="panel">No archived runs match these filters.</p>';
}

export async function loadArchive() {
  try {
    const response = await fetch('./experiments.json', {cache: 'no-cache'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    catalog = validateArchive(await response.json());
    $('archive-experiment').replaceChildren(new Option('All experiments', ''), ...catalog.experiments.map(item => new Option(item.title, item.id)));
    $('archive-status').replaceChildren(new Option('All statuses', ''), ...[...new Set(catalog.runs.map(run => run.status || 'unknown'))].sort().map(status => new Option(status, status)));
    $('archive-status-message').textContent = `Collected ${catalog.exported_at}. Failed and inconclusive runs are retained. No model weights are included.`;
    if (catalog.errors?.length) $('archive-status-message').textContent += ` ${catalog.errors.length} collection errors are recorded in the index.`;
    render();
  } catch (error) {
    $('archive-status-message').textContent = 'No Modal archive has been published yet. Training charts remain available. Run the Modal results collector to add reports and traces.';
  }
}

for (const id of ['archive-experiment', 'archive-status', 'archive-order']) $(id).addEventListener('change', render);
$('archive-search').addEventListener('input', render);
$('archive-download').addEventListener('click', () => {
  if (!catalog) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(catalog, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = 'gpt-lab-experiment-archive.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
loadArchive();

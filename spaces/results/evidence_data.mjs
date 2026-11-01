// Archive evidence is separate from training snapshots: no fabricated curves.
export function validateArchive(data) {
  if (!data || data.schema_version !== 1 || !Array.isArray(data.runs) || !Array.isArray(data.experiments)) throw new Error('Invalid experiment archive.');
  const ids = new Set();
  for (const run of data.runs) {
    if (!run || typeof run.id !== 'string' || ids.has(run.id) || typeof run.experiment !== 'string' || !Array.isArray(run.artifacts)) throw new Error('Invalid archive run.');
    ids.add(run.id);
    for (const artifact of run.artifacts) if (!evidencePath(artifact.path)) throw new Error('Invalid evidence path.');
  }
  return data;
}

export function evidencePath(path) {
  if (typeof path !== 'string' || !path.startsWith('archive/') || /[\\\x00-\x1f]/.test(path)) return null;
  if (path.split('/').some(part => !part || part === '.' || part === '..')) return null;
  return './' + path.split('/').map(encodeURIComponent).join('/');
}

export function filterArchive(data, {experiment = '', status = '', query = '', order = 'newest'} = {}) {
  const needle = query.toLowerCase();
  return data.runs.filter(run => (!experiment || run.experiment === experiment) && (!status || run.status === status)
    && `${run.name} ${run.experiment} ${(run.commits || []).join(' ')} ${run.status}`.toLowerCase().includes(needle))
    .sort((a, b) => a.experiment.localeCompare(b.experiment) || (order === 'name' ? a.name.localeCompare(b.name)
      : String(b.started_at || '').localeCompare(String(a.started_at || '')) || a.name.localeCompare(b.name)));
}

// Read only small report objects. Trace events and per-update arrays are never walked.
export function reportMeasurements(value, source, context = '', depth = 0) {
  if (!value || typeof value !== 'object' || depth > 9) return [];
  if (Array.isArray(value)) return [];
  const values = Object.fromEntries(Object.entries(value).filter(([, v]) => typeof v === 'number' && Number.isFinite(v)));
  const result = Object.keys(values).length ? [{source, context: context || 'summary', values}] : [];
  for (const [key, child] of Object.entries(value)) {
    if (child && typeof child === 'object' && !Array.isArray(child)) result.push(...reportMeasurements(child, source, [context, key].filter(Boolean).join('/'), depth + 1));
  }
  return result;
}

export const METRICS = [
  {id: 'throughput', title: 'Throughput', unit: 'tok/s', keys: ['steady_global_tokens_per_second', 'steady_train_loop_tokens_per_second', 'steady_tokens_per_second', 'tokens_per_second', 'tokens_per_sec']},
  {id: 'update', title: 'Update time', unit: 'ms', keys: ['steady_mean_train_step_ms', 'mean_update_ms', 'mean_step_ms', 'median_update_ms']},
  {id: 'reserved', title: 'Peak reserved / GPU', unit: 'GiB', keys: ['peak_reserved_bytes_after_warmup', 'peak_reserved_bytes'], factor: 1 / 1024 ** 3},
  {id: 'allocated', title: 'Peak allocated / GPU', unit: 'GiB', keys: ['peak_allocated_bytes_after_warmup', 'peak_allocated_bytes'], factor: 1 / 1024 ** 3},
  {id: 'startup', title: 'Process including startup', unit: 's', keys: ['process_startup_inclusive_seconds']},
  {id: 'updates', title: 'Measured updates', unit: '', keys: ['steady_updates', 'measured_updates']},
];

export function comparisonRows(run, extra = []) {
  const all = [...extra, ...(run.measurements || [])];
  // A paired report already contains the individual leg reports. Use one source,
  // rather than displaying the same measurement again for every copy and rank.
  const paired = ['comparison.json', 'manifest.json', 'report.json'].find(source =>
    all.some(m => m.source === source && /^(results|candidates|legs)\/[^/]+/.test(m.context)));
  const groups = new Map();
  for (const m of all) {
    let variant, scope;
    if (paired) {
      if (m.source !== paired) continue;
      const match = /^(?:results|candidates|legs)\/([^/]+)(?:\/(.*))?$/.exec(m.context);
      if (!match) continue;
      [, variant, scope = 'summary'] = match;
    } else {
      if (/rank[_-]?\d+\.json$/.test(m.source) || /(?:^|\/)\d+(?:\/|$)/.test(m.context)) continue;
      variant = m.source.replace(/\/(?:results?|summary|communication-results)\.json$/, '').replace(/\.json$/, '');
      scope = m.context;
    }
    const key = `${m.source}:${variant}`;
    if (!groups.has(key)) groups.set(key, {variant, source: m.source, metrics: {}, scopes: {}});
    const row = groups.get(key);
    for (const metric of METRICS) {
      const field = metric.keys.find(k => Number.isFinite(m.values[k]));
      let value = field ? m.values[field] * (metric.factor || 1) : null;
      // Timing summaries differ by producer. Only treat full-update contexts as updates.
      if (metric.id === 'update' && value === null && /^(timing|update|steady_train_step_time_ms)$/.test(scope)) value = m.values.mean_ms ?? m.values.median_ms ?? null;
      if (value === null) continue;
      if (['reserved', 'allocated'].includes(metric.id)) {
        // Use the maximum recorded per-rank peak, never the sum of GPU memory.
        if (row.metrics[metric.id] !== undefined && row.metrics[metric.id] >= value) continue;
      } else if (row.metrics[metric.id] !== undefined) continue;
      row.metrics[metric.id] = value;
      row.scopes[metric.id] = `${m.context} / ${field || 'mean_ms or median_ms'}`;
    }
  }
  return [...groups.values()].filter(r => Object.keys(r.metrics).length);
}

export function primaryReport(run) {
  return ['comparison.json', 'report.json', 'manifest.json', 'summary.json'].map(name =>
    run.artifacts.find(a => a.name === name || a.name === `${run.source?.directory || run.name}/${name}`))
    .find(a => a && a.bytes < 8 * 1024 ** 2 && a.kind !== 'trace' && !a.parse_error);
}

export function reportSetup(data) {
  const labels = {device_name:'GPU', world_size:'GPUs', dtype:'Precision', device_batch_size:'Microbatch / GPU', total_batch_size:'Global batch', max_seq_len:'Sequence', max_seq_length:'Sequence', sequence_length:'Sequence', optimizer:'Optimizer', optimizer_name:'Optimizer', measured_updates:'Measured updates', warmup_updates:'Excluded warmup updates'};
  const facts = new Map();
  function walk(value, depth = 0) {
    if (!value || typeof value !== 'object' || Array.isArray(value) || depth > 4) return;
    for (const [key, child] of Object.entries(value)) {
      if (labels[key] && ['string','number'].includes(typeof child)) {
        if (!facts.has(labels[key])) facts.set(labels[key], new Set());
        facts.get(labels[key]).add(String(child));
      }
      if (!/memory|steps|timing|ranks|trace/.test(key)) walk(child, depth + 1);
    }
  }
  walk(data);
  return [...facts].map(([label, values]) => ({label, value:[...values].join(' / ')}));
}

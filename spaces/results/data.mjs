// Pure data helpers. Preserve sparse histories; never invent missing counters.
export const COLORS = ['#e47745', '#4c8b80', '#697fb6', '#b08bca', '#b79c4c', '#819ba7'];
export const finite = value => typeof value === 'number' && Number.isFinite(value);
export const metricKey = key => /^(train|eval|core|optim)\//.test(key) || ['lrm', 'muon_momentum', 'weight_decay', '_runtime', 'step'].includes(key);

export function validateSnapshot(data) {
  if (!data || data.schema_version !== 1 || !Array.isArray(data.runs)) throw new Error('Expected schema_version: 1 and a runs array.');
  if (data.runs.length > 100) throw new Error('Please load at most 100 runs at once.');
  const ids = new Set();
  for (const run of data.runs) {
    if (!run || typeof run.id !== 'string' || !run.id || ids.has(run.id)) throw new Error('Every run needs a unique, non-empty string id.');
    ids.add(run.id);
    if (!Array.isArray(run.history) || run.history.length > 100000) throw new Error('Each run needs a history array with at most 100,000 rows.');
    if (!run.history.every(row => row && typeof row === 'object' && !Array.isArray(row) && finite(row.step))) throw new Error('Every history row needs a numeric step.');
    if (run.config != null && (typeof run.config !== 'object' || Array.isArray(run.config))) throw new Error('Run config must be an object.');
  }
  return data;
}

export function points(run, key, axis = 'step', smoothing = 0) {
  const scale = axis === '_runtime' || axis === 'train/total_training_time' ? 3600 : 1;
  const rows = run.history.filter(row => finite(row[key]) && finite(row[axis])).map(row => [row[axis] / scale, row[key]]).sort((a, b) => a[0] - b[0]);
  let previous;
  return rows.map(([x, raw]) => {
    previous = previous === undefined ? raw : smoothing * previous + (1 - smoothing) * raw;
    return [x, previous, raw];
  });
}

export function last(run, key) {
  const rows = run.history.filter(row => finite(row[key])).sort((a, b) => a.step - b.step);
  return rows.length ? rows.at(-1)[key] : null;
}

export function extrema(run, key, direction = 'min') {
  const values = run.history.map(row => row[key]).filter(finite);
  return values.length ? values.reduce((a, b) => direction === 'min' ? Math.min(a, b) : Math.max(a, b)) : null;
}

export function evaluationSeconds(run) {
  // Deduplicate repeated records of the same evaluation at a logged step.
  const durations = new Map();
  for (const row of run.history) for (const key of ['core/step_time_ms', 'eval/step_time_ms']) {
    if (finite(row[key]) && row[key] >= 0) durations.set(`${row.step}:${key}`, row[key] / 1000);
  }
  return durations.size ? [...durations.values()].reduce((a, b) => a + b, 0) : null;
}

export function flatten(value, prefix = '', result = Object.create(null), depth = 0) {
  if (depth > 12) { result[prefix] = '[nested value]'; return result; }
  if (value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length) {
    for (const [key, child] of Object.entries(value)) flatten(child, prefix ? `${prefix}.${key}` : key, result, depth + 1);
  } else result[prefix] = value;
  return result;
}

export function demoSnapshot() {
  const runs = ['A', 'B'].map((letter, index) => ({
    id: `synthetic-${letter}`, label: `Example run ${letter}`, status: 'synthetic', group: 'Interface preview',
    config: {model: {layers: 8, width: 512, context: 1024}, optimizer: {name: 'illustrative', learning_rate: index ? 0.0015 : 0.001}, training: {seed: 42, global_batch_tokens: 65536}, evaluation: {examples_per_task: 100, protocol: 'synthetic-preview'}},
    history: Array.from({length: 51}, (_, i) => {
      const row = {step: i * 20, 'train/total_tokens': i * 20 * 65536, 'train/total_training_time': i * 55, _runtime: i * 70, 'train/raw_loss': 3.2 + 3.9 * Math.exp(-i / 12) + Math.sin(i * 1.7 + index) * 0.05, 'train/tokens_per_sec': 23500 + Math.sin(i + index) * 1500, lrm: Math.min(1, (i + 1) / 3) * Math.max(0.05, (51 - i) / 35)};
      if (i && i % 10 === 0) Object.assign(row, {'eval/bpb': 1.3 + Math.exp(-i / 15) * 0.8, 'eval/step_time_ms': 12000, 'core/core': 0.08 + i / 500, 'core/step_time_ms': 110000 + i * 100, 'core/max_per_task': 100, 'core/example-task-1/accuracy': 0.3 + i / 400, 'core/example-task-2/accuracy': 0.4 + i / 450});
      return row;
    }),
  }));
  return {schema_version: 1, synthetic: true, runs};
}

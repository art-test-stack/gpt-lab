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

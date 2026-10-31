import test from 'node:test';
import assert from 'node:assert/strict';
import {validateArchive, evidencePath, filterArchive} from '../../spaces/results/evidence_data.mjs';

test('evidence URLs stay inside the archive and encode file names', () => {
  assert.equal(evidencePath('archive/cuda/run/trace #1.json'), './archive/cuda/run/trace%20%231.json');
  for (const value of ['https://example.com/x', 'javascript:alert(1)', 'archive/../secret', 'archive/x\\y']) assert.equal(evidencePath(value), null);
});
test('archive retains failed and metric-free runs, filters without ranking scores', () => {
  const data = {schema_version: 1, experiments: [{id: 'cuda', title: 'CUDA'}], runs: [
    {id: 'a', name: 'baseline', experiment: 'cuda', status: 'completed', artifacts: []},
    {id: 'b', name: 'capture-failed', experiment: 'cuda', status: 'failed', artifacts: [{path: 'archive/cuda/b/console.log'}]},
  ]};
  assert.equal(validateArchive(data), data);
  assert.deepEqual(filterArchive(data, {status: 'failed'}).map(r => r.id), ['b']);
  assert.equal(filterArchive(data, {query: 'CAPTURE'}).length, 1);
  assert.equal(filterArchive(data, {experiment: 'fp8'}).length, 0);
  assert.throws(() => validateArchive({...data, runs: [data.runs[0], data.runs[0]]}));
});

import {comparisonRows, reportMeasurements, primaryReport} from '../../spaces/results/evidence_data.mjs';

test('paired capture report restores missing metrics and uses maximum per-GPU memory', () => {
  const run = {measurements: []};
  const measurements = reportMeasurements({results: {
    disabled: {steady_global_tokens_per_second: 1298919, steady_mean_train_step_ms: 403.3,
      memory_by_rank: {'0': {steady: {peak_reserved_bytes: 60 * 1024 ** 3}}, '1': {steady: {peak_reserved_bytes: 61 * 1024 ** 3}}}},
    forward_backward: {steady_global_tokens_per_second: 1292484, steady_mean_train_step_ms: 405.4},
  }}, 'comparison.json');
  const rows = comparisonRows(run, measurements);
  assert.equal(rows.length, 2);
  assert.equal(rows[0].metrics.throughput, 1298919);
  assert.equal(rows[0].metrics.reserved, 61);
  assert.equal(rows[1].metrics.update, 405.4);
});

test('paired manifest supersedes duplicate leg reports without mixing screen and full protocols', () => {
  const run = {measurements: [
    {source:'manifest.json', context:'candidates/screen-baseline', values:{steady_train_loop_tokens_per_second: 100}},
    {source:'manifest.json', context:'candidates/full-baseline', values:{steady_train_loop_tokens_per_second: 99}},
    {source:'full-baseline/communication-results.json', context:'summary', values:{steady_tokens_per_second: 120}},
  ]};
  const rows = comparisonRows(run);
  assert.deepEqual(rows.map(r => [r.variant, r.metrics.throughput]), [['screen-baseline', 100], ['full-baseline', 99]]);
});

test('small summary loading never selects profiler traces or large evidence', () => {
  assert.equal(primaryReport({name:'r', artifacts:[{name:'r/report.json', path:'archive/r/report.json', bytes:30, kind:'report/data'}]}).bytes, 30);
  assert.equal(primaryReport({name:'r', artifacts:[{name:'r/report.json', bytes:9*1024**2, kind:'trace'}]}), undefined);
  assert.equal(reportMeasurements({steps:[{tokens_per_second:99}]},'report.json').length, 0);
});

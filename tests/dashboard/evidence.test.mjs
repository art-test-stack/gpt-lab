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

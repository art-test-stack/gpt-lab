import test from 'node:test';
import assert from 'node:assert/strict';
import {validateSnapshot, points, last, evaluationSeconds, demoSnapshot} from '../../spaces/results/data.mjs';

test('token axes omit sparse evaluation rows instead of inventing token counts', () => {
  const run = {history:[{step:0,'train/total_tokens':100,'eval/bpb':0},{step:1,'eval/bpb':1.2}]};
  assert.deepEqual(points(run,'eval/bpb','train/total_tokens'),[[100,0,0]]);
  assert.equal(points(run,'eval/bpb','step').length,2);
});
test('time axes convert seconds to hours and loss EMA retains raw values', () => {
  const run = {history:[{step:0,'_runtime':3600,'train/raw_loss':4},{step:1,'_runtime':7200,'train/raw_loss':2}]};
  assert.deepEqual(points(run,'train/raw_loss','_runtime',0.5),[[1,4,4],[2,3,2]]);
});
test('latest metrics use training step, not import order, and preserve zero', () => {
  assert.equal(last({history:[{step:3,'core/core':0},{step:1,'core/core':0.1}]},'core/core'),0);
  assert.equal(last({history:[]},'core/core'),null);
});
test('recorded evaluation duration deduplicates repeated logs, not distinct passes', () => {
  assert.equal(evaluationSeconds({history:[{step:1,'core/step_time_ms':1000},{step:1,'core/step_time_ms':1000},{step:2,'core/step_time_ms':2000,'eval/step_time_ms':500}]}),3.5);
  assert.equal(evaluationSeconds({history:[]}),null);
});
test('snapshot validation rejects duplicate ids and malformed histories', () => {
  assert.throws(()=>validateSnapshot({schema_version:1,runs:[{id:'a',history:[]},{id:'a',history:[]}]}));
  assert.throws(()=>validateSnapshot({schema_version:1,runs:[{id:'a',history:[{step:'bad'}]}]}));
  assert.equal(validateSnapshot(demoSnapshot()).runs.length,2);
});

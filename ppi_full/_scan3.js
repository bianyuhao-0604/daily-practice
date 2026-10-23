const fs = require('fs');
const p = process.argv[2];
const lines = fs.readFileSync(p, 'utf8').split('\n').filter(Boolean);
const out = [];
for (const ln of lines) {
  let o; try { o = JSON.parse(ln); } catch { continue; }
  const d = o.data || {};
  if (!/surf_from_npz/.test(JSON.stringify(d))) continue;
  const label = (d.label || (d.toolCall && d.toolCall.label) || '').replace(/\s+/g, ' ').slice(0, 160);
  out.push({ t: o.type, seq: o.seq, time: o.time, jobId: d.jobId, status: d.status, label, detail: (d.detail || '').slice(0, 200) });
}
for (const r of out) console.log(JSON.stringify(r));
console.log('--- last 12 events overall ---');
for (const ln of lines.slice(-12)) {
  let o; try { o = JSON.parse(ln); } catch { continue; }
  console.log(o.type, o.seq, new Date(o.time).toISOString(), JSON.stringify(o.data).slice(0, 220));
}

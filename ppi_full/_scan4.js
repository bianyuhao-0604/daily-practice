const fs = require('fs');
const p = process.argv[2];
const lines = fs.readFileSync(p, 'utf8').split('\n').filter(Boolean);
for (const ln of lines) {
  let o; try { o = JSON.parse(ln); } catch { continue; }
  if (!o.data || o.data.jobId !== 'pwsh-c5b4f825b8a847e1') continue;
  console.log(o.type, o.seq, new Date(o.time).toISOString(), JSON.stringify(o.data).slice(0, 1500));
  console.log('---');
}
// also print last tool call arguments mentioning that job id
for (const ln of lines.slice(-60)) {
  let o; try { o = JSON.parse(ln); } catch { continue; }
  if (o.type === 'tool/call' && /c5b4f825b8a847e1/.test(ln)) console.log('CALL', o.seq, new Date(o.time).toISOString(), o.data.name, String(o.data.arguments).slice(0, 300));
}

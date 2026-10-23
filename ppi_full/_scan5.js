const fs = require('fs');
const p = process.argv[2];
const lines = fs.readFileSync(p, 'utf8').split('\n').filter(Boolean);
const callIds = new Set();
for (const ln of lines) {
  let o; try { o = JSON.parse(ln); } catch { continue; }
  if (o.type === 'job/registered' && o.data.jobId === 'pwsh-c5b4f825b8a847e1') callIds.add(o.data.toolCall.callId);
}
console.log('callIds', [...callIds]);
for (const ln of lines) {
  let o; try { o = JSON.parse(ln); } catch { continue; }
  if (!o.data) continue;
  const cid = o.data.callId || o.data.toolCallId || (o.data.message && o.data.message.toolCallId);
  if (!callIds.has(cid)) continue;
  const s = JSON.stringify(o.data).slice(0, 3000);
  console.log(o.type, o.seq, new Date(o.time).toISOString(), s);
  console.log('---');
}

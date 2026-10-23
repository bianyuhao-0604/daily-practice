const fs = require('fs');
const p = process.argv[2];
const lines = fs.readFileSync(p, 'utf8').split('\n').filter(Boolean);
let shown = 0;
for (let i = lines.length - 1; i >= 0 && shown < 12; i--) {
  let o; try { o = JSON.parse(lines[i]); } catch { continue; }
  const d = o.data || {};
  const m = d.message;
  if (o.type === 'tool/result' && m && m.toolCallId && !/job_output|job_list/.test(JSON.stringify(m).slice(0, 300))) continue;
  if (o.type !== 'tool/result') continue;
  shown++;
  const txt = (m.content || []).map(c => c.text || ('[' + c.type + ']')).join(' ').slice(0, 1200);
  console.log('==', o.seq, new Date(o.time).toISOString(), m.toolCallId);
  console.log(txt);
}

const fs = require('fs');
const p = process.argv[2];
const lines = fs.readFileSync(p, 'utf8').split('\n').filter(Boolean);
console.log('lines', lines.length);
const pat = /surf_from_npz|pwsh-[0-9a-f]{16}|tool-jobs/;
let n = 0;
for (let i = 0; i < lines.length; i++) {
  const ln = lines[i];
  if (!pat.test(ln)) continue;
  n++;
  const obj = JSON.parse(ln);
  const keys = Object.keys(obj).join(',');
  let s = JSON.stringify(obj);
  console.log('#' + i, '|', keys, '|', s.slice(0, 500));
  if (n > 60) break;
}

const fs = require('fs');
const zlib = require('zlib');
const p = process.argv[2];
const buf = zlib.zstdDecompressSync(fs.readFileSync(p));
const text = buf.toString('utf8');
const lines = text.split('\n').filter(Boolean);
console.log('lines', lines.length, 'bytes', text.length);
const hits = [];
for (const ln of lines) {
  if (/surf_from_npz|pwsh-8a6107fc6c7e4c7f|tool-jobs|job_id/.test(ln)) hits.push(ln);
}
console.log('hits', hits.length);
for (const h of hits.slice(-40)) console.log(h.slice(0, 700));

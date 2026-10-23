const fs = require('fs');
const zlib = require('zlib');
const src = process.argv[2], dst = process.argv[3];
const buf = fs.readFileSync(src);
const magic = Buffer.from([0x28, 0xb5, 0x2f, 0xfd]);
const offs = [];
let i = 0;
while ((i = buf.indexOf(magic, i)) !== -1) { offs.push(i); i += 4; }
console.log('frames', offs.length);
const chunks = [];
let fails = 0;
for (let k = 0; k < offs.length; k++) {
  const start = offs[k];
  const end = k + 1 < offs.length ? offs[k + 1] : buf.length;
  try {
    chunks.push(zlib.zstdDecompressSync(buf.subarray(start, end)));
  } catch (e) {
    fails++;
    if (fails < 5) console.log('fail frame', k, start, end, e.code || e.message);
  }
}
console.log('ok', chunks.length, 'fails', fails);
fs.writeFileSync(dst, Buffer.concat(chunks));
console.log('bytes', fs.statSync(dst).size);

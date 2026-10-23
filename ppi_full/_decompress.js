const fs = require('fs');
const zlib = require('zlib');
const p = process.argv[2];
const out = fs.createWriteStream(process.argv[3]);
const d = zlib.createZstdDecompress();
fs.createReadStream(p).pipe(d).pipe(out);
out.on('finish', () => console.log('done', fs.statSync(process.argv[3]).size));

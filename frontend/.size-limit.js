const fs = require("fs");
const path = require("path");

// Next/Turbopack hashes chunk names per build, so we discover them after `next build`.
const chunkDir = path.join(__dirname, ".next/static/chunks");
const chunkFiles = fs.existsSync(chunkDir)
  ? fs
      .readdirSync(chunkDir)
      .filter((name) => name.endsWith(".js"))
      .map((name) => path.join(".next/static/chunks", name))
  : [".next/static/chunks/*.js"];

module.exports = [
  {
    // Baseline ~398KB gzip (2026-07-20, Next 16.2.6 / Turbopack)
    name: "JS chunks total (gzip)",
    path: chunkFiles,
    gzip: true,
    limit: "500 kB",
  },
  ...chunkFiles.map((file) => ({
    name: `chunk ${path.basename(file)}`,
    path: file,
    gzip: true,
    limit: "200 kB",
  })),
];

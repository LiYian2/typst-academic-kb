import { execFileSync } from "node:child_process";
import { chmod, mkdir, rm } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Cloudflare Pages build images do not include the Typst CLI by default.
// This script pins the compiler version used by the site, downloads the
// official Linux binary, and exposes it through node_modules/.bin so npm
// scripts can call `typst` normally.
const typstVersion = "0.14.2";
const platformArchive = "typst-x86_64-unknown-linux-musl.tar.xz";
const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const binDir = path.join(projectRoot, "node_modules", ".bin");
const typstBin = path.join(binDir, "typst");
const tmpDir = path.join(projectRoot, ".build", "typst-install");
const archivePath = path.join(tmpDir, platformArchive);
const downloadUrl = `https://github.com/typst/typst/releases/download/v${typstVersion}/${platformArchive}`;

if (existsSync(typstBin)) {
  console.log(`Typst already installed at ${path.relative(projectRoot, typstBin)}`);
  process.exit(0);
}

await mkdir(binDir, { recursive: true });
await rm(tmpDir, { recursive: true, force: true });
await mkdir(tmpDir, { recursive: true });

console.log(`Downloading Typst ${typstVersion} for Cloudflare Pages...`);
execFileSync("curl", ["-L", "--fail", "--output", archivePath, downloadUrl], { stdio: "inherit" });
execFileSync("tar", ["-xJf", archivePath, "-C", tmpDir], { stdio: "inherit" });
execFileSync("cp", [path.join(tmpDir, `typst-x86_64-unknown-linux-musl`, "typst"), typstBin], { stdio: "inherit" });
await chmod(typstBin, 0o755);

console.log(`Installed Typst ${typstVersion} at ${path.relative(projectRoot, typstBin)}`);

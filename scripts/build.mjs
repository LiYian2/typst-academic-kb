import { execFileSync } from "node:child_process";
import { cp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const contentDir = path.join(projectRoot, "content");
const staticDir = path.join(projectRoot, "static");
const buildDir = path.join(projectRoot, ".build");
const preparedDir = path.join(buildDir, "content");
const renderedDir = path.join(buildDir, "rendered");
const publicDir = path.join(projectRoot, "public");
const originalPdfDir = path.join(projectRoot, "original_repo/ai-academic-notes/docs/public/pdfs");
const publicPdfDir = path.join(publicDir, "pdfs");
const htmlPreamble = `
// @type: raw-html pages use their raw content directly.
// @type: typst pages get the html preamble for math rendering.
#show math.equation: it => html.elem(
  "span",
  attrs: (
    class: if it.block { "math math-block" } else { "math math-inline" },
    "data-math": repr(it.body),
    title: "点击复制公式",
  ),
)[#html.frame(it)]
`;

await resetDirectories();
const sourceFiles = await collectFiles(contentDir, ".typ");
const documents = [];
const sources = [];

for (const sourcePath of sourceFiles) {
  const raw = await readFile(sourcePath, "utf8");
  const meta = readMetadata(sourcePath, raw);

  sources.push({ meta, raw });
  if (meta.type === "module") {
    await prepareModule(meta, raw);
  } else if (meta.type !== "raw-html") {
    await prepareSource(meta, raw);
  }
}

for (const { meta, raw } of sources) {
  if (meta.type === "module") continue;

  if (meta.type === "raw-html") {
    // Raw HTML pages: strip metadata comments, use rest as body
    const body = raw.replace(/^\/\/[^\n]*\n?/gm, "").trim();
    documents.push({ ...meta, body, toc: [] });
  } else {
    meta.body = await compileTypst(meta);
    meta.toc = addHeadingIdsAndExtractToc(meta);
    documents.push(meta);
  }
}

documents.sort(compareDocuments);

const sidebar = renderSidebar(documents);
for (const document of documents) {
  await writePage(document, sidebar);
}
await writeHomePage(documents, sidebar);
await cp(staticDir, publicDir, { recursive: true });
await copyPdfs();
console.log(`Built ${documents.length} documents into ${path.relative(projectRoot, publicDir)}/`);

async function resetDirectories() {
  await rm(buildDir, { recursive: true, force: true });
  await rm(publicDir, { recursive: true, force: true });
  await mkdir(preparedDir, { recursive: true });
  await mkdir(renderedDir, { recursive: true });
  await mkdir(publicDir, { recursive: true });
}

async function collectFiles(directory, extension) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await collectFiles(fullPath, extension));
    if (entry.isFile() && fullPath.endsWith(extension)) files.push(fullPath);
  }
  return files;
}

function readMetadata(sourcePath, raw) {
  const relativePath = path.relative(contentDir, sourcePath);
  const slug = relativePath.replace(/\.typ$/, "").split(path.sep).join("/");
  const doc = {
    sourcePath,
    relativePath,
    slug,
    category: slug.split("/")[0],
    title: titleCase(slug.split("/").at(-1)),
    description: "",
    order: 999,
    type: "typst",
  };
  for (const [, key, value] of raw.matchAll(/^\/\/\s*@([\w-]+):\s*(.+)$/gm)) {
    if (key === "title") doc.title = value.trim();
    if (key === "description") doc.description = value.trim();
    if (key === "order") doc.order = Number(value.trim());
    if (key === "type") doc.type = value.trim();
  }
  return doc;
}

async function prepareModule(document, raw) {
  const preparedPath = path.join(preparedDir, document.relativePath);
  await mkdir(path.dirname(preparedPath), { recursive: true });
  await writeFile(preparedPath, raw, "utf8");
}

async function prepareSource(document, raw) {
  const preparedPath = path.join(preparedDir, document.relativePath);
  await mkdir(path.dirname(preparedPath), { recursive: true });
  await writeFile(preparedPath, `${htmlPreamble}\n${replaceWikiLinks(raw)}`, "utf8");
}

function replaceWikiLinks(source) {
  return source.replace(/\[\[([a-zA-Z0-9/_-]+)(?:\|([^\]]+))?\]\]/g, (_, slug, label) => {
    return `#link("/${slug}/")[${label || titleCase(slug.split("/").at(-1))}]`;
  });
}

async function compileTypst(document) {
  const outputPath = path.join(renderedDir, document.relativePath.replace(/\.typ$/, ".html"));
  await mkdir(path.dirname(outputPath), { recursive: true });
  execFileSync("typst", [
    "compile",
    "--features", "html",
    "--format", "html",
    "--root", buildDir,
    path.join(preparedDir, document.relativePath),
    outputPath,
  ], { cwd: projectRoot, stdio: "inherit" });
  const html = await readFile(outputPath, "utf8");
  const body = html.match(/<body[^>]*>([\s\S]*?)<\/body>/i)?.[1];
  if (!body) throw new Error(`Typst did not emit an HTML body for ${document.relativePath}`);
  return body.trim();
}

function addHeadingIdsAndExtractToc(document) {
  const toc = [];
  let fallbackIndex = 0;
  document.body = document.body.replace(/<h([1-6])([^>]*)>([\s\S]*?)<\/h\1>/gi, (_, level, attrs, content) => {
    const plainText = stripTags(content).trim();
    const existingId = attrs.match(/\sid="([^"]+)"/i)?.[1];
    const id = existingId || slugify(plainText) || `section-${++fallbackIndex}`;
    if (Number(level) >= 2) toc.push({ level: Number(level), id, text: plainText });
    return `<h${level}${existingId ? attrs : `${attrs} id="${escapeAttribute(id)}"`}>${content}</h${level}>`;
  });
  return toc;
}

function renderSidebar(documents, currentSlug = "") {
  const groups = Map.groupBy(documents, (document) => document.category);
  const nav = [...groups.entries()].map(([category, pages]) => `
    <section class="nav-group">
      <div class="nav-group-title">${escapeHtml(titleCase(category))}</div>
      ${pages.map((page) => `<a href="/${page.slug}/"${page.slug === currentSlug ? ' class="active"' : ""}>${escapeHtml(page.title)}</a>`).join("\n")}
    </section>`).join("\n");
  return `<div class="nav-groups">${nav}</div>`;
}

function renderToc(toc) {
  if (!toc.length) return "";
  return `<aside class="toc" aria-label="本文目录">
    <div class="toc-title">On this page</div>
    <ol>${toc.map((item) => `<li class="depth-${item.level}"><a href="#${escapeAttribute(item.id)}">${escapeHtml(item.text)}</a></li>`).join("")}</ol>
  </aside>`;
}

async function writePage(document, sidebar) {
  const destination = path.join(publicDir, document.slug, "index.html");
  await mkdir(path.dirname(destination), { recursive: true });
  await writeFile(destination, renderLayout({
    title: document.title,
    description: document.description,
    sidebar: renderSidebar(documents, document.slug),
    toc: renderToc(document.toc),
    body: `<article class="article" data-pagefind-body>
      <header class="article-meta">
        <h1 data-pagefind-meta="title">${escapeHtml(document.title)}</h1>
        <p>${escapeHtml(document.description)}</p>
      </header>
      ${document.body}
    </article>`,
  }), "utf8");
}

async function writeHomePage(documents, sidebar) {
  const cards = documents.map((document) => `
    <li><a href="/${document.slug}/">${escapeHtml(document.title)}</a><br><span>${escapeHtml(document.description)}</span></li>`).join("");
  await writeFile(path.join(publicDir, "index.html"), renderLayout({
    title: "Academic Knowledge Base",
    description: "Typst-first academic notes",
    sidebar,
    toc: "",
    body: `<article class="article" data-pagefind-body>
      <header class="article-meta"><h1 data-pagefind-meta="title">Academic Knowledge Base</h1>
      <p>使用 Typst 写作的个人学术知识库。</p></header>
      <h2>Notes</h2><ul>${cards}</ul>
    </article>`,
  }), "utf8");
}

function renderLayout({ title, description, sidebar, toc, body }) {
  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="${escapeAttribute(description)}">
  <title>${escapeHtml(title)} · Academic KB</title>
  <link rel="stylesheet" href="/styles.css">
  <link rel="stylesheet" href="/pagefind/pagefind-component-ui.css">
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <a class="brand" href="/">Academic KB</a>
      <div class="toolbar">
        <button class="tool-button" id="nav-toggle" type="button">目录</button>
        <button class="tool-button" id="theme-toggle" type="button">明暗</button>
      </div>
      <div id="search-panel"><pagefind-searchbox show-sub-results></pagefind-searchbox></div>
      ${sidebar}
    </aside>
    <main>${body}</main>
    ${toc}
  </div>
  <script src="/pagefind/pagefind-component-ui.js" type="module"></script>
  <script src="/site.js"></script>
</body>
</html>`;
}

function compareDocuments(left, right) {
  return left.category.localeCompare(right.category) || left.order - right.order || left.title.localeCompare(right.title);
}

function titleCase(value) {
  return value.replace(/[-_]+/g, " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

function slugify(value) {
  return value.toLowerCase().replace(/<[^>]+>/g, "").replace(/[^\p{Letter}\p{Number}]+/gu, "-").replace(/^-|-$/g, "");
}

function stripTags(value) {
  return value.replace(/<[^>]*>/g, "");
}

function escapeHtml(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll("'", "&#39;");
}

async function copyPdfs() {
  try {
    await cp(originalPdfDir, publicPdfDir, { recursive: true });
    console.log(`Copied PDFs from ${path.relative(projectRoot, originalPdfDir)} to ${path.relative(projectRoot, publicPdfDir)}/`);
  } catch {
    console.log("No PDFs found to copy (directory may not exist)");
  }
}

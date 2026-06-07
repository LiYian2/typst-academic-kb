# Typst Academic Knowledge Base

一个以 Typst 为唯一正文格式的静态学术知识库。构建器调用 Typst 原生 HTML
导出，再生成 sidebar、右侧目录、深色模式、代码复制按钮和 Pagefind 搜索。

## Why no Zola?

Zola 的正文输入是 CommonMark。为了让 `content/**/*.typ` 保持唯一事实来源，本项目
直接使用一个很薄的 Node 构建器封装 Typst HTML 输出，避免额外生成 Markdown
中间层。Typst HTML 导出目前仍是实验性功能，因此转换逻辑集中在
`scripts/build.mjs` 的 `compileTypst()` 中，未来替换转换器不会影响正文。

## Local build

依赖：

- Node.js 22+
- Typst 0.14+

```bash
npm install
npm run build
npm run preview
```

只验证 Typst 和站点模板，不生成搜索索引：

```bash
npm run build:site
```

## Writing

文章写在 `content/**/*.typ`。每篇文章顶部使用注释配置元数据：

```typst
// @title: Attention Mechanism
// @description: Transformer 中注意力机制的基础笔记
// @order: 10
```

内部链接支持 `[[math/linear-algebra]]` 和
`[[math/linear-algebra|线性代数]]`。路径对应 `content/` 下不含 `.typ` 的路径。

## Cloudflare Pages

本项目保留 Cloudflare Pages Git 集成作为唯一自动部署入口。连接
`LiYian2/typst-academic-kb` 后，每次推送 `main` 都会触发 Cloudflare 构建。

Pages 构建配置：

- Production branch: `main`
- Build command: `npm ci && npm run build:cloudflare`
- Build output directory: `public`
- Node.js version: `22`

Cloudflare Pages 默认没有 Typst CLI。`npm run build:cloudflare` 会先运行
`scripts/install-typst.mjs`，下载固定版本 Typst `0.14.2` 到
`node_modules/.bin/typst`，再执行普通构建。

`wrangler.jsonc` 也记录了项目名和输出目录，便于本地预览和手动部署。
如果需要手动创建 Pages 项目：

```bash
npx wrangler pages project create typst-academic-kb --production-branch=main
```

如果需要跳过 Git 集成并手动上传当前本地构建：

```bash
npm run build
npm run deploy -- --project-name=typst-academic-kb --branch=main
```

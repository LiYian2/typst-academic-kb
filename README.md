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

仓库推送到 GitHub 后，GitHub Actions 会构建并部署到 Cloudflare Pages。先在
GitHub 仓库中配置：

- Secret `CLOUDFLARE_API_TOKEN`：具有 Pages 编辑权限的 API Token。
- Secret `CLOUDFLARE_ACCOUNT_ID`：Cloudflare Account ID。
- Variable `CLOUDFLARE_PAGES_PROJECT`：Pages 项目名；未配置时使用
  `typst-academic-kb`。

`wrangler.jsonc` 也记录了项目名和输出目录，便于本地预览和手动部署。
首次部署前先创建 Pages 项目：

```bash
npx wrangler pages project create typst-academic-kb --production-branch=main
```

之后可以手动执行 `npm run deploy`，或让 GitHub Actions 自动上传。

const root = document.documentElement;
const savedTheme = localStorage.getItem("theme");
const preferredDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
root.dataset.theme = savedTheme || (preferredDark ? "dark" : "light");

document.querySelector("#theme-toggle")?.addEventListener("click", () => {
  root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
  localStorage.setItem("theme", root.dataset.theme);
});

document.querySelector("#nav-toggle")?.addEventListener("click", () => {
  document.querySelector(".sidebar")?.classList.toggle("open");
});

for (const block of document.querySelectorAll("pre")) {
  const button = document.createElement("button");
  button.className = "copy-button";
  button.type = "button";
  button.textContent = "复制";
  button.addEventListener("click", async () => {
    await navigator.clipboard.writeText(block.querySelector("code")?.textContent || block.textContent);
    button.textContent = "已复制";
    setTimeout(() => { button.textContent = "复制"; }, 1200);
  });
  block.append(button);
}

for (const formula of document.querySelectorAll("[data-math]")) {
  formula.tabIndex = 0;
  formula.setAttribute("role", "button");
  formula.setAttribute("aria-label", `${formula.dataset.math}。点击复制公式`);
  formula.addEventListener("click", async () => {
    await navigator.clipboard.writeText(formula.dataset.math);
    formula.title = "公式已复制";
    setTimeout(() => { formula.title = "点击复制公式"; }, 1200);
  });
}

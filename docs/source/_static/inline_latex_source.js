function typesetInlineLatexSources() {
  if (typeof window.MathJax === "undefined" || typeof window.MathJax.typesetPromise !== "function") {
    return Promise.resolve();
  }
  var blocks = document.querySelectorAll(".source-with-latex");
  if (!blocks.length) {
    return Promise.resolve();
  }
  return window.MathJax.typesetPromise(Array.from(blocks)).catch(function () {});
}

function ensureMathJaxLoaded() {
  if (typeof window.MathJax !== "undefined" && typeof window.MathJax.typesetPromise === "function") {
    return Promise.resolve();
  }

  // Configure MathJax before loading when Sphinx did not inject it.
  if (typeof window.MathJax === "undefined") {
    window.MathJax = {
      options: { processHtmlClass: "mathjax-process" }
    };
  }

  return new Promise(function (resolve) {
    var script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js";
    script.async = true;
    script.onload = function () { resolve(); };
    script.onerror = function () { resolve(); };
    document.head.appendChild(script);
  });
}

document.addEventListener("DOMContentLoaded", function () {
  ensureMathJaxLoaded().then(typesetInlineLatexSources);
});

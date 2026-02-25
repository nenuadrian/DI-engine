document.addEventListener("DOMContentLoaded", function () {
  if (typeof window.MathJax === "undefined" || typeof window.MathJax.typesetPromise !== "function") {
    return;
  }
  var blocks = document.querySelectorAll(".source-with-latex");
  if (!blocks.length) {
    return;
  }
  window.MathJax.typesetPromise(Array.from(blocks)).catch(function () {});
});

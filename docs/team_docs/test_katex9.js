const katex = require('katex');
const testCases = [
  "\\text{Exit P\\&L}",
  "\\text{Exit P\\&L}",
  "P\\&L",
  "gradient(Sharpe, threshold \\pm 5\\%)",
  "\\widehat{SR}",
  "\\text{perf}_{taker\\_view}",
  "2 \\times \\text{sqrt}{P_{new}/P_{old}}",
  "\\dots",
  "\\Delta dS + \\frac{1}{2}\\Gamma dS^2 + \\mathcal{V} d\\sigma + \\dots"
];
for (let tc of testCases) {
  try {
    katex.renderToString(tc);
    console.log("Success:", tc);
  } catch (e) {
    console.log("Fail:", tc, "=>", e.message);
  }
}

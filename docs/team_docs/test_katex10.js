const katex = require('katex');
const testCases = [
  "\\text{Exit P}\\&\\text{L}",
  "\\widehat{S}",
  "\\hat{S}\\hat{R}",
  "\\ldots",
  "..."
];
for (let tc of testCases) {
  try {
    katex.renderToString(tc);
    console.log("Success:", tc);
  } catch (e) {
    console.log("Fail:", tc, "=>", e.message);
  }
}

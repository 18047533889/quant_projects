const katex = require('katex');
const testCases = [
  "\\text{ann_return}(r_{long} - r_{short})",
  "\\mathrm{ann\\_return}(r_{long} - r_{short})"
];
for (let tc of testCases) {
  try {
    katex.renderToString(tc);
    console.log("Success:", tc);
  } catch (e) {
    console.log("Fail:", tc, e.message);
  }
}

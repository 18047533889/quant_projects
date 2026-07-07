const katex = require('katex');
const testCases = [
  "\\text{Gross Return}_t",
  "\\mathrm{Gross\\_Return}_t",
  "\\text{Gross\\_Return}_t",
  "Gross\\_Return_t",
  "\\text{Gross-Return}_t"
];
for (let tc of testCases) {
  try {
    katex.renderToString(tc);
    console.log("Success:", tc);
  } catch (e) {
    console.log("Fail:", tc, e.message);
  }
}

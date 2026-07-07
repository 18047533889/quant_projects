const katex = require('katex');
try {
  katex.renderToString("\\text{Gross_Return}_t");
} catch (e) {
  console.log("Error:", e.message);
}
try {
  katex.renderToString("\\text{Gross\\_Return}_t");
  console.log("Escaped worked");
} catch (e) {
  console.log("Error 2:", e.message);
}

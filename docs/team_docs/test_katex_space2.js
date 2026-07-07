const katex = require('katex');
try {
  console.log(katex.renderToString("A _{t}"));
} catch (e) {
  console.error("Error:", e.message);
}

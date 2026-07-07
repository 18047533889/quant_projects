const katex = require('katex');
try {
  const rendered = katex.renderToString("A\\_t");
  console.log("Success!", rendered);
} catch (e) {
  console.error("Error:", e.message);
}

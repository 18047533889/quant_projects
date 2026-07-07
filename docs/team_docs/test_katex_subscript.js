const katex = require('katex');
try {
  const rendered = katex.renderToString("r_{long\\_t}");
  console.log("Success! Length:", rendered.length);
} catch (e) {
  console.error("Error:", e.message);
}

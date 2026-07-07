const katex = require('katex');
try {
  const rendered = katex.renderToString("A_ t + B_ t");
  console.log("Success! Length:", rendered.length);
} catch (e) {
  console.error("Error:", e.message);
}

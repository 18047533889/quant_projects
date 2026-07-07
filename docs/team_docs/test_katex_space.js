const katex = require('katex');
try {
  const rendered = katex.renderToString("Data_ Lineage");
  console.log("Success! Rendered.");
} catch (e) {
  console.error("Error:", e.message);
}

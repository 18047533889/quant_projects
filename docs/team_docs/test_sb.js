const katex = require('katex');
try {
  const rendered = katex.renderToString("r\\sb{long\\sb{t}} - r\\sp{2}");
  console.log("Success! Length:", rendered.length);
} catch (e) {
  console.error("Error:", e.message);
}

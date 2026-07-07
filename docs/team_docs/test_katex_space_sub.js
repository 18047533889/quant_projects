const katex = require('katex');
const cases = [
  "A _{t}",
  "\\text{Return} _{t}",
  "r _{long _{t}}",
  "||w _{t}|| _{1}"
];

for (let c of cases) {
  try {
    katex.renderToString(c);
    console.log("Success:", c);
  } catch(e) {
    console.log("Fail:", c, e.message);
  }
}

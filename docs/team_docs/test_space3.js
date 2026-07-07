const katex = require('katex');
const cases = [
  "a_ {b}",
  "a_ b",
  "\\lambda ||w_ t - w_ {t-1}||_ 1",
  "\\mathrm{Track}_ {Target}",
  "f_ {\\mathrm{raw}}"
];

for (let c of cases) {
  try {
    katex.renderToString(c);
    console.log("Success:", c);
  } catch(e) {
    console.log("Fail:", c, e.message);
  }
}

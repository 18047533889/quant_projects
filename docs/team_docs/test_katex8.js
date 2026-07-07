const katex = require('katex');
const testCases = [
  "1 - normalized_violation_count",
  "1 - \\mathrm{normalized\\_violation\\_count}",
  "f(Data_Lineage, AST) \\rightarrow Track_{Target}",
  "f(\\mathrm{Data\\_Lineage}, \\mathrm{AST}) \\rightarrow \\mathrm{Track}_{Target}"
];
for (let tc of testCases) {
  try {
    katex.renderToString(tc);
    console.log("Success:", tc);
  } catch (e) {
    console.log("Fail:", tc, e.message);
  }
}

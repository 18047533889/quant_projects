const MarkdownIt = require('markdown-it');
const md = new MarkdownIt();
let out = md.render("$\\text{Net Return}\\_t$");
console.log("With escape:", out);
out = md.render("$\\text{Net Return}_t = \\text{Gross Return}_t$");
console.log("Without escape:", out);

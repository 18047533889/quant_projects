const MarkdownIt = require('markdown-it');
const mk = require('markdown-it-katex');

const md = new MarkdownIt();
md.use(mk);

const text = "Test $\\text{Gross_Return}_t$ and $Data_Lineage$";
try {
  console.log(md.render(text));
} catch(e) {
  console.error(e.message);
}

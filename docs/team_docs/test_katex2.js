const MarkdownIt = require('markdown-it');
const mk = require('markdown-it-katex');
const md = new MarkdownIt().use(mk);
console.log(md.render('Test: $f(Data_Lineage, AST, Output_Shape, Semantic) \\rightarrow Track_{Target}$'));

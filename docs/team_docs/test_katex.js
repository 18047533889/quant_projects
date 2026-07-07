const MarkdownIt = require('markdown-it');
const mk = require('markdown-it-katex');
const md = new MarkdownIt().use(mk);
console.log(md.render('Test: $M = \\text{median}(X)$'));

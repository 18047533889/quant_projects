const fs = require('fs');
const katex = require('katex');

const text = fs.readFileSync('企业级量化因子评估与自动化入库系统需求文档.md', 'utf-8');

const inlineRegex = /(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)/g;
const blockRegex = /\$\$(.*?)\$\$/gs;

let m;
let fails = 0;
while ((m = inlineRegex.exec(text)) !== null) {
  try {
    katex.renderToString(m[1]);
  } catch (e) {
    console.log("Fail inline:", m[1], "=>", e.message);
    fails++;
  }
}

while ((m = blockRegex.exec(text)) !== null) {
  try {
    katex.renderToString(m[1]);
  } catch (e) {
    console.log("Fail block:", m[1], "=>", e.message);
    fails++;
  }
}

console.log("Total fails:", fails);

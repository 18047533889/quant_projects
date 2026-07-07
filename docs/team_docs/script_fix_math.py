import re
import os

def fix_latex_formulas(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 修复 & 符号导致的 LaTeX 错误（在 \text{} 中或公式中需要转义）
    # md-to-pdf 引擎在渲染带有 & 的 LaTeX 时容易崩溃，特别是未转义的 &
    def escape_ampersand(match):
        formula = match.group(0)
        # 如果公式中包含 & 且不是 \&，则替换为 \&
        # 注意要避开 LaTeX 环境中的对齐符号 &，但在我们的文档中大部分 & 是逻辑与
        if '&' in formula and '\\&' not in formula:
            # 简单策略：将所有 & 替换为 \&，因为在我们的公式语义中 & 几乎都是“逻辑与”
            return formula.replace('&', '\\&')
        return formula

    # 匹配 $...$ 和 $$...$$
    content = re.sub(r'\$\$.*?\$\$', escape_ampersand, content, flags=re.DOTALL)
    content = re.sub(r'\$.*?\$', escape_ampersand, content)

    # 2. 修复百分号 % 导致的 LaTeX 错误（% 是 LaTeX 的注释符，必须转义为 \%）
    def escape_percent(match):
        formula = match.group(0)
        if '%' in formula and '\\%' not in formula:
            return formula.replace('%', '\\%')
        return formula

    content = re.sub(r'\$\$.*?\$\$', escape_percent, content, flags=re.DOTALL)
    content = re.sub(r'\$.*?\$', escape_percent, content)

    # 3. 修复下划线 _ 导致的 LaTeX 错误（在 \text{} 外的下划线会被误认为下标，但在 \text{} 内需要转义）
    # 发现文档中有 signal\_intensity 这种写法，有时又没有转义
    # 我们统一确保 \text{} 内部的下划线被转义
    def fix_text_underscores(match):
        formula = match.group(0)
        # 寻找 \text{...} 块
        def replace_in_text(text_match):
            inner = text_match.group(1)
            return '\\text{' + inner.replace('_', '\\_') + '}'
        
        fixed_formula = re.sub(r'\\text\{(.*?)\}', replace_in_text, formula)
        return fixed_formula

    content = re.sub(r'\$\$.*?\$\$', fix_text_underscores, content, flags=re.DOTALL)
    content = re.sub(r'\$.*?\$', fix_text_underscores, content)

    # 4. 修复双反斜杠 \\ 导致的解析问题（在 Markdown 表格中，\\ 可能被解析为换行）
    # 特别是 \text{...} \ 或 \sqrt{...} \ 这种
    # 但 LaTeX 换行确实需要 \\。在表格中建议使用 \backslash 
    
    # 5. 针对之前报错中提到的具体错误进行修复
    # 报错：unexpected eof expecting ... }
    # 检查是否有未闭合的括号
    
    # 6. 修复特定的错误公式
    # 错误：$tanh(alpha * \text{signal\_intensity})` 或 `softplus(intensity)$
    # 这里反引号侵入了公式
    content = content.replace('$tanh(alpha * \\text{signal\\_intensity})` 或 `softplus(intensity)$', 
                              '$tanh(alpha * \\text{signal\\_intensity}) \\text{ 或 } \\text{softplus}(intensity)$')

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("LaTeX formulas fixed in Markdown.")

if __name__ == "__main__":
    file_path = "/home/yluel/share/projects/quantsociety_backend_project/docs/team_docs/企业级量化因子评估与自动化入库系统需求文档.md"
    fix_latex_formulas(file_path)

import re

def fix_latex_pipes(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all table rows. A table row starts with | and ends with |
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        # Check if line is likely a table row
        if line.strip().startswith('|'):
            # Find formulas in this line
            # We'll use a regex to find $...$ or $$...$$
            def replace_pipe_in_math(match):
                formula = match.group(0)
                # Replace | with \vert inside the formula, but be careful not to replace \| if it already exists
                # Also, we might have \left| and \right|
                formula = formula.replace('\\left|', '\\left\\vert ')
                formula = formula.replace('\\right|', '\\right\\vert ')
                # Replace remaining unescaped | with \vert
                # A bit tricky with regex, let's just replace all | that are not preceded by \
                formula = re.sub(r'(?<!\\)\|', r'\\vert ', formula)
                return formula
                
            new_line = re.sub(r'\$\$.*?\$\$', replace_pipe_in_math, line)
            new_line = re.sub(r'\$(.*?)\$', replace_pipe_in_math, new_line)
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    print("Fixed pipes in tables.")

if __name__ == "__main__":
    file_path = "/home/yluel/share/projects/quantsociety_backend_project/docs/team_docs/企业级量化因子评估与自动化入库系统需求文档.md"
    fix_latex_pipes(file_path)

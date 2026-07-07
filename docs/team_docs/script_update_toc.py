import re

def update_toc(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    toc_lines = []
    in_code_block = False
    
    for line in lines:
        if line.startswith('```'):
            in_code_block = not in_code_block
            continue
            
        if in_code_block:
            continue
            
        match = re.match(r'^(#{1,6})\s+(.*)', line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            
            if level == 1 and "需求文档" in title:
                continue
                
            # Skip the TOC heading itself
            if title.startswith("目录"):
                continue
                
            anchor = title.lower().replace(' ', '-').replace('(', '').replace(')', '').replace(':', '').replace('.', '').replace('/', '').replace('\\', '').replace('：', '').replace('（', '').replace('）', '').replace(',', '')
            anchor = re.sub(r'[^\w\- ]', '', title.lower().replace(' ', '-'))
            indent = '  ' * (level - 1)
            toc_lines.append(f"{indent}- [{title}](#{anchor})")

    new_toc = '\n'.join(toc_lines)
    
    start_idx = -1
    end_idx = -1
    
    for i, line in enumerate(lines):
        if line.startswith('## 目录'):
            start_idx = i + 1
        elif start_idx != -1 and line.startswith('## 核心术语表'):
            end_idx = i
            break
            
    if start_idx != -1 and end_idx != -1:
        new_content = '\n'.join(lines[:start_idx]) + '\n' + new_toc + '\n\n<div style="page-break-before: always;"></div>\n\n' + '\n'.join(lines[end_idx:])
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("TOC updated successfully.")
    else:
        print("Could not find TOC boundaries.", start_idx, end_idx)

if __name__ == "__main__":
    file_path = "/home/yluel/share/projects/quantsociety_backend_project/docs/team_docs/企业级量化因子评估与自动化入库系统需求文档.md"
    update_toc(file_path)

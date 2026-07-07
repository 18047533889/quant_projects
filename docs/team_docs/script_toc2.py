import re

def generate_toc(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split the content to find where to insert the TOC
    # Look for the TOC placeholder or the start of the first heading
    
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
            
            # Skip the main title
            if level == 1 and "企业级量化因子评估与自动化入库系统需求文档" in title:
                continue
                
            # Create link anchor
            anchor = title.lower().replace(' ', '-').replace('(', '').replace(')', '').replace(':', '').replace('.', '').replace('/', '').replace('\\', '').replace('：', '').replace('（', '').replace('）', '').replace(',', '')
            # A more robust anchor generation for github/markdown
            anchor = re.sub(r'[^\w\- ]', '', title.lower().replace(' ', '-'))
            
            indent = '  ' * (level - 1)
            toc_lines.append(f"{indent}- [{title}](#{anchor})")

    return '\n'.join(toc_lines)

if __name__ == "__main__":
    file_path = "/home/yluel/share/projects/quantsociety_backend_project/docs/team_docs/企业级量化因子评估与自动化入库系统需求文档.md"
    toc = generate_toc(file_path)
    print(toc)

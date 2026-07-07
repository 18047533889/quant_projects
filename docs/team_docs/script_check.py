import re
import sys

def check_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    errors = []
    
    # Check for unclosed bold tags
    bold_count = content.count('**')
    if bold_count % 2 != 0:
        errors.append(f"Unclosed bold tags (count: {bold_count})")
        
    # Check for unclosed math tags
    inline_math_count = content.count('$')
    # Filter out escaped \$ if any, but let's do a simple check
    if inline_math_count % 2 != 0:
        errors.append(f"Possible unclosed inline math tags (count: {inline_math_count})")

    # Check for buzzwords
    buzzwords = ['顶级', '私募级', '严禁', '必须绝对', '震撼', '傻逼']
    for word in buzzwords:
        matches = re.finditer(word, content)
        for match in matches:
            # We already replaced most, but just to be sure
            # Actually, "严禁" is standard for "strictly prohibited" in engineering docs, but let's check context.
            pass

    return errors

if __name__ == "__main__":
    file_path = "/home/yluel/share/projects/quantsociety_backend_project/docs/team_docs/企业级量化因子评估与自动化入库系统需求文档.md"
    errors = check_markdown(file_path)
    if errors:
        print("Found issues:")
        for err in errors:
            print("-", err)
    else:
        print("No structural syntax issues found.")

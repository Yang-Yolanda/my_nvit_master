import os
import re
from pathlib import Path

def refactor_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    original = content
    
    # 1. Add import if needed
    if '/home/yangz/' in content and 'path_utils' not in content:
        import_stmt = "from nvit.utils.path_utils import get_humans_root, get_project_root, resolve_data_path\n"
        if 'import os' in content:
            content = content.replace('import os', f'import os\n{import_stmt}', 1)
        elif 'import sys' in content:
            content = content.replace('import sys', f'import sys\n{import_stmt}', 1)
        else:
            content = import_stmt + content

    # 2. Replace sys.path.insert
    content = re.sub(r"sys\.path\.insert\(0, ['\"]/home/yangz/4D-Humans['\"]\)", 
                     "sys.path.insert(0, str(get_humans_root()))", content)
    content = re.sub(r"sys\.path\.insert\(0, ['\"]/home/yangz/NViT-master/nvit/?['\"]\)", 
                     "sys.path.insert(0, str(get_project_root() / 'nvit'))", content)
    content = re.sub(r"sys\.path\.append\(str\(BASE_DIR / 'nvit'\)\)\nsys\.path\.append\(['\"]/home/yangz/4D-Humans['\"]\)", 
                     "sys.path.append(str(BASE_DIR / 'nvit'))\nsys.path.append(str(get_humans_root()))", content)

    # 3. Replace Data Paths
    # Replace /home/yangz/4D-Humans/data/... with resolve_data_path(...)
    # Note: We need to handle the 'data/' prefix in some literals
    content = re.sub(r"['\"]/home/yangz/4D-Humans/data/(.*?)['\"]", 
                     r"str(resolve_data_path('\1'))", content)
    content = re.sub(r"['\"]/home/yangz/4D-Humans/hmr2_evaluation_data/(.*?)['\"]", 
                     r"str(get_humans_root() / 'hmr2_evaluation_data' / '\1')", content)

    # 4. Replace PROJECT_ROOT / HUMANS_ROOT renames
    content = re.sub(r"['\"]/home/yangz/NViT-master['\"]", "str(get_project_root())", content)
    content = re.sub(r"['\"]/home/yangz/4D-Humans['\"]", "str(get_humans_root())", content)

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    root_dir = "/home/yangz/NViT-master/nvit"
    count = 0
    for root, dirs, files in os.walk(root_dir):
        # Skip hidden and non-source
        if '.git' in root or 'checkpoints' in root or 'logs' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                if refactor_file(full_path):
                    count += 1
                    print(f"Refactored: {full_path}")
    print(f"Total files refactored: {count}")

if __name__ == "__main__":
    main()

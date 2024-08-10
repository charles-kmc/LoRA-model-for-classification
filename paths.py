import os 

dir_root = os.path.dirname(os.path.abspath(__file__))

def create_results_dir(dir_root = dir_root):
    root_dir, class_folders, project_name = dir_root.rsplit("/", 2)
    save_dir = os.path.join(root_dir, "Results",class_folders, project_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


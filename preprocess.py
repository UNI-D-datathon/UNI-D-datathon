import os
import shutil


def preprocess_training(base_dir):
    clean_dir = os.path.join(base_dir, 'clean')
    noisy_dir = os.path.join(base_dir, 'noisy')

    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    source_dirs = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if 'GT' in dir_name:
                source_dirs.append(os.path.join(root, dir_name))

    # if not source_dirs:
    #     raise ValueError("No directory containing 'GT' found")

    # 파일 옮김
    for source_dir in source_dirs:
        for filename in os.listdir(source_dir):
            if filename.endswith('.jpg'):
                shutil.move(os.path.join(source_dir, filename), os.path.join(clean_dir, filename))

    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name not in ['clean', 'noisy'] and 'GT' not in dir_name:
                current_dir = os.path.join(root, dir_name)
                for filename in os.listdir(current_dir):
                    if filename.endswith('.jpg'):
                        shutil.move(os.path.join(current_dir, filename), os.path.join(noisy_dir, filename))

    for root, dirs, files in os.walk(base_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if dir_name not in ['clean', 'noisy']:
                shutil.rmtree(dir_path)

    print('Training preprocessing done')

def preprocess_validation(base_dir):
    clean_dir = os.path.join(base_dir, 'clean')
    noisy_dir = os.path.join(base_dir, 'noisy')

    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    source_dirs = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if 'GT' in dir_name:
                source_dirs.append(os.path.join(root, dir_name))

    # if not source_dirs:
    #     raise ValueError("No directory containing 'GT' found")

    for source_dir in source_dirs:
        for filename in os.listdir(source_dir):
            if filename.endswith('.jpg'):
                shutil.move(os.path.join(source_dir, filename), os.path.join(clean_dir, filename))

    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name not in ['clean', 'noisy'] and 'GT' not in dir_name:
                current_dir = os.path.join(root, dir_name)
                for filename in os.listdir(current_dir):
                    if filename.endswith('.jpg'):
                        shutil.move(os.path.join(current_dir, filename), os.path.join(noisy_dir, filename))

    for root, dirs, files in os.walk(base_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if dir_name not in ['clean', 'noisy']:
                shutil.rmtree(dir_path)

    print('Validation preprocessing done')

preprocess_validation('./data/Validation/원천/훼손블러/밝은조도/')
preprocess_validation('./data/Validation/원천/훼손블러/어두운조도/')
preprocess_training('./data/Training/원천/훼손블러/밝은조도/')
preprocess_training('./data/Training/원천/훼손블러/어두운조도/')
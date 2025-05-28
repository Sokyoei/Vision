import sys
from pathlib import Path


def main():
    config_h_msbuild, config_h, solution_dir = sys.argv[1:]

    vision_root = str(Path(solution_dir)).replace('\\', '/')
    with open(config_h_msbuild, 'r') as f:
        in_file_text = f.read()

    with open(config_h, 'w') as f:
        f.write(in_file_text.replace('@@VISION_ROOT@@', f'"{vision_root}"'))


if __name__ == '__main__':
    main()

"""
使用 LibreOffice 将 PPT 转换为图片

```shell
sudo apt install libreoffice
pip install pdf2image poppler-utils
```
"""

import subprocess
from pathlib import Path

from pdf2image import convert_from_path


def ppt2image(ppt_file_path: str, output_folder_path: str) -> bool:
    ppt_file_path: Path = Path(ppt_file_path)
    output_folder_path: Path = Path(output_folder_path)

    output_folder_path.mkdir(parents=True, exist_ok=True)
    temp_pdf_file = ppt_file_path.with_suffix(".pdf")

    # ppt to pdf
    cmd = ['libreoffice', '--headless', '--convert-to', 'pdf', ppt_file_path, temp_pdf_file]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        result.check_returncode()
    except subprocess.CalledProcessError:
        return False

    images = convert_from_path(temp_pdf_file, dpi=300, fmt='png')
    for i, img in enumerate(images):
        img.save(output_folder_path / f'page_{i+1}.png', 'PNG')

    temp_pdf_file.unlink()

    return True


def main():
    pptfile = 'sample2.pptx'
    png_folder = 'output_images'

    result = ppt2image(pptfile, png_folder)
    print(result)


if __name__ == "__main__":
    main()

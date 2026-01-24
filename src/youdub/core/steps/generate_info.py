import json
import os
from PIL import Image

def resize_thumbnail(folder, size=(1280, 960)):
    image_suffix = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    for suffix in image_suffix:
        image_path = os.path.join(folder, f'download{suffix}')
        if os.path.exists(image_path):
            break
    else:
        return None

    with Image.open(image_path) as img:
        img_ratio = img.width / img.height
        target_ratio = size[0] / size[1]

        if img_ratio < target_ratio:
            new_height = size[1]
            new_width = int(new_height * img_ratio)
        else:
            new_width = size[0]
            new_height = int(new_width / img_ratio)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        new_img = Image.new('RGB', size, "black")

        x_offset = (size[0] - new_width) // 2
        y_offset = (size[1] - new_height) // 2
        new_img.paste(img, (x_offset, y_offset))

        new_img_path = os.path.join(folder, 'video.png')
        new_img.save(new_img_path)
        return new_img_path

def generate_summary_txt(folder):
    summary_path = os.path.join(folder, 'summary.json')
    if not os.path.exists(summary_path):
        return

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    title = summary.get("title", "Untitled")
    author = summary.get("author", "Unknown")
    summary_text = summary.get("summary", "")
    
    txt = f'{title} - {author}\n\n{summary_text}'
    with open(os.path.join(folder, 'video.txt'), 'w', encoding='utf-8') as f:
        f.write(txt)

def generate_info(folder):
    generate_summary_txt(folder)
    resize_thumbnail(folder)
    
def generate_all_info_under_folder(root_folder):
    for root, dirs, files in os.walk(root_folder):
        if 'download.info.json' in files:
            generate_info(root)
    return f'Generated all info under {root_folder}'


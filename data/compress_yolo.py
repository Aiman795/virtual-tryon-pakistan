import os
from PIL import Image

input_dir = 'data/yolo_dataset/images'
count = 0

for split in ['train', 'val']:
    folder = os.path.join(input_dir, split)
    for f in os.listdir(folder):
        if f.endswith('.jpg') or f.endswith('.png'):
            path = os.path.join(folder, f)
            img = Image.open(path).convert('RGB')
            img = img.resize((416, 416))
            img.save(path, 'JPEG', quality=80)
            count += 1
            if count % 100 == 0:
                print(f'{count} done')

print('All compressed!')
print(f'Total: {count} images')

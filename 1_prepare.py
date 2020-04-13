import os
from imutils import paths
import argparse
translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "ragno": "spider", "scoiattolo": "squirrel"}

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='Path to the data.')
args = vars(ap.parse_args())

my_path= args['dataset']
BASE_IMG = os.path.join(os.getcwd(), my_path)

dir_paths = [os.path.join(my_path, dir) for dir in os.listdir(BASE_IMG)]
for dir_path, (k, v) in zip(dir_paths, translate.items()):
    os.rename(dir_path, dir_path.replace(k, v))

dir_paths = [os.path.join(my_path, dir) for dir in os.listdir(BASE_IMG)]
for dir_path in dir_paths:
    i = 0
    imgs = paths.list_images(dir_path)
    for img in imgs:
        print(img)
        os.rename(img, os.path.join(dir_path,  f'{i:04d}.png'))
        i += 1

for dir_path, (k, v) in zip(dir_paths, translate.items()):
    os.rename(dir_path, dir_path.replace(k, v))

print('...')
for dir_path in dir_paths:
    class_name = dir_path.split('\\')
    print(f'Number of images in folder {class_name[-1]}: {len(os.listdir(dir_path))}')
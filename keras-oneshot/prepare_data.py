import os
import sys
import shutil

src_folder = sys.argv[1]
debug = False
if len(sys.argv) > 2:
    debug = sys.argv[2]

def delete_empty(root_path):
    for f in os.listdir(root_path):
        path = root_path + f 
        if os.path.isdir(path):
            if not os.listdir(path):
                print(path)
                os.rmdir(path)

def listdir(path):
    l = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            l.append(f)
    return l


delete_empty(src_folder)

folders = listdir(src_folder)
if debug:
    print('Debug Mode!')
    folders = folders[:120]

train_range = int(len(folders)*0.7)

if not os.path.exists('train'):
    print( 'train  does not exist, creating')
    os.mkdir('train')
else:
    print('Already exists: train')

if not os.path.exists('val'):
    print( 'val  does not exist, creating')
    os.mkdir('val')
else:
    print('Already exists: val')

for i in range(train_range):
    folder = folders[i]
    path = os.path.join(src_folder, folder)
    shutil.copytree(path,'train/' + folder)



for i in range(train_range, len(folders)):
    folder = folders[i]
    path = os.path.join(src_folder, folder)
    shutil.copytree(path,'val/' + folder)

    
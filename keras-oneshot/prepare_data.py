import os
import sys
import shutil
import rm_empty
src_folder = sys.argv[1]
debug = False
if len(sys.argv) > 2:
    debug = sys.argv[2]

def clear_out_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def listdir(path):
    l = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            l.append(f)
    return l


rm_empty.delete_empty(src_folder)

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
    clear_out_folder('train')


if not os.path.exists('val'):
    print( 'val  does not exist, creating')
    os.mkdir('val')
else:
    print('Already exists: val')
    clear_out_folder('val')


for i in range(train_range):
    folder = folders[i]
    path = os.path.join(src_folder, folder)
    shutil.copytree(path,'train/' + folder)



for i in range(train_range, len(folders)):
    folder = folders[i]
    path = os.path.join(src_folder, folder)
    shutil.copytree(path,'val/' + folder)

    

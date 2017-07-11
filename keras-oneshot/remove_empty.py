import os

for f in os.listdir('train'):
    path = 'train/' + f 
    if os.path.isdir(path):
        if not os.listdir(path):
            print(path)
            os.rmdir(path)

for f in os.listdir('train'):
    path = 'val/' + f 
    if os.path.isdir(path):
        if not os.listdir(path):
            print(path)
            os.rmdir(path)


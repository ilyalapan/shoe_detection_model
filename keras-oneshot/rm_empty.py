import os
import sys
def delete_empty(root_path):
    print('Removing empty:  ', root_path)
    for f in os.listdir(root_path):
        path = os.path.join(root_path,f)
        if os.path.isdir(path):
            if not os.listdir(path):
                print(path)
                os.rmdir(path)

def main():
    path = sys.argv[1]
    delete_empty(path)

if __name__ == "__main__":
    # execute only if run as a script
    main()

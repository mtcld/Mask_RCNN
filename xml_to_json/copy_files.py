import os
import argparse
import shutil as sh

parser = argparse.ArgumentParser(epilog='This just copy folder has 2 levels')
parser.add_argument('-e',help='file extension')
parser.add_argument('-i',help='input path')
parser.add_argument('-o',help='output path',default='./out')
parser.add_argument('-r',help='rate of train/test',default=0.8)
args = parser.parse_args()


extension   = args.e
input_path  = args.i
output_path = args.o
rate        = args.r


output_path = os.path.join(output_path,'train')
if not os.path.exists(output_path) :
    os.makedirs(output_path)
    os.makedirs(output_path.replace('/train','/test'))


# create a dict contain src and dest path
copy_list = {}
folder_list = os.listdir(input_path)
for folder in folder_list:
    folder_path = os.path.join(input_path,folder)
    file_list = os.listdir(folder_path)
    for file in file_list:
        if file.endswith(extension):
            src_dir=os.path.abspath(os.path.join(folder_path,file))
            dest_dir=os.path.abspath(os.path.join(output_path,file))
            copy_list[src_dir]=dest_dir


for idx ,key in enumerate(copy_list.keys()):
    if idx < int(len(copy_list)*rate):
        sh.copyfile(key,copy_list[key])
    else:
        sh.copyfile(key,copy_list[key].replace('/train/','/test/'))
    print('{} => {}'.format(src_dir,dest_dir))

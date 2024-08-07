import os
import re

path = 'TRY_formal'


def make_the_dir():  # 一开始上面两个函数都要执行一下，相当于初始化
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(20):
        num = len(os.listdir(path))
        if not os.path.exists(path + '/No.{}'.format(num)):
            os.makedirs(path + '/No.{}'.format(num))


# 获取该目录下所有文件，存入列表中
def rename_the_dir():
    new = 'GOOD'
    file = os.listdir(path)
    for i in file:
        old_name = path + '/' + i       # 设置旧文件名（就是路径+文件名）
        new_name = path + '/' + new + i     # 设置新文件名

        os.rename(old_name, new_name)       # 用os模块中的rename方法对文件改名

import shutil
# os.remove是删除的意思
def move_the_file():
    former_path = 'TRY_formal'
    latter_path = 'TRY_latter'
    aaa = os.listdir(former_path)
    for file in aaa:
        shutil.move(former_path+'/'+file, latter_path+'/'+file)
        num_2 = len(os.listdir(latter_path))
        os.rename(latter_path+'/'+file, latter_path+'/{}'.format(num_2))

if __name__ == "__main__":
    # make_the_dir()
    # rename_the_dir()
    move_the_file()

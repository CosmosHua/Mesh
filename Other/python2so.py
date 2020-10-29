# coding:utf-8
#!/usr/bin/python3
# https://github.com/ArvinMei/py2so

import os, sys, shutil, time
from distutils.core import setup
from Cython.Build import cythonize


################################################################################
"""
Function：为更好的隐藏源码，将py文件编译为so文件。
Requirement：1.系统要安装python-devel和gcc；2.python要安装cython。
Usage:  1.编译当前文件夹：python py2so.py
        2.编译某个文件夹：python py2so.py DIR
Result：在文件夹build/下
生成完成后：若需要py/pyc，须将启动的py/pyc拷贝到编译目录并删除so文件。
"""


build_dir = "build"
starttime = time.time()
################################################################################
def getpy(basepath=os.path.abspath('.'), parentpath='', name='', excepts=(), copyOther=False, delC=False):
    """Get the paths of *.py files:
    :param basepath: root path
    :param parentpath: parent path
    :param name: file or dir
    :param excepts: the excluded files
    :param copy: whether copy other files
    :return: the iterator of py files"""
    fullpath = os.path.join(basepath, parentpath, name)
    for fname in os.listdir(fullpath):
        ffile = os.path.join(fullpath, fname)
        #print(basepath, parentpath, name, ffile)
        if os.path.isdir(ffile) and fname != build_dir and not fname.startswith('.'):
            for f in getpy(basepath, os.path.join(parentpath, name), fname, excepts, copyOther, delC):
                yield f
        elif os.path.isfile(ffile):
            ext = os.path.splitext(fname)[1]
            if ext == ".c":
                if delC and os.stat(ffile).st_mtime > starttime: os.remove(ffile)
            elif ffile not in excepts and os.path.splitext(fname)[1] not in('.pyc', '.pyx'):
                if os.path.splitext(fname)[1] in('.py', '.pyx') and not fname.startswith('__'):
                    yield os.path.join(parentpath, name, fname)
                elif copyOther:
                    dstdir = os.path.join(basepath, build_dir, parentpath, name)
                    if not os.path.isdir(dstdir): os.makedirs(dstdir)
                    shutil.copyfile(ffile, os.path.join(dstdir, fname))
        else: pass


currdir = os.path.abspath('.')
parentpath = sys.argv[1] if len(sys.argv)>1 else ""
setupfile = os.path.join(os.path.abspath('.'), __file__)
build_tmp_dir = build_dir + "/temp"
################################################################################
module_list = list(getpy(basepath=currdir, parentpath=parentpath, excepts=(setupfile)))
try:
    setup(ext_modules = cythonize(module_list),script_args=["build_ext", "-b", build_dir, "-t", build_tmp_dir])
except Exception: print("Some Error:", sys.exc_info()[0])
else: module_list = list(getpy(basepath=currdir, parentpath=parentpath, excepts=(setupfile), copyOther=True))

module_list = list(getpy(basepath=currdir, parentpath=parentpath, excepts=(setupfile), delC=True))
if os.path.exists(build_tmp_dir): shutil.rmtree(build_tmp_dir)

print("Complate! Time:", time.time()-starttime, 's')


################################################################################

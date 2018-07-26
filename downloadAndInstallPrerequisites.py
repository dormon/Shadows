#!/usr/bin/python

import os
from subprocess import call
import argparse

parser = argparse.ArgumentParser(description='Download/pull all Shadows dependencies, compile and install them.')
parser.add_argument('--threads', type=int, default=4,  help='number of threads for compilation')
parser.add_argument('--dontBuildDebug',action='store_true')
parser.add_argument('--dontBuildRelease',action='store_true')
parser.add_argument('--installDir', type=str, default="install", help='where to install all repositories')
parser.add_argument('--repoDir', type=str, default="repositories", help='where to download repositories')

args = parser.parse_args()

threads = args.threads
buildDebug = not args.dontBuildDebug
buildRelease = not args.dontBuildRelease
installDir = args.installDir
repoDir = args.repoDir
curDir = os.path.abspath(".")

if not os.path.isabs(installDir):
    installDir = os.path.join(os.path.abspath("."),installDir)

if not os.path.isdir(installDir):
    os.makedirs(installDir)

if not os.path.isabs(repoDir):
    repoDir = os.path.join(os.path.abspath("."),repoDir)

if not os.path.isdir(repoDir):
    os.makedirs(repoDir)

def getGitDirectory(url):
    return url[url.rfind("/")+1:url.rfind(".")]

def clone(url):
    os.chdir(repoDir)
    gitDir = getGitDirectory(url)
    if not os.path.isdir(gitDir):
        print "cloning: "+gitDir
        call(["git","clone",url])
    else:
        print "executing git pull on: "+gitDir
        os.chdir(gitDir)
        call(["git","pull"])
        os.chdir("..")
    os.chdir(curDir)

gits = [
("git@github.com:spurious/SDL-mirror.git"     ,[]),
("git@github.com:assimp/assimp.git"           ,["-DASSIMP_BUILD_SAMPLES=OFF","-DASSIMP_BUILD_ASSIMP_TOOLS=OFF","DASSIMP_BUILD_TESTS=OFF"]),
("git@github.com:g-truc/glm.git"              ,[]),
("git@github.com:dormon/SDL2CPP.git"          ,[]),
("git@github.com:dormon/imguiDormon.git"      ,[]),
("git@github.com:dormon/imguiOpenGLDormon.git",[]),
("git@github.com:dormon/imguiSDL2Dormon.git"  ,[]),
("git@github.com:dormon/imguiSDL2OpenGL.git"  ,[]),
("git@github.com:dormon/geGL.git"             ,[]),
("git@github.com:dormon/BasicCamera.git"      ,[]),
("git@github.com:dormon/MealyMachine.git"     ,[]),
("git@github.com:dormon/TxtUtils.git"         ,[]),
("git@github.com:dormon/ArgumentViewer.git"   ,[]),
        ]

def buildAndInstall(url,args = []):
    os.chdir(repoDir)
    dirName = getGitDirectory(url)
    os.chdir(dirName)
    if not os.path.isdir("build/linux/debug"):
       os.makedirs("build/linux/debug")
    if not os.path.isdir("build/linux/release"):
        os.makedirs("build/linux/release")

    basicArgs  = ["cmake","-DCMAKE_INSTALL_PREFIX="+installDir,"-DBUILD_SHARED_LIBS=ON"] 
    debugArg   = ["-DCMAKE_BUILD_TYPE=Debug"]
    releaseArg = ["-DCMAKE_BUILD_TYPE=Release"]
    dirArg     = ["../../.."]
    os.chdir("build/linux/")
    if buildDebug:
        os.chdir("debug")
        if not os.path.isfile("Makefile"):
            call(basicArgs+debugArg+args+dirArg)
        call(["make","-j"+str(threads),"install"])
        os.chdir("..")
    if buildRelease:
        os.chdir("release")
        if not os.path.isfile("Makefile"):
            call(basicArgs+releaseArg+args+dirArg)
        call(["make","-j"+str(threads),"install"])
        os.chdir("..")

for i in gits:
    clone(i[0])

for i in gits:
    buildAndInstall(i[0],i[1])

#!/usr/bin/python

import os
from subprocess import call
from subprocess import Popen, PIPE
import argparse
import re

parser = argparse.ArgumentParser(description='Download/pull all Shadows dependencies, compile and install them.')
parser.add_argument('--threads', type=int, default=4,  help='number of threads for compilation')
parser.add_argument('--dontBuildDebug',action='store_true')
parser.add_argument('--dontBuildRelease',action='store_true')
parser.add_argument('--installDir', type=str, default="install", help='where to install all repositories')
parser.add_argument('--repoDir', type=str, default="repositories", help='where to download repositories')

args = parser.parse_args()

threads      = args.threads
buildDebug   = not args.dontBuildDebug
buildRelease = not args.dontBuildRelease
installDir   = args.installDir
repoDir      = args.repoDir
curDir       = os.path.abspath(".")

def getGCC():
    GCCs = ["g++","g++-5","g++-6","g++-7"]
    standards = ["--std=c++14","--std=c++17"]
    
    def hasGCC(what):
        p = Popen(["which",what],stdout=PIPE,stderr=PIPE)
        p.communicate()
        return not p.returncode
    
    hasGCCs = map(lambda x:hasGCC(x),GCCs)
    
    if not reduce(lambda x,y:x or y,hasGCCs):
        print "there is no g++"
        exit(0)
    
    def getVersion(whatGCC):
        versionLine = Popen([whatGCC,"--version"],stdout=PIPE,stderr=PIPE).communicate()[0].split("\n")[0];
        return re.sub(".*\s([0-9](\\.[0-9])+).*","\\1",versionLine)
    
    def isVersionLess(a,b):
        a = a.split(".")
        b = b.split(".")
        while len(a) < len(a):
            a += ["0"]
        while len(b) < len(a):
            b += ["0"]
        ab = zip(a,b)
        for i in ab:
            if int(i[0]) >= int(i[1]):
                return False
        return True
    
    def getNewestGCC():
        allGCCs = zip(GCCs,hasGCCs)
        existingGCCs = filter(lambda x:x[1],allGCCs)
        existingGCCs = map(lambda x:x[0],existingGCCs)
        versions = map(lambda x:getVersion(x),existingGCCs)
        gccWithVersion = zip(existingGCCs,versions)
        newestGCC = reduce(lambda x,y:x if isVersionLess(x[1],y[1]) else y,gccWithVersion)[0]
        return newestGCC
    
    gcc = getNewestGCC()
    
    def supportStandard(standard):
        return not (Popen([gcc,standard],stdout=PIPE,stderr=PIPE).communicate()[0].find("unrecognized") >= 0)
    
    supportedStandards = map(lambda x:supportStandard(x),standards)
    
    if not reduce(lambda x,y:x or y,supportedStandards):
        print "your g++ is too old and does not support required C++ standard: ",standards[0]
        exit(0)
    
    def getNewestStandard():
        return filter(lambda x:x[1],zip(standards,supportedStandards))[-1][0]
    
    standard = getNewestStandard();
    return (gcc,standard)

gcc = getGCC()




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
("git@github.com:dormon/Vars.git"             ,[]),
        ]

def buildAndInstall(url,args = []):
    os.chdir(repoDir)
    dirName = getGitDirectory(url)
    os.chdir(dirName)
    if not os.path.isdir("build/linux/debug"):
       os.makedirs("build/linux/debug")
    if not os.path.isdir("build/linux/release"):
        os.makedirs("build/linux/release")

    basicArgs  = ["cmake","-DCMAKE_INSTALL_PREFIX="+installDir,"-DBUILD_SHARED_LIBS=ON","-DCMAKE_CXX_COMPILER="+gcc[0],"-DCMAKE_CXX_FLAGS="+gcc[1]] 
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
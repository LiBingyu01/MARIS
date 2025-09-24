import time
import copy
import os
import sys
import tarfile
import re
import webbrowser

PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve

VOC_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar"
VOC_TEST_URL="http://host.robots.ox.ac.uk/eval/downloads/VOC2010test.tar/"
PASD_URL="https://codalabuser.blob.core.windows.net/public/%s"
JSON_REGEX='trainval_.*'


progress = None
json_regex = re.compile(JSON_REGEX)

def input23(prompt):
    """
    Calls raw_input() if python 2, input() if python 3
    """
    if PYTHON_VERSION == 2:
        return raw_input(prompt)
    elif PYTHON_VERSION == 3:
        return input(prompt)


def printProgress(count, blockSize, totalSize):
    global progress
    if progress is None:
        print('Total size: %.2f GB' % (float(totalSize) / (2 ** 30)))
        progress = 0

    prev_progress = progress
    progress = int(float(count * blockSize) / totalSize * 100)
    if progress > prev_progress:
        print("Download %d%% complete." % progress)


if len(sys.argv) < 3 or sys.argv[1] == '-h' or sys.argv[1] == '--help':
    print("usage: python download.py <dataset> <folder>\n" \
          + "Download PASCAL in Detail data to <folder>.\n" \
          + "<dataset> options: 'pascal' to download VOCdevkit,\n"
          + "trainval_preview1 to download trainval_preview1.json.")
    exit(1)

if not os.path.isdir(sys.argv[2]):
    print("%s is not a directory." % sys.argv[2])
    exit(1)


rootdir = sys.argv[2]

# Download original PASCAL VOC data
if sys.argv[1].lower() == 'pascal':
    filepath = os.path.join(rootdir, "VOCtrainval_03-May-2010.tar")

    # Download trainval
    if os.path.exists(os.path.join(rootdir, 'VOCdevkit')):
        print("Directory %s already exists - assuming PASCAL trainval already downloaded."
              % os.path.join(rootdir, 'VOCdevkit'))
    else:
        print("Downloading VOCdevkit 2010 trainval data to %s." % filepath)

        if os.path.exists(filepath):
            print("Tar file appears to already be downloaded. Using it.")
        else:
            urlretrieve(VOC_URL, filepath + '.download', reporthook=printProgress)
            os.rename(filepath + '.download', filepath)
            print("Download complete!")


        print("Unpacking. This may take a few minutes...")
        if os.path.exists(os.path.join(rootdir, 'VOCdevkit')):
            print("VOCdevkit directory already present. Aborting unpacking.")
        else:
            tar = tarfile.open(filepath)
            tar.extractall(path=rootdir)
            tar.close()

        print("Cleaning up.")
        if os.path.exists(os.path.join(rootdir, 'VOCdevkit')):
            os.remove(filepath)

    # Download test
    testTar = os.path.join(rootdir, 'download.tar')
    if not os.path.exists(testTar):
        print(("To download the test images, visit %s,\n"
              + "create an account, and save download.tar (about 1.13 GB) "
              + "to %s.")
              % (VOC_TEST_URL, os.path.abspath(testTar)))

        s = None
        while s not in ['y','n']:
            s = input23('Download test images now? (opens browser) y/n: ').lower()
        if s == 'y':
            webbrowser.open(VOC_TEST_URL)

        while not os.path.exists(testTar):
            input23('Press enter once download.tar is done downloading to %s.' % testTar)
            if not os.path.exists(testTar):
                print("File %s doesn't exist." % os.path.abspath(testTar))

    print("Unpacking test data from %s. This may take a few minutes..." % testTar)
    tar = tarfile.open(testTar)
    tar.extractall(path=rootdir)
    tar.close()

    print("Success! You can delete download.tar now.")

# Download PASCAL in Detail JSON
elif json_regex.match(sys.argv[1].lower()):
    filename = sys.argv[1].lower() + '.json'
    filepath = os.path.join(rootdir, filename)
    url = PASD_URL % filename

    if os.path.exists(filepath):
        print("%s already exists. Aborting." % filepath)
        exit(1)

    print("Downloading %s to %s from:\n%s" % (filename, filepath, url))

    urlretrieve(url, filepath + '.download', reporthook=printProgress)
    os.rename(filepath + '.download', filepath)
    print("Download complete!")
else:
    print('Don\'t recognize dataset %s' % sys.argv[1].lower())

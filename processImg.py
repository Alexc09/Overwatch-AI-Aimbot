'''
- Used to rename screenshot .pngs to 1..2..3....~
- Used before annotations are made. YOU MUST RENAME IMAGES BEFORE DOING ANNOTATIONS
python processImg.py Workspace/images/screenshots
'''

import glob, os, sys, time

# The 2nd item will be the dir
# python processImg.py Workspace/images/screenshots
SCREENSHOT_DIR = sys.argv[1]
label_img_tool = "../RealTimeObjectDetection/labelImg/labelImg.py"

# SCREENSHOT_DIR = "Workspace/images/screenshots"

# os.chdir(SCREENSHOT_DIR)

file_list = []
file_types = ["*.jpg",  "*.png"]
for file_type in file_types:
    file_list.extend(glob.glob(SCREENSHOT_DIR + "/" + file_type))

count = 1
for file in file_list:
    print(file)
    try:
        os.rename(file, f'{SCREENSHOT_DIR}/{count}.png')
        # annot = file.split('.')[0] + ".xml"
        # os.rename(annot, f'{count}.xml')
        count += 1
    except Exception as e:
        print('Failed, moving on')
        print(e)
        continue

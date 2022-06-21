import os  # dealing with directories

for dir in os.listdir('./images'):
    for img in os.listdir('./images/' + dir):
        olddir = './images/' + dir + '/' + img
        newdir = './images_new/' + dir + '.' + img.split('.')[0] + '.' + img.split('.')[1]
        os.rename(dir, newdir)

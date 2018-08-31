from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

# NMS is in TF https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression
# https://github.com/tryolabs/luminoth/blob/3c67eb34e601e72ae800287c57caf9be6538d601/luminoth/models/fasterrcnn/rpn_proposal.py

# http://host.robots.ox.ac.uk/pascal/VOC/voc2012/examples/index.html

# IOU https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
SIDE = 224 # Desired Img Size

id = "2012_004328"
img = Image.open("Data/Images/{}.jpg".format(id))
img = img.resize([SIDE,SIDE], resample= Image.BILINEAR)

root = ET.parse("Data/Annot/{}.xml".format(id)).getroot()

name = root.find("object/name").text
bbox = root.find("object/bndbox")

print name
width = float(root.find("size/width").text)
height = float(root.find("size/height").text)

# Make sure these are ints before printing out.
x_max = float(bbox.find("xmax").text)/width * SIDE
y_max = float(bbox.find("ymax").text)/height * SIDE
x_min = float(bbox.find("xmin").text)/width * SIDE
y_min = float(bbox.find("ymin").text)/height * SIDE

draw = ImageDraw.Draw(img)
draw.rectangle([x_max, y_max, x_min, y_min], outline=(0,200,0))
draw.text([x_max, y_min], name, fill=(0,200,0))
img.show()

# Generate 64, 128,256 pixel and .5, 1, 1.5 Aspect Ratio
# Output 60 such, 25% Foreground 75% Background (15/45 split)
# Then make tons of BBoxes
# Calculate their IOC
# Calc if foreground/background
# Be able to split it 25/75

# Divide image RGB channels by the mean.

# Write out the data.
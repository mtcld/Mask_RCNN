
import xml.etree.ElementTree as ET
import json
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a',help='annotations path')
parser.add_argument('-i',help='image folder path')
parser.add_argument('-v',help='verbose mode',action='store_true',default=False)
parser.add_argument('-o',help='output filename',default='anno.json')
args = parser.parse_args()


'''
xml format:
annotation
  file_name
  folder
  source
    sourceImage
    sourceAnnotation
  object
    name
    detected
    verified
    occluded
    attributes
    date
    id
    polygon
      username
      pt
        x
        y
      pt
        x
        y

    object
      name
      detected
      verified
      occluded
      attributes
      date
      id
      polygon
        username
        pt
          x
          y
        pt
          x
          y


json format:
{ 'filename': '28503151_5b5b7ec140_b.jpg',
  'regions': {
      '0': {
          'region_attributes': {},
          'shape_attributes': {
              'all_points_x': [...],
              'all_points_y': [...],
              'name': 'polygon'}},
      ... more regions ...
  },
  'size': 100202
}'''

annopath= args.a
imagepath = args.i
output_path = args.o
verbose = args.v
data=[]#data will be write
xml_path=os.path.join(os.getcwd(),annopath)
for f in os.listdir(xml_path):

  if not f.endswith('.xml'):
    print('wrong extension')
    continue
  xml_file = os.path.join(xml_path, f)
  image_path= os.path.join(os.getcwd(),imagepath,f.split('.xml')[0] +'.jpg')
  try :
    file_size= os.path.getsize(image_path)
    tree = ET.parse(xml_file)
    root = tree.getroot()
    if root.tag != 'annotation':
      raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

    filename={}
    regions={}
    region={}
    region_count=0
    for elem in root:#loop through root
      if elem.tag=='filename':
        fname=elem.text
      if elem.tag=='object':
        points={}
        xs=[]
        ys=[]
        for polygon in elem:#check if this object is hotplate
          if polygon.text in ['broken', 'crack', 'carck', 'broke', 'crackk']:

            for polygon in elem:#loop inside object object

              if polygon.tag=='polygon':
                for pt in polygon:#loop inside polygon
                  for ax in pt:
                    if ax.tag=='x':
                      xs.append(ax.text)
                    if ax.tag=='y':
                      ys.append(ax.text)
                  points['all_points_x']=xs
                  points['all_points_y']=ys
            attr={}
            attr['region_attributes']={}
            attr['shape_attributes']=points

            regions[region_count]=attr
            region_count+=1

            filename['filename'] = fname
            filename['size'] = file_size
            filename['regions']=regions
            data.append(filename)
            if verbose :
              print(xml_file, polygon.text)
  except:
      print('aborted',xml_file)

with open(output_path,'w') as f:
  json.dump(data,f,indent=2)

print('file saved in:',os.path.abspath(output_path))

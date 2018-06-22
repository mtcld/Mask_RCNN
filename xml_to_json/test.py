import json
import xml.etree.ElementTree as ET
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-j',help='json path')
parser.add_argument('-i',help='image folder path')
parser.add_argument('-a',help='xml folder path')
args=parser.parse_args()


def compare_size(anno,fname,error_point):
  fpath = os.path.join(os.getcwd(),args.i,fname)+'.jpg'
  try :
    actual_size = os.path.getsize(fpath)

  except:
    print(fpath+" doesn't exist")
  if not anno['size'] == actual_size:
    error_point+=1
    print(fpath +' wrong size')




  return error_point

def check_polygon(anno,fpath,error_point):

  try:
    tree = ET.parse(fpath)
  except:
    print(fpath + " doesn't exist")
  root = tree.getroot()
  no_poly=0
  for elem in root:#loop through root
      if elem.tag=='object':
        points={}
        xs=[]
        ys=[]
        for idx,polygon in enumerate(elem):#loop inside object object
          if polygon.tag=='polygon':
            for pt in polygon:#loop inside polygon
              for ax in pt:
                if ax.tag=='x':
                  xs.append(ax.text)
                if ax.tag=='y':
                  ys.append(ax.text)
        try:
          if not xs == anno['regions'][str(no_poly)]['shape_attributes']['all_points_x']:
            error_point+=1
          if not ys == anno['regions'][str(no_poly)]['shape_attributes']['all_points_y']:
            error_point+=1
            print(fpath + ' wrong ys')
          no_poly+=1
        except:
          print(fpath+ " region doesn't exist")
  return error_point



def main():
  with open(args.j) as f:
      annos = json.load(f)
  listfile = os.listdir(os.path.abspath(args.a))
  error_point=0

  for idx,file in enumerate(listfile):
    fname = file.split('.xml')[0]
    if annos[idx]['filename']==fname+'.jpg':
      anno = annos[idx]
      fpath = os.path.join(os.getcwd(),args.a,fname) +'.xml'
      error_point+= compare_size(anno,fname,error_point)
      error_point+= check_polygon(anno,fpath,error_point)


  if error_point> 0 :
    print(fname)
  else:
    print('Everything is perfect')




if __name__ == '__main__':
  main()

#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import signal

def show_pic(image):
    plt.imshow(image)
    plt.show()
    
    
def normxcorr2(template, image):
  """Do normalized cross-correlation on grayscale images.

  When dealing with image boundaries, the "valid" style is used. Calculation
  is performed only at locations where the template is fully inside the search
  image.

  Args:
  - template (2D float array): Grayscale template image.
  - image (2D float array): Grayscale search image.

  Return:
  - scores (2D float array): Heat map of matching scores.
  """
  #TODO
  #print(image.shape)
  t_rows , t_cols = template.shape
  img_rows , img_cols = image.shape
  
  
  #t_sum = np.zeros( template.shape )
  kernel = np.ones( template.shape )
  img_sum = np.zeros( image.shape )
  
  norm_image = (image - np.mean(image))/np.std(image)
  norm_template = (template - np.mean(template))/np.std(template)
  
  t_sum = np.sum( norm_template * norm_template)
  image_sum = norm_image * norm_image
  
  scores = signal.correlate2d(norm_image, norm_template, mode='valid') 
  scores = scores /np.sqrt( t_sum * signal.correlate2d(image_sum , kernel, mode = 'valid') )
  '''
  for i in range( 0 , img_rows - t_rows+1):
      for j in range( 0, img_cols - t_cols+1):
          
          for row in range( 0 , t_rows):
              for col in range(0 , t_cols):
                  scores[i,j] = scores[i,j] + np.float(template[row,col]) * np.float(image[ i + row ,j + col ])
                  img_sum[i,j] = img_sum[i,j] + np.float(image[ i + row ,j + col ]) * np.float(image[ i + row ,j + col])
          scores[i,j] = scores[i,j]/( np.sqrt(t_sum) * np.sqrt(img_sum[i,j]) )
  '''  
  return scores

def draw_match_image(image , coords , rows , cols):
  match_image = image.copy()
  for pos in coords:
      y = [ pos[0] , pos[0]+rows ]
      x =[ pos[1] , pos[1]+cols ]
     
      pt1 = (x[0],y[0])
      pt2 = (x[1],y[0])
      cv2.line( match_image, pt1, pt2, (0, 255, 0), 2)

      pt1 = (x[0],y[0])
      pt2 = (x[0],y[1])
      cv2.line( match_image, pt1, pt2, (0, 255, 0), 2)
      
      pt1 = (x[1],y[1])
      pt2 = (x[1],y[0])
      cv2.line( match_image, pt1, pt2, (0, 255, 0), 2)

      pt1 = (x[1],y[1])
      pt2 = (x[0],y[1])
      cv2.line( match_image, pt1, pt2, (0, 255, 0), 2)

  return match_image

def find_matches(template, image, thresh=None):
  """Find template in image using normalized cross-correlation.

  Args:
  - template (3D uint8 array): BGR template image.
  - image (3D uint8 array): BGR search image.

  Return:
  - coords (2-tuple or list of 2-tuples): When `thresh` is None, find the best
      match and return the (x, y) coordinates of the upper left corner of the
      matched region in the original image. When `thresh` is given (and valid),
      find all matches above the threshold and return a list of (x, y)
      coordinates.
  - match_image (3D uint8 array): A copy of the original search images where
      all matched regions are marked.
  """
  t_rows , t_cols , channels = template.shape
  img_rows , img_cols, channels = image.shape
  scores = np.zeros( (img_rows , img_cols) )
  
  gray_scale_image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
  gray_scale_template = cv2.cvtColor(template , cv2.COLOR_BGR2GRAY)
  
  scores =  normxcorr2(gray_scale_template , gray_scale_image)
  #print(np.max(scores))
  #scores = scores/np.max(scores)
  coords = []
  if(thresh == None):
      pos = np.unravel_index(np.argmax(scores) , scores.shape)
      coords.append(pos)
  else:
      row , col = np.where(scores > thresh)
      leng = len(row)
      for i in range(leng):
          coords.append((row[i],col[i]))
  #print(coords)
  
  match_image = np.uint8( draw_match_image(image, coords , t_rows , t_cols) )

  return coords , match_image


def main(argv):
  #print("?????????????????????")
  template_img_name = argv[0]
  #template_img_name = 'face'
  search_img_name = argv[1]
  #search_img_name = 'king'
  template_img = cv2.imread("data/" + template_img_name + ".png", cv2.IMREAD_COLOR)
  search_img = cv2.imread("data/" + search_img_name + ".png", cv2.IMREAD_COLOR)
  #
  coords,match_image = find_matches(template_img, search_img)

  cv2.imwrite("output/" + search_img_name + ".png", match_image)


if __name__ == "__main__":
  main(sys.argv[1:])

# example usage: python p4.py face king
# expected results can be seen here: https://hackmd.io/toS9iEujTtG2rPoxAdPk8A?view
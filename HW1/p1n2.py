#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import math
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from collections import Counter

def draw_circle_attributes(image, attribute_list):
    attributed_image = image.copy()
    if attribute_list is not None:
        for circles in attribute_list:
            for circle in circles:
                center_x = (int)(circle["center"]["x"])
                center_y = (int)(circle["center"]["y"])

                center = (center_x, center_y)
                # circle center
                cv2.circle(attributed_image, center, 1, (255, 255, 255), 3)
                # circle outline
                radius = (int)(circle["radius"])
                cv2.circle(attributed_image, center, radius, (255, 0, 255), 3)

           
    #show_pic(attributed_image)
    return attributed_image

def draw_edge_attributes(image, attribute_list):
    attributed_image = image.copy()
    if attribute_list is not None:
        for lines in attribute_list:
            for line in lines:
                angle = (float)(line["angle"])
                distance = (float)(line["distance"])

                a = np.cos(angle)
                b = np.sin(angle)
                x0 = a * distance
                y0 = b * distance
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

                cv2.line(
                    attributed_image,
                    pt1,
                    pt2,
                    (0, 255, 0),
                    2,
                )
    #show_pic(attributed_image)
    return attributed_image
def binarize(gray_image, thresh_val):
    """ Function to threshold grayscale image to binary
        Sets all pixels lower than threshold to 0, else 255

        Args:
        - gray_image: grayscale image as an array
        - thresh_val: threshold value to compare brightness with

        Return:
        - binary_image: the thresholded image
    """
    # TODO: 255 if intensity >= thresh_val else 0
    
    binary_image = np.zeros((gray_image.shape) , dtype = np.uint8 )
    binary_image[ gray_image >= thresh_val ] = 255

    return binary_image

def label(binary_image):
    """ Function to labeled components in a binary image
        Uses a sequential labeling algorithm

        Args:
        - binary_image: binary image with multiple components to label

        Return:
        - lab_im: binary image with grayscale level as label of component
    """

    _ , lab_im = cv2.connectedComponents(binary_image, connectivity=4, ltype=cv2.CV_32S)
    lab_im = lab_im / np.max(lab_im)
    lab_im = np.uint8( lab_im * 255 )
    #plt.imshow(lab_im , cmap = 'gray')
    return lab_im


def get_attribute(labeled_image):
    """ Function to get the attributes of each component of the image
        Calculates the position, orientation, and roundedness

        Args:
        - labeled_image: image file with labeled components

        Return:
        - attribute_list: a list of the aforementioned attributes
    """
    # TODO
    labels = np.unique(labeled_image)
    tmp_image = labeled_image.copy()
    attribute_list = []
    #vec = np.array([1,2,3,4])
    #vec = vec*vec
    #print(vec)
    for i in range(1,len(labels)):
        #calculate position
        cols , rows = np.where(labeled_image == labels[i] )
        cols = np.array(cols,dtype = np.float)
        rows = np.array(rows,dtype = np.float)
        
        center_x = np.mean(rows)  
        center_y = np.mean(cols) 
        a = np.sum( (rows - center_x)**2 )
        b = 2 * np.sum((cols - center_y)*(rows - center_x) )
        c = np.sum( (cols - center_y)**2 )
        #print(a ,b, c)
        tmp = a-c
        if(np.abs(tmp)<1e-6): tmp = 1e-6
        theta1 = math.atan( b/tmp  ) / 2.0
        if(theta1 < 0):
            theta1 = theta1 + np.pi/2
            
        theta2 = theta1 + np.pi/2
      
        theta_min = theta1
        theta_max = theta2

        E_min = a * (math.sin(theta_min)*math.sin(theta_min)) - b * (math.sin(theta_min) *math.cos(theta_min)) + c * (math.cos(theta_min)*math.cos(theta_min))
        E_max = a * (math.sin(theta_max)*math.sin(theta_max)) - b * (math.sin(theta_max) *math.cos(theta_max)) + c * (math.cos(theta_max)*math.cos(theta_max))
        #print("##################")
        #print(E_min , E_max)
        #print(theta_min , theta_max)
        
        if(E_min > E_max):
            E_min , E_max = E_max , E_min
            theta_min , theta_max = theta_max , theta_min
        theta_min = -1*theta_min
        if(theta_min <0): theta_min = theta_min + np.pi*2
        if(np.abs(E_max )<1e-8): E_max = 1e-8
        attribute = {'position': {'x':center_x , 'y':labeled_image.shape[0] - center_y }, 'orientation': theta_min, 'roundedness': E_min / E_max}
        attribute_list.append(attribute)
    return attribute_list

def draw_attributes(image, attribute_list):
    num_row = image.shape[0]
    attributed_image = image.copy()
    if attribute_list is not None:
        for attribute in attribute_list:
            center_x = (int)(attribute["position"]["x"])
            center_y = (int)(attribute["position"]["y"])
            slope = np.tan(attribute["orientation"])

            cv2.circle(attributed_image, (center_x, num_row - center_y), 2, (0, 255, 0), 2)
            cv2.line(
                        attributed_image,
                        (center_x, num_row - center_y),
                        (center_x + 20, int(20 * (-slope) + num_row - center_y)),
                        (0, 255, 0),
                        2,
                    )
            cv2.line(
                        attributed_image,
                        (center_x, num_row - center_y),
                        (center_x - 20, int(-20 * (-slope) + num_row - center_y)),
                        (0, 255, 0),
                        2,
                    )

        
    #plt.imshow(attributed_image)
    #plt.show()
    return attributed_image

def non_maximal_suppression(grad_mag , grad_x , grad_y):
    image_row, image_col = grad_mag.shape

    output = np.zeros(grad_mag.shape)

    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction =  np.arctan(grad_y[row,col] / np.maximum(grad_x[row,col] , 1e-8) )  

            if (0 <= direction < np.pi / 8) or (15 * np.pi / 8 <= direction <= 2 * np.pi):
                before_pixel = grad_mag[row, col - 1]
                after_pixel = grad_mag[row, col + 1]

            elif (np.pi / 8 <= direction < 3 * np.pi / 8) or (9 * np.pi / 8 <= direction < 11 * np.pi / 8):
                before_pixel = grad_mag[row + 1, col - 1]
                after_pixel = grad_mag[row - 1, col + 1]

            elif (3 * np.pi / 8 <= direction < 5 * np.pi / 8) or (11 * np.pi / 8 <= direction < 13 * np.pi / 8):
                before_pixel = grad_mag[row - 1, col]
                after_pixel = grad_mag[row + 1, col]

            else:
                before_pixel = grad_mag[row - 1, col - 1]
                after_pixel = grad_mag[row + 1, col + 1]

            if grad_mag[row, col] >= before_pixel and grad_mag[row, col] >= after_pixel:
                output[row, col] = grad_mag[row, col]
    return output


def find_root(node, parent):
    if (parent[node] ==  node) :
        return node
    else:
        parent[node] = find_root(parent[node], parent)
        return parent[node]
    
def union(node1, node2, parent):
    # 分别找x,y节点的根
    root1, root2 = find_root(node1, parent), find_root(node2, parent)
    if (root1 == root2):
        return False
    if (root1 != root2):
        parent[root1] = root2
        return True


def hysteresis(grad_mag, lim1=100, lim2=200 , connectivity=8 ):
    rows, cols = grad_mag.shape
    if(lim1 > lim2):
        lim1,lim2 = lim2,lim1
        
    edge_group = (rows*cols)
    parent = [i for i in range(rows*cols + 1 )]
    dx = [-1,1,0,0 ,-1,-1,1,1]
    dy = [0,0,1,-1, -1,1,-1,1]
    
    for row in range(rows):
        for col in range(cols):
            #non-edge
            if(grad_mag[row][col] < lim1):
                continue
            #number of the node
            node1 = row * cols + col
            #edge
            if(grad_mag[row][col] > lim2):
                union(node1 , edge_group ,parent)
            
            for orient in range(connectivity):
                new_row = row+dx[orient]
                new_col = col+dy[orient]
                if(new_row < 0 or new_row >= rows):
                    continue
                if(new_col < 0 or new_col >= cols):
                    continue
                if(grad_mag[new_row][new_col] < lim1):
                    continue
                    
                node2 = new_row * cols +new_col    
                union(node1,node2,parent)
                
    output = np.zeros(grad_mag.shape)
    for row in range(rows):
        for col in range(cols):
            node1 = row*cols + col
            node2 = edge_group
            root1, root2 = find_root(node1, parent), find_root(node2, parent)
            if (root1 == root2):
                output[row][col] = 255
    return output

def sep_ind_edge(edge_image, labeled_image, label, connectivity=8 ):
    
    rows, cols = edge_image.shape
    edge_group = (rows*cols)
    parent = [i for i in range(rows*cols + 1 )]
    dx = [-1,1,0,0 ,-1,-1,1,1]
    dy = [0,0,1,-1, -1,1,-1,1]
    obj_image = np.zeros(edge_image.shape)
    obj_image[ labeled_image == label ] = 1
    
    #obj_image is pixels that are on edge and parts of object
    obj_image = obj_image * edge_image
    
    for row in range(rows):
        for col in range(cols):
            #non-edge
            if(edge_image[row][col]==0):
                continue
            #number of the node
            node1 = row * cols + col
            #edge
            if(edge_image[row][col] == 255 and obj_image[row][col] == 255 ):
                union(node1 , edge_group ,parent)
            
            for orient in range(connectivity):
                new_row = row+dx[orient]
                new_col = col+dy[orient]
                if(new_row < 0 or new_row >= rows):
                    continue
                if(new_col < 0 or new_col >= cols):
                    continue
                    
                if(edge_image[new_row][new_col] == 0):
                    continue
                    
                node2 = new_row * cols +new_col    
                union(node1,node2,parent)
                
    output = np.zeros(edge_image.shape)
    for row in range(rows):
        for col in range(cols):
            node1 = row*cols + col
            node2 = edge_group
            root1, root2 = find_root(node1, parent), find_root(node2, parent)
            if (root1 == root2):
                output[row][col] = 255
    return output

def detect_edges(image, sigma, threshold):
    """Find edge points in a grayscale image.

      Args:
      - image (2D uint8 array): A grayscale image.

      Return:
        - edge_image (2D binary image): each location indicates whether it belongs to an edge or not
    """
    ############################################################################
    #image = gaussian_filter(image,sigma = lim)
    image = binarize(image , 128)
    grad_x = gaussian_filter1d(image, sigma = sigma , axis = 0, order = 1)
    grad_y = gaussian_filter1d(image, sigma = sigma , axis = 1, order = 1)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y )
    grad_mag = non_maximal_suppression(grad_mag , grad_x , grad_y)
    # Remove the image borders
    grad_mag[:3] = 0
    grad_mag[-3:] = 0
    grad_mag[:, :3] = 0
    grad_mag[:, -3:] = 0
    
    #normalizing
    #print(np.max(grad_mag))
    grad_mag = ( grad_mag / np.max(grad_mag) ) *255
    
    edge_image = np.zeros(grad_mag.shape,dtype = np.uint8)
    edge_image [ grad_mag >= threshold*255 ] = 255;
    #edge_image = hysteresis( grad_mag,threshold*255/2.0,threshold*255,4)
    #plt.imshow(edge_image , cmap = 'gray')
    #plt.colorbar()
    #plt.show()
    
    edge_image2 = cv2.Canny(image , 100 ,200 )
    edge_image2 [ edge_image2 >=200] = 255
    #plt.imshow(edge_image2 , cmap = 'gray')
    #plt.show() 
    
    #edge_image = hysteresis( grad_mag,100,150,8)
    #plt.imshow(edge_image , cmap = 'gray')
    #plt.colorbar()
    #plt.show()
    
    edge_image[:20] = 0
    edge_image[-20:] = 0
    edge_image[:, :20] = 0
    edge_image[:, -20:] = 0
    
    return edge_image

def show_pic(image):
    plt.imshow(image,cmap='gray')
    plt.show()

def calculate_length(theta , rho , image ,lim):
    #show_pic(image)
    cnt = 0
    rows, cols = image.shape
    tmp_image = image.copy()

    theta = -1*theta
    a = np.cos(theta)
    b = np.sin(theta)

    for row in range (rows):
        for col in range(cols):
            if(image[row][col] == 0):
                continue
            val = col * a - row * b - rho  
            if(np.abs(val) < np.abs(lim)):
                cnt = cnt+1
    
    #print(cnt ,lim, a, b)
    return cnt
def find_most_common(sub_matrix):
    count = np.zeros((256))
    labels =np.unique(sub_matrix)
    for i in labels:
        if(i == 0 ): continue
        x,y =np.where(sub_matrix ==i)
        count[i] = len(x)
    
    output = np.argmax(count)
    return output
    
def get_circle_attribute(labeled_image, gray_image):
    attribute_list = []

    #initialization
    #print('############################################# circle')
    #plt.imshow(gray_image,cmap = 'gray')
    #plt.show()
    
    hough_image = np.zeros(gray_image.shape)
    labels = np.unique(labeled_image)
    
    img = gray_image.copy()
    
    rows,cols = hough_image.shape
    
    tmp_circles = []
        
    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT,dp=1,minDist =20,param1=50,param2=60,minRadius=5,maxRadius=350)
    
    '''
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0,:]:
            circle_attributes = {'center': {'x':circle[0] ,'y':circle[1]} ,'radius':circle[2] }
            tmp_circles.append(circle_attributes)
                
    attribute_list.append(tmp_circles)
    '''
    if circles is not None:
            circles = np.int16(np.around(circles))
            
    for i in range(1,len(labels)) :
        
        tmp_circles=[]
        if circles is not None:
            for circle in circles[0,:]:
                ### to decide which label it belongs
                flag = False
                x_min = np.min([ np.max([circle[0] - circle[2],0]) , rows])
                y_min = np.min([ np.max([circle[1] - circle[2],0]) , cols])
                
                x_max = np.min([ np.max([circle[0] + circle[2],0]) , rows])
                y_max = np.min([ np.max([circle[1] + circle[2],0]) , cols])
                
                sub_matrix = labeled_image[y_min:y_max , x_min:x_max]
                #count = np.bincount(sub_matrix)
                #print(np.max(sub_matrix),np.min(sub_matrix))
                tmp_label = find_most_common(sub_matrix)
                #tmp_label = Counter(sub_matrix).most_common(1)[0][0]
                if(tmp_label == labels[i]):
                    circle_attributes = {'center': {'x':circle[0] ,'y':circle[1]} ,'radius':circle[2] }
                    tmp_circles.append(circle_attributes)
        attribute_list.append(tmp_circles)
        #plt.imshow(img , cmap='gray')
        #plt.show()

    return attribute_list

def linesort(a):
    leng,b,c = a.shape
    lines = []
    for k in range(leng):
        lines.append([a[k][0][1], a[k][0][0]])

    lines.sort()
    return lines

def get_edge_attribute(labeled_image, edge_image):
    '''
      Function to get the attributes of each edge of the image
            Calculates the angle, distance from the origin and length in pixels
      Args:
        labeled_image: binary image with grayscale level as label of component
        edge_image (2D binary image): each location indicates whether it belongs to an edge or not

      Returns:
         attribute_list: a list of list [[dict()]]. For example, [lines1, lines2,...],
         where lines1 is a list and it contains lines for the first object of attribute_list in part 1.
         Each item of lines 1 is a line, i.e., a dictionary containing keys with angle, distance, length.
         You should associate objects in part 1 and lines in part 2 by putting the attribute lists in same order.
         Note that votes in HoughLines opencv-python is not longer available since 2015. You will need to compute the length yourself.
    '''
  # TODO
    
    #show_pic(labeled_image)
    #show_pic(edge_image)
    
    attribute_list = []

    #initialization
    lines = []
    edge_image[:20] = 0
    edge_image[-20:] = 0
    edge_image[:, :20] = 0
    edge_image[:, -20:] = 0
    hough_image = np.zeros(edge_image.shape)
    labels = np.unique(labeled_image)
    
    img = edge_image.copy()
    
    for i in range(1,len(labels)) :
        hough_image = np.uint8(sep_ind_edge(edge_image, labeled_image,labels[i]))
        tmp_image = hough_image.copy()

        #show_pic(tmp_image)
        tmp_lines = []
        lines = cv2.HoughLines(hough_image, 1 , np.pi/40 , 30)
        #lines = cv2.HoughLines(hough_image, 1 , np.pi/45 , 30)
        if lines is not None:
            lines = linesort(lines)
            lines_len =len(lines)
            for k in range(lines_len):
                line = lines[k]
                rho = line[1]
                theta = line[0]
                length =  calculate_length(theta , rho , tmp_image,0.8)
                if(length < 30): continue
                line_attribute = {'angle': theta , 'distance': rho, 'length':length }
                leng = len(tmp_lines)
                #if(leng == 0):
                #    tmp_lines.append(line_attribute)
                flag_add = True;
                if(leng>0):
                    for i in range(leng):
                        if ( np.abs(tmp_lines[i]['angle']-theta) < np.pi/8.0 ):
                            if(np.abs(tmp_lines[i]['distance']- rho) <= 100.0):
                                flag_add = False
                                if(length >= tmp_lines[i]['length']):
                                    tmp_lines[i] = line_attribute
                if(flag_add == True):
                    tmp_lines.append(line_attribute)
                    '''    
                    if ( np.abs(tmp_lines[leng-1]['angle']-theta) > np.pi/8.0 ):
                        tmp_lines.append(line_attribute)
                        
                    if ( np.abs(tmp_lines[leng-1]['angle']-theta) < np.pi/8.0 ):
                        if(np.abs(tmp_lines[leng-1]['distance']- rho) > 100.0):
                            tmp_lines.append(line_attribute)
                        if(np.abs(tmp_lines[leng-1]['distance']- rho) <= 100.0):
                            if(length >= tmp_lines[leng-1]['length']):
                                tmp_lines[leng-1] = line_attribute
                    '''
                #cv2.line(img,(x1,y1),(x2,y2),(255,255,255),2)
            
        attribute_list.append(tmp_lines)
        #plt.imshow(img , cmap='gray')
        #plt.show()
    
    return attribute_list



def main(argv):
    img_name = argv[0]
    #img_name = 'two_objects'
    #img_name = 'coins'
    #img_name = 'pacman'
    #img_name = 'many_objects_2'
    thresh_val = int(argv[1])
    #thresh_val = 128
    img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('output/' + img_name + "_gray.png", gray_image)

    # part 1
    binary_image = binarize(gray_image, thresh_val = thresh_val)
    #gray_image = binary_image
    cv2.imwrite('output/' + img_name + "_binary.png", binary_image)
  

    labeled_image = label(binary_image)
    cv2.imwrite('output/' + img_name + "_labeled.png", labeled_image)

    attribute_list = get_attribute(labeled_image)
    print('attribute list:')
    print(attribute_list)

    attributed_image = draw_attributes(img, attribute_list)
    cv2.imwrite("output/" + img_name + "_attributes.png", attributed_image)
###############################################################################

    # part 2
    # feel free to tune hyperparameters or use double-threshold
    edge_image = detect_edges(binary_image, sigma=0.3, threshold = 0.225 )
    #edge_image = detect_edges(gray_image, sigma=0.5, threshold = 0.45 , lim = 2)
    
    cv2.imwrite("output/" + img_name + "_edges.png", edge_image)

    edge_attribute_list = get_edge_attribute(labeled_image, edge_image)
    print('edge attribute list:')
    print(edge_attribute_list)

    attributed_edge_image = draw_edge_attributes(img, edge_attribute_list)
    cv2.imwrite("output/" + img_name + "_edge_attributes.png", attributed_edge_image)
    #show_pic(attributed_edge_image)
    # extra credits for part 2: show your circle attributes and plot circles
   
    #circle_attribute_list = get_circle_attribute(labeled_image, binary_image)
    circle_attribute_list = get_circle_attribute(labeled_image, gray_image)
    print('circle attribute list:')
    print(circle_attribute_list)
    attributed_circle_image = draw_circle_attributes(img, circle_attribute_list)
    cv2.imwrite("output/" + img_name + "_circle_attributes.png", attributed_circle_image)
    #plt.imshow(attributed_circle_image)
  # part 3


if __name__ == '__main__':
    main(sys.argv[1:])
# example usage: python p1n2.py two_objects 128
# expected results can be seen here: https://hackmd.io/toS9iEujTtG2rPoxAdPk8A?view
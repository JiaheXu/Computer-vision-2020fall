#!/usr/bin/env python3
import cv2
from p1n2 import *
import os
import matplotlib.pyplot as plt
import numpy as np

def seq_test_obj(test_object):
    object_attributes = []
    '''
    convert to gray_scale img
    '''

    

    rows,cols,channel = test_object.shape
    new_image = np.zeros((rows+60, cols+60,3),dtype = np.uint8 )
    new_image[30 : rows+30 , 30:cols+30 ,:] = test_object
    
    image = new_image.copy()
    img = new_image.copy()
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    thresh_val = 128
        
    binary_image = binarize(gray_image, thresh_val = thresh_val)
        #cv2.imwrite('output/' + img_name + "_binary.png", binary_image)
  

    labeled_image = label(binary_image)
    #plt.imshow(labeled_image,cmap='gray')
    #plt.show()
    #cv2.imwrite('output/' + img_name + "_labeled.png", labeled_image)

    attribute_list = get_attribute(labeled_image)
    #print('attribute list:')
    #print(attribute_list)


    edge_image = detect_edges(binary_image, sigma=0.3, threshold = 0.225 )
    #cv2.imwrite("output/" + img_name + "_edges.png", edge_image)


    edge_attribute_list = get_edge_attribute(labeled_image, edge_image)
    #print('edge attribute list:')
    #print(edge_attribute_list)
    #attributed_edge_image = draw_edge_attributes(img, edge_attribute_list)
    #cv2.imwrite("output/" + img_name + "_edge_attributes.png", attributed_edge_image)
    #plt.imshow(attributed_edge_image)
    #plt.show()
   
    circle_attribute_list = get_circle_attribute(labeled_image, gray_image)
    #print('circle attribute list:')
    #print(circle_attribute_list)
    #attributed_circle_image = draw_circle_attributes(img, circle_attribute_list)
    #plt.imshow(attributed_circle_image)
    #plt.show()
    #cv2.imwrite("output/" + img_name + "_circle_attributes.png", attributed_circle_image)
    object_attributes.append({
                                  'attribute_list': attribute_list,
                                  'edge_attribute_list': edge_attribute_list,
                                  'circle_attribute_list': circle_attribute_list
                                  })

    return object_attributes

'''

    Args:
        object_database: a list training images and each training image is stored as dictionary with keys name and image
        test_object: test image, a 2D unit8 array

    Returns:
        object_names: a list of filenames from object_database whose patterns match the test image
        You will need to use functions from p1n2.py
'''

def best_match(object_database, test_object):
    object_attributes = []
    for obj in object_database:
        #print("###################################")
        #print(obj['name'])
        '''
        convert to gray_scale img
        '''
        rows,cols,channel = obj['image'].shape

        
        new_image = np.zeros((rows+60, cols+60,channel),dtype = np.uint8 )
        new_image[30 : rows+30 , 30:cols+30 , :] = obj['image'] 
        obj['image'] =new_image
        image = obj['image'].copy()
        img =obj['image'].copy()
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh_val = 128
        
        binary_image = binarize(gray_image, thresh_val = thresh_val)
        #cv2.imwrite('output/' + img_name + "_binary.png", binary_image)
  

        labeled_image = label(binary_image)
        #cv2.imwrite('output/' + img_name + "_labeled.png", labeled_image)

        attribute_list = get_attribute(labeled_image)
        #print('attribute list:')
        #print(attribute_list)


        edge_image = detect_edges(binary_image, sigma=0.3, threshold = 0.225 )
        #cv2.imwrite("output/" + img_name + "_edges.png", edge_image)


        edge_attribute_list = get_edge_attribute(labeled_image, edge_image)
        #print('edge attribute list:')
        #print(edge_attribute_list)
        #attributed_edge_image = draw_edge_attributes(img, edge_attribute_list)
        #cv2.imwrite("output/" + img_name + "_edge_attributes.png", attributed_edge_image)
        #plt.imshow(attributed_edge_image)
        #plt.show()
   
        circle_attribute_list = get_circle_attribute(labeled_image, gray_image)
        #print('circle attribute list:')
        #print(circle_attribute_list)
        #attributed_circle_image = draw_circle_attributes(img, circle_attribute_list)
        #plt.imshow(attributed_circle_image)
        #plt.show()
        
        #cv2.imwrite("output/" + img_name + "_circle_attributes.png", attributed_circle_image)
        object_attributes.append({'name': obj['name'] ,
                                  'attribute_list': attribute_list,
                                  'edge_attribute_list': edge_attribute_list,
                                  'circle_attribute_list': circle_attribute_list
                                  })
    
    # TODO
    object_names = []
    test_object = seq_test_obj(test_object)
    test_objs = test_object[0]
    test_objs_num = len(test_objs['attribute_list'])
    train_objs_num = len(object_attributes)

    for test_num in range(test_objs_num):
        
        grade1 = 0.0
        grade2 = 0.0
        grade3 = 0.0
        grades = np.zeros((train_objs_num))
        
        for train_num in range(train_objs_num):
            obj = object_attributes[train_num]
            
            val1 = np.abs(obj[ 'attribute_list'][0]['roundedness'] - test_objs['attribute_list'][test_num]['roundedness'] )
            if (val1 <= 0.04 ):
                grade1 = 100
            if (0.04<val1 and val1<=0.08):
                grade1 = 80
            if (0.08<val1 and val1<=0.1):
                grade1 = 60
            if (0.1< val1 ): 
                grade1 = 0.0
            
            lines_num1 = len( obj['edge_attribute_list'][0])
            lines_num2 = len( test_objs['edge_attribute_list'][test_num])
            val2 = np.maximum(lines_num1 , lines_num2) / np.maximum ( 1e-4 , np.minimum(lines_num1 , lines_num2))
            if(lines_num1 == lines_num2):
                theta_dif1 = []
                theta_dif2 = []
                length_list1 = []
                length_list2 = []
                if(lines_num1 == 1):
                    grade2 = 100
                    
                if(lines_num1 >1):
                    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    #print(test_num , train_num)
                    #print(lines_num1)
                    #print(obj['edge_attribute_list'])
                    #print(test_objs['edge_attribute_list'])
                    
                    for line_num in range(0,lines_num1):
                        length = obj['edge_attribute_list'][0][line_num]['length']
                        length_list1.append(length)
                        
                    
                        length = test_objs['edge_attribute_list'][test_num][line_num]['length']
                        length_list2.append(length)
                        
                    length_list1.sort()
                    length_list2.sort()
                    list1 = np.array(length_list1, dtype = np.float)
                    list1 = list1/np.max(list1)
                    
                    list2 = np.array(length_list2, dtype = np.float)
                    list2 = list2/np.max(list2)
                    
                    error1 = np.sum( np.abs(list1 - list2) )
                    
                    for line_num in range(1,lines_num1):
                        theta1 = obj['edge_attribute_list'][0][line_num]['angle']
                        theta2 = obj['edge_attribute_list'][0][line_num-1]['angle']
                        
                        tmp_theta = np.abs(theta1-theta2)
                        if(tmp_theta > np.pi/2):
                            tmp_theta = tmp_theta - np.pi/2
                        theta_dif1.append( tmp_theta  )
                        
                    
                        theta1 = test_objs['edge_attribute_list'][test_num][line_num]['angle']
                        theta2 = test_objs['edge_attribute_list'][test_num][line_num-1]['angle']
                        
                        tmp_theta = np.abs(theta1-theta2)
                        if(tmp_theta > np.pi/2):
                            tmp_theta = tmp_theta - np.pi/2
                        theta_dif2.append( tmp_theta )
                    theta_dif1.sort()
                    theta_dif2.sort()
                    dif1 = np.array(theta_dif1, dtype = np.float)
                    dif2 = np.array(theta_dif2, dtype = np.float)
                    
                    error2 = np.sum(np.abs(dif1-dif2))
                    
                    
                    if(error2 <= np.pi/6.0 and error1<=0.2):
                        grade2 = 100
                    if(error2 > np.pi/6.0 or error1>0.2 ):
                        grade2 = 50
                        
            if (lines_num1 !=lines_num2 and 1.0<val2 <1.1):
                grade2 = 50
            if (1.1< val2 and val2<= 1.5):
                grade2 = 40
            if (1.5< val2 and val2<=5.0):
                grade2 = 20
            if (5 < val2):
                grade2 = 0
                
            circles_num1 = len( obj['circle_attribute_list'][0])
            circles_num2 = len( test_objs['circle_attribute_list'][test_num])
            grade3 = 0.0
            if(circles_num1 == circles_num2):
                grade3 = 100.0
            grades[ train_num ] = grade1*0.4 + grade2*0.4 + grade3*0.2
            
            #print(test_num ,train_num)
            #print(grade1,' ',grade2,' ',grade3)
        match = np.argmax(grades)
        object_names.append( object_attributes[match]['name'] )
    return object_names 

def main(argv):
    img_name = argv[0]
    #img_name = 'many_objects_1'
    #img_name = 'test1'
    test_img = cv2.imread('test/' + img_name + '.png', cv2.IMREAD_COLOR)

    train_im_names = os.listdir('train/')
    object_database = []
    
    for train_im_name in train_im_names:
        train_im = cv2.imread('train/' + train_im_name, cv2.IMREAD_COLOR)
        object_database.append({'name': train_im_name, 'image':train_im})
    object_names = best_match(object_database, test_img)
    print(object_names)


if __name__ == '__main__':
    main(sys.argv[1:])

# example usage: python p3.py test1
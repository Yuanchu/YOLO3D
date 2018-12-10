import cv2
import numpy as np
import pdb

if __name__ == '__main__' :
    target_box = np.load("target.npy")
    pred_box = np.load("all_boxes.npy")
    
    # Read source image.
    im_src = cv2.imread('cloud.png')
    # Four corners of the book in source image
    pts_src = np.array(((466,90), (560,90), (466,352),(560,352)), dtype=np.float32)
 
    # Read destination image.
    im_dst = cv2.imread('road.png')
    
    # Four corners of the book in destination image.
    pts_dst = np.array(((1188,400), (37,400), (687,191), (571,190)), dtype=np.float32)

    # Calculate Homography
    h = cv2.getPerspectiveTransform(pts_src, pts_dst)

    blank_img = np.zeros((im_src.shape[0], im_src.shape[1],3), np.uint8)
    blank_img.fill(255)
    
    for i in range(0,5):
        pred_img_y = int(pred_box[i][0]*1024.0 / 32.0)   # 32 cell = 1024 pixels   
        pred_img_x = int(pred_box[i][1]*512.0 / 16.0)    # 16 cell = 512 pixels 
        pred_img_width  = int(pred_box[i][2]*1024.0 / 32.0)   # 32 cell = 1024 pixels   
        pred_img_height = int(pred_box[i][3]*512.0 / 16.0)    # 16 cell = 512 pixels 
        x1 = int(pred_img_y-pred_img_width/2)
        y1 = int(pred_img_x-pred_img_height/2)
        x2 = int(pred_img_y+pred_img_width/2)
        y2 = int(pred_img_x+pred_img_height/2)

        print((x1,y1),(x2,y2))

        cv2.rectangle(blank_img, (x1,y1),(x2,y2),(0,0,255),1)
    

    for j in range(5):
        img_y = int(target_box[j][1] * 1024.0)   # 32 cell = 1024 pixels   
        img_x = int(target_box[j][2] * 512.0)    # 16 cell = 512 pixels 
        img_width  = int(target_box[j][3] * 1024.0)   # 32 cell = 1024 pixels   
        img_height = int(target_box[j][4] * 512.0)    # 16 cell = 512 pixels 
        rect_top1 = int(img_y - img_width / 2)
        rect_top2 = int(img_x - img_height / 2)
        rect_bottom1 = int(img_y + img_width / 2)
        rect_bottom2 = int(img_x + img_height / 2)
        cv2.rectangle(blank_img, (rect_top1, rect_top2), (rect_bottom1,rect_bottom2), (255, 0, 0), 1)
    
    warped = cv2.warpPerspective(blank_img, h, (im_dst.shape[1], im_dst.shape[0]))             
    
    for i in range(warped.shape[0]):                                        
        for j in range(warped.shape[1]):                                 
            pixel = warped[i][j]                                         
            if list(pixel) == [255,255,255]:
                continue
            if pixel[2] == 255:
                im_dst[i][j][:] = [0,0,255]
                tmp = int(i - 10*abs(167-j)/167)
                im_dst[tmp][j][:]=[0,0,255]
            elif pixel[0] == 255:                             
                im_dst[i][j][:] = [255,0,0] 

    cv2.imshow("blank", blank_img)
    cv2.imshow("warped", warped)
    cv2.imshow("dest", im_dst)
    cv2.waitKey(0)

    

'''
        original = np.array([((x1,y1),(x2,y2))],dtype=np.float32)
        import pdb; pdb.set_trace()
        converted = cv2.warpPerspective(original, h, (im_dst.shape[0], im_dst.shape[1]))
OA        
        import pdb; pdb.set_trace()

    #    trans_top1, trans_top2 = transform_point(rect_top1, rect_top2, h, im_src, im_dst.shape[1], im_dst.shape[0])
    #    trans_bot1, trans_bot2 = transform_point(rect_bottom1, rect_bottom2, h, im_src, im_dst.shape[1], im_dst.shape[0])
        
        #cv2.rectangle(im_src, (x1,y1),(x2,y2),(0,255,0),1)
        # Warp source image to destination based on homography
        im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
        cv2.rectangle(im_out, (converted[0][0], converted[0][1]), (converted[1][0], converted[1][1]), (0, 255, 0), 1)

        cv2.imshow("cc",im_out) 
        cv2.waitKey(0)

OA
  OBOA  for i in range(im_out.shape[0]):
vOBOA        for j in range(im_out.shape[1]):
            pixel = im_out[i][j]
            #print(pixel)
            if pixel[2] > 220 and pixel[1] < 50:
                im_dst[i][j][:] = [0,0,255]
            elif pixel[0] > 230 and pixel[1] < 50:
                im_dst[i][j][:] = [255,0,0]
    #import pdb;pdb.set_trace()

    
        # Display images
        # cv2.imshow("Warped Source Image", im_out)
        
        cv2.imshow("aa",im_src) 
        cv2.imshow("bb",im_dst)
        cv2.imshow("cc",im_out)
        cv2.waitKey(0)
'''

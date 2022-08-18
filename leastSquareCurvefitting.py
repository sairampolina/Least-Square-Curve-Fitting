"""
Author: Sairam Polina (sairamp@umd.edu), 2022
Brief: Code to fit Least Square Curve,for the ball in projectile motion in two different videos.
Date: 3rd February, 2022
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

 
def ballcenter_coordinates(frame):
    
    """
   Return the center of the red blob in cartesian co-ordinate system
   

   Parameters
   ----------
    frame: np.array
       Input frame of video
       
   Returns
   -------
   x,y : ints
     co-ordinate of the center in cartesian co-ordinate system.

    """
    
    red_array=frame[:,:,2]
    
    
    indices=np.where(red_array<210)
    col_indices=list(indices[1])
    col_min=min(col_indices) #min of column indices
    col_min_index=col_indices.index(col_min)
    # print(col_min_index)
    x_min=indices[0][col_min_index]
    
    col_max=max(col_indices) #max of column indices
    col_max_index=col_indices.index(col_max)
    x_max=indices[0][col_max_index]
   
    # calculating x,y wrt to cartesian-coordinate system
    y=((frame.shape[0]-x_min)+(frame.shape[0]-x_min))/2
    x=(col_min+col_max)/2
    
    return x,y


def plot_datapoints(st):
    
    """
    Return the co-ordinates of center of the red bolb in each frame
    
    Plots the datapoints i.e the center of the red bolb

   Parameters
   ----------
   st : string
      name of video file as a string  

   Returns
   -------
   Return the co-ordinates of center of the red bolb in each frame

   """
    
    #path
    video_path="./data/"+st
    
    #creating video object
    vobj=cv2.VideoCapture(video_path)
    
    
    
    coordinates_list=[[],[]]
    
    while(vobj.isOpened()):
        ret,frame=vobj.read()
        if ret==True:
            
            # cv2.imshow("Video",frame)
            x, y = ballcenter_coordinates(frame)
            coordinates_list[0].append(x)
            coordinates_list[1].append(y)
            # print(coordinates_list)
            cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
        else:
            break
        
    print("plotting datapoints of "+ st+"\n")
   
    fig,axes=plt.subplots()
    axes.plot(coordinates_list[0],coordinates_list[1],'ro')
    axes.set_title('Plot of Data Points')
    plt.show()
    
    vobj.release()
    cv2.destroyAllWindows()
    
    return coordinates_list



def plot_LS_curve(points):
      
    """
    Return the co-ordinates of center of the red bolb in each frame
    
    Plots the datapoints i.e the center of the red bolb

   Parameters
   ----------
   st : string
      name of video file as a string  

   Returns
   -------
   Return the co-ordinates of center of the red bolb in each frame

   """

    x=points[0][:]
    y=points[1][:]
    o=np.ones(len(x))
    # print(o)
    
    #calculating coefficient matrix
    
    X=np.stack((np.square(x),x,o)).T
    
    Y=np.vstack(y)
   
    step1=np.dot(X.transpose(),X)
    t2=np.dot(np.linalg.inv(step1),X.transpose())

    B=np.dot(t2,Y)
    

    
    # plotting least square curve
    y_curve=np.dot(X,B)
    
    print("plotting least square curve"+"\n")
    
    fig,axes=plt.subplots()
    axes.plot(x,y,'ro',x,y_curve,'-b')
    axes.set_title('Plot of Least Square Curve Fitting')          
    plt.show()



if __name__=="__main__":
    
    #plotting data of video1

    points_video1=plot_datapoints("ball_video1.mp4")


    # plotting data of video2

    points_video2=plot_datapoints("ball_video2.mp4")
       
    # plot Least Square Curve for video1
   
    plot_LS_curve(points_video1)
    
    # plot Least Square Curve for video2
    
    plot_LS_curve(points_video2)
    
    
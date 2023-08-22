import cv2
#from pupil_apriltags import Detector
from djitellopy import Tello
from ultralytics import YOLO
from time import sleep
import numpy as np

#isTello=True
isTello=False

# 0x01: u,d,l r     0x02: fwd, bwd
# 0x04: cw, ccw     0x07: all
moving=0x7

if isTello==False: 
    moving=0

obj_size=0 # sum of object top( or bottom) and left(or right) side length
qr_edge_size = 14000
qr_tolerance = 1400+1400 # qr_size/10

rc_params = [0,0,0,0] # l-r, b-f, d-u, yaw   -100~100
rc_default = 15

#isTakeoff = False

def TurningSign(frame, direction):  # direction : True-Left, False-right function draws the direction the drone will turn
    YEL = (0, 255, 255)
    radius = 36 #frame.shape[1]>>5                    
    axes = (radius, radius)  # The x,y radius must be the same if is a circle otherwise it is eclipse
    angle = 0  #results in a lot of spinning
    
    #center = (frame.shape[0]>>8 , frame.shape[1]-radius*2)
    arc_end_x, arc_end_y = frame.shape[1]>>1, frame.shape[0]-radius*4 
    #print(f'{arc_end_x}, {arc_end_y}, {arc_end_x}, {arc_end_y}')
    if direction: # left
        center = ( arc_end_x-radius , arc_end_y)
        startAngle = 0
        endAngle = 270
    else: # right
        center = ( arc_end_x+radius , arc_end_y)
        startAngle = -90
        endAngle = 180
    thickness = 4
    cv2.ellipse(frame, center, axes, angle, startAngle, endAngle, YEL, thickness)

    pts = np.array([[arc_end_x-(radius>>1),arc_end_y],[arc_end_x+(radius>>1),arc_end_y],[arc_end_x, arc_end_y-(radius>>1)]],
                   np.int32)
    cv2.fillPoly(frame, [pts], YEL)
    

def MovementSign(frame, moving, start_rect, end_rect, move_dir): #draws out the direction the drone will travel
    global rc_params, rc_default  # l-r, b-f, d-u, yaw   -100~100
    rc_params=[0,0,0,0]
    #rc_default=20
    YEL = (0, 255, 255)
    dist = 25
    mid_x=frame.shape[1]>>1
    mid_y=frame.shape[0]>>1
    dir_text = ''
    if move_dir["up"]:
        pt1=[mid_x, end_rect[1]]
        pts=np.array([pt1, [pt1[0]-dist,pt1[1]+dist], [pt1[0]+dist,pt1[1]+dist]])
        cv2.fillPoly(frame, [pts], YEL)
        # dir_text+="up"
        if moving&0x01:
            rc_params[2]=rc_default
        
    if move_dir["down"]:
        pt1=[mid_x, start_rect[1]]
        cv2.fillPoly(frame, [np.array([pt1, [pt1[0]-dist,pt1[1]-dist], [pt1[0]+dist,pt1[1]-dist]])], YEL)
        dir_text+="down"
        if moving&0x01:
            rc_params[2]=rc_default*(-1)
        
    if move_dir["left"]:
        pt1=[end_rect[0], mid_y]
        cv2.fillPoly(frame, [np.array([pt1, [pt1[0]+dist,pt1[1]-dist], [pt1[0]+dist, pt1[1]+dist]])], YEL)
        # dir_text+="left"
        if moving&0x01:
            rc_params[0]=rc_default*(-1)
        
    if move_dir["right"]:
        pt1=[start_rect[0], mid_y]
        cv2.fillPoly(frame,  [np.array([pt1, [pt1[0]-dist,pt1[1]-dist], [pt1[0]-dist, pt1[1]+dist]])], YEL)
        # dir_text+="right"
        if moving&0x01:
            rc_params[0]=rc_default
       
    if move_dir["forward"]:
        if moving&0x02:
            rc_params[1]=rc_default
        tri_len=60
        start_pt = [mid_x, tri_len]
        dir_text+="forward"
        cv2.fillPoly(frame, [np.array([start_pt, [start_pt[0]-tri_len, start_pt[1]+tri_len], 
                                      [start_pt[0]+tri_len, start_pt[1]+tri_len]])], YEL)
                
    if move_dir["backward"]:
        if moving&0x02:
            rc_params[1]=rc_default*(-1)
        tri_len=60
        start_pt = [mid_x, tri_len+tri_len]
        # dir_text+="backward"
        cv2.fillPoly(frame, 
                     [np.array([start_pt, [start_pt[0]-tri_len, start_pt[1]-tri_len], 
                                      [start_pt[0]+tri_len, start_pt[1]-tri_len]])], YEL)
    
    if move_dir["cw"]:
        if moving&0x04:
            rc_params[3]=rc_default  # 회전 cw (rc_parmas[3]) 
        #    rc_params[0]=rc_default*(-1) # Move left
        dir_text+="cw"
        TurningSign(frame, False)  # direction : Ture-Left, False-right
        
    if move_dir["ccw"]:  
        if moving&0x04:
            rc_params[3]=rc_default*(-1) # 회전 ccw
           # rc_params[0]=rc_default # Move Right
        dir_text+="ccw"
        TurningSign(frame, True)
        
    if isTello:
        str=f"rc {rc_params[0]} {rc_params[1]} {rc_params[2]} {rc_params[3]}"
        tello.send_command_without_return(str)
    frame=cv2.putText(frame, dir_text, (200,100), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (255, 0, 0), 2, cv2.LINE_AA)
    




distance = 450
model = YOLO('best.pt')

#tello.takeoff()
obj_center_prev=[-1, -1]
obj_center=[-1,-1] # [x,y] if it is -1 it is not on the screen


if isTello: #checks if the drone is active and creates the object for the rest of the code to reference
    tello = Tello() # if this is in the try it bugs out
    try:
        # Create Tello Object
        tello.connect()
        print(f"Battery Life Percentage: {tello.get_battery()}")
        #tello.reboot()
        sleep(1)

        #if tello.send_command_with_return("takeoff", 7) != 'ok':
        if moving!=0:
            tello.takeoff()
            print('takeoff')
            
        # Start the video Stream
        tello.streamon()
        print('tello init')
        sleep(1)
        
    except cv2.error as e:
        print('Tello Init Error!! : [{e}]')
        tello.streamoff()
        tello.end()
        sys.exit(1)


if isTello==False: #if drone is not found use back camera
    try:
        #cap = cv2.VideoCapture(0 , cv2.CAP_DSHOW) # Use This for Windows + Webcam
        #cap = cv2.VideoCapture(1) # SURFACE BACK CAM
        cap = cv2.VideoCapture(0) # USB OR FRONT CAM
        #cap = cv2.VideoCapture(f'rtsp://admin:admin@192.168.0.62:1935')  # IP Camera

        #cv2.namedWindow('Capturing', cv2.WINDOW_NORMAL)
        cap.set(cv2.CAP_PROP_FRAME_COUNT,5) # set 5 frame per Second

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    except cv2.error as e:
        print('CV Start Error!!')
        sys.exit(1)


""" at_detector = Detector(
   families="tag36h11",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
) """

count=0

while True: # main control loop for tracking and following april tag 
    move_dir = {"up":False,      "down":False, 
            "left":False,    "right":False, 
            "forward":False, "backward":False,
            "cw":False,      "ccw":False
           } 
    # If ESC is pressed then stop
    key = cv2.waitKey(1) & 0xff

    if key == 27: # ESC
        break
    elif key == ord(' '):
        print('space')
        if isTello and moving!=0:
            tello.takeoff()
        break
    elif key == ord('a') or key == ord('A'):
        move_dir["ccw"]=True
        print('ccw')
    elif key == ord('d') or key == ord('D'):
        move_dir["cw"]=True
        print('cw')
    elif key == ord('w') or key == ord('W'):
        move_dir["forward"]=True
        print('forward')
    elif key == ord('s') or key == ord('S'):
        move_dir["down"]=True
        print('descend')

    
    try:
        ret=False
        retry=0
        if isTello:  # Get the frame reader from Tello Cam
          while retry<5 and ret==False:
            frame = tello.get_frame_read().frame
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #frame = cv2.flip(tello.get_frame_read().frame, 1) # TTello is atomaticall flipped so april tag is detected
            if frame.any():
                ret=True
            else:
              retry+=1
              sleep(1)
        else: # get a frame from webCam
            ret, frame = cap.read() # Just getting actual Webcam resolution.
            
        if ret==False:
          print('openCV : Failed to capture!!!')
          tello.streamoff()
          tello.end()
          exit(1)
    except cv2.error as e:
        print(f'openCV reading Error {e}')
        exit(1)

    if isTello==False:
        frame = cv2.resize(frame, dsize=(1280,720), interpolation=cv2.INTER_CUBIC)
    
    results = model.predict(source=frame, conf=0.7, imgsz=960)
    f=frame
    
    cen_x, cen_y = f.shape[1]>>1, f.shape[0]>>1 # whole frame center
    adj_x, adj_y = 0, 0  # Tello camera look down. need rect move down.
    if isTello:
        adj_x=0 #(cen_x>>2)-48
        adj_y=50 #cen_y>>3

    x_range=70
    y_range=50
    rect_pt1 = (cen_x-x_range, cen_y-y_range-adj_y) #(cen_x-(cen_x>>2)-adj_x, cen_y-(cen_y>>2)-adj_y)
    rect_pt2 = (cen_x+x_range+adj_x, cen_y+y_range+adj_y)

    if len(results[0].boxes)==0:
        frame = cv2.rectangle(f, rect_pt1, rect_pt2,
                (0, 0, 255), 5)


    #grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #d=at_detector.detect(grey_frame)
    

    #if len(d)==0: # if april tag isnt found sleep for 0.5 seconds and look again
    if len(results[0].boxes)==0: # no detections
        obj_center=[-1, -1]
    #    if isTello:
    #        frame=cv2.flip(frame, 1)
            
        #cv2.imshow("Tello View", cv2.flip(grey_frame, 1))
        cv2.imshow("Tello View",frame)
        continue

    idx, highest_i, highest_v = 0, 0, 0.0
    for box in results[0].boxes:
        if box.conf.item()>highest_v:
            highest_v=box.conf.item()
            highest_i=idx
        idx+=1


    #int_rect = [int(f) for f in results[highest_i].boxes.xyxy.tolist()[0]]
    int_rect = [int(f) for f in results[0].boxes[highest_i].xyxy[0].tolist()]  # [ x1, y1, x2, y2]

    if len(results[0].boxes[highest_i]) == 0:
        
        frame=cv2.rectangle(frame, (int_rect[0], int_rect[1]), (int_rect[2], int_rect[3]), (0, 0, 255), 5)

    obj_width= int_rect[2]-int_rect[0] # x2-x1
    obj_height = int_rect[3]-int_rect[1] # y2-y1
    obj_center=[int_rect[0]+int(obj_width/2), int_rect[1]+int(obj_height/2)]
    frame=cv2.rectangle(frame, (int_rect[0], int_rect[1]), (int_rect[2], int_rect[3]), (255, 0, 0), 5)
    frame=cv2.putText(frame, f'{highest_v:.2f}', (int_rect[0], int_rect[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    frame=cv2.circle(frame, (obj_center[0], obj_center[1]), 5, (255,0,0), cv2.FILLED)
    
    
    """     
    for r in d: #loops through all the april tags and finds id 125
        if r.tag_id!=125: #if the tag_id is 125 continue
            print(f'tag_id: {r.tag_id}')
            center=[-1, -1]
            continue
        SortCorners(r.corners)
        (ptA, ptB, ptC, ptD) = ( (int(n[0]), int(n[1]) ) for n in r.corners )
        center=[int(frame.shape[1]-r.center[0]), int(r.center[1])] # since the camera will be flipped switch the corners
        
        # draw the bounding box of the AprilTag detection
        # (255, 0, 0) : BlUE
        cv2.line(frame, ptA, ptB, (255, 0, 0), 5) # BLUE   top-right to btm-right (will be flip L-R)
        cv2.line(frame, ptB, ptD, (0, 255, 0), 5) # GRN    btm-right to btm-left (will be flip)
        cv2.line(frame, ptA, ptC, (0, 0, 255), 5) # RED    top-right to top-left (will be flip)
        cv2.line(frame, ptC, ptD, (255, 0, 255), 5) #B+R   top-left  to btm-left (will be flip)
       
        up_edge= (ptC[0]-ptA[0])*(ptC[0]-ptA[0]) + (ptC[1]-ptA[1])*(ptC[1]-ptA[1]) #상  1m : 9000~10000
        dn_edge= (ptD[0]-ptB[0])*(ptD[0]-ptB[0]) + (ptD[1]-ptB[1])*(ptD[1]-ptB[1]) #하  1.5m : 4000
        
        left_edge = (ptB[0]-ptA[0])*(ptB[0]-ptA[0]) + (ptB[1]-ptA[1])*(ptB[1]-ptA[1]) #좌  
        right_edge= (ptD[0]-ptC[0])*(ptD[0]-ptC[0]) + (ptD[1]-ptC[1])*(ptD[1]-ptC[1]) #우
        
        
        #frame=cv2.putText(frame, f"T{up_edge}, B{dn_edge}, L{left_edge}, R{right_edge}", (100,200), cv2.FONT_HERSHEY_SIMPLEX,
        #                  1, (255,0,0), 2, cv2.LINE_AA)
     """

    f=frame
    #if isTello:
    #    f=cv2.flip(frame, 1)
    
    cen_x, cen_y = f.shape[1]>>1, f.shape[0]>>1 # whole frame center
    adj_x, adj_y = 0, 0  # Tello camera look down. need rect move down.
    if isTello:
        adj_x=0 #(cen_x>>2)-48
        adj_y=50 #cen_y>>3

    x_range=70
    y_range=50
    rect_pt1 = (cen_x-x_range, cen_y-y_range-adj_y) #(cen_x-(cen_x>>2)-adj_x, cen_y-(cen_y>>2)-adj_y)
    rect_pt2 = (cen_x+x_range+adj_x, cen_y+y_range+adj_y)
    f = cv2.rectangle(f, rect_pt1, rect_pt2,
            (0, 255, 255), 5)
    #r.center=(frame.shape[1]-r.center[0], r,center[1])
    
    dir_text=''
    if obj_center[0]<rect_pt1[0]:
        #move_dir["right"]=True
        #dir_text+='right '
        move_dir["ccw"]=True
        dir_text+='ccw '        
        
        
    if obj_center[0]>rect_pt2[0]:
        #move_dir["left"]=True
        #dir_text+='left '
        move_dir["cw"]=True
        dir_text+='cw '


    if obj_center[1]>rect_pt2[1]:
        move_dir["down"]=True
        dir_text+='down '
 

    if obj_width+obj_height < distance and move_dir["cw"]==False and move_dir["ccw"]==False:
        move_dir["forward"]=True
        dir_text+='forward'
    
    """ 
    if obj_width+obj_height > distance and count>100:
        tello.land()
        break
    """    
    count+=1
    
    """ 
    if obj_center[1]<rect_pt1[1]:
        move_dir["up"]=True
        dir_text+='up '

    if obj_center[1]>rect_pt2[1]:
        move_dir["down"]=True
        dir_text+='down '

    #up+dn edge가 left+right보다 짧아지면 회전을 함.
    #if the top and the bottom edge is shorter the the left and right edge then spin the drone respectivly
    if (left_edge+right_edge)-(up_edge+dn_edge) > qr_tolerance>>1:
        if right_edge>left_edge:
            move_dir["cw"]=True
            dir_text+='cw '
        elif left_edge>right_edge:
            move_dir["ccw"]=True
            dir_text+='ccw '
    
    # qr_edge_size = 14000
    # qr_tolerance = 1400*2 # qr_size/10
    if left_edge+right_edge > qr_edge_size + qr_tolerance :
        move_dir["backward"]=True
        dir_text+='backward'
        
    if left_edge+right_edge < qr_edge_size - qr_tolerance :
        move_dir["forward"]=True
        dir_text+='forward'
    """        
    # f=cv2.putText(f, dir_text, (100,200), cv2.FONT_HERSHEY_SIMPLEX,
    #                      2, (255, 0, 0), 2, cv2.LINE_AA)
    f=cv2.putText(f, f'{distance-obj_width-obj_height}', (100,200), cv2.FONT_HERSHEY_SIMPLEX,
                          2, (255, 0, 0), 2, cv2.LINE_AA)
    
    #TurningSign(f, left_edge<right_edge)
    # 0x01: up, dn, l, r
    # 0x02: fwd, bwd
    # 0x04: cw, ccw
    # 0x07: all
    MovementSign(f, moving, rect_pt1, rect_pt2, move_dir)
    
    cv2.imshow("Tello View", f)
    #sleep(0.25)
    
if isTello:
    tello.streamoff()
    tello.end()
else:
    cap.release()  # finish capturing

cv2.destroyWindow('Tello View')
cv2.destroyAllWindows()


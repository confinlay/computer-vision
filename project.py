"""Skeleton code for python script to process a video using OpenCV package

:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""
import cv2
import numpy as np

# Timestamp constants (only used for section 1)
g1 = 1590
g2 = 3010
g3 = 3750
g4 = 4510

b1 = 6090
b2 = 7500
b3 = 9180
b4 = 10740

last = 12500
morph = 15000



# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper

def openVideo(input_video_file, output_video_file):
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # saving output video as .mp4
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))
    return (cap, out)

def grayscale(frame, cap):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        
def objectGrabbing(mask):
    kernel = np.ones((41, 41), np.uint8)
    # Closing 
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Dialating
    mask_dilated = cv2.morphologyEx(mask_closed, cv2.MORPH_DILATE, kernel)
    return cv2.absdiff(mask, mask_dilated)

def createOverlay(frame, mask):
    # Green overlay
    overlay = np.zeros_like(frame, dtype=np.uint8)
    overlay[:,:,1] = mask  
    new_frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    return new_frame

def binaryColorGrab(H_up, H_low, S_up, S_low, V_up, V_low, frame):
    lower_color_bounds = np.array([H_low, S_low, V_low])
    upper_color_bounds = np.array([H_up, S_up, V_up])
   
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_color_bounds, upper_color_bounds)
    new_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return new_frame, mask

def applyHorizontalEdgeDetection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_horizontal = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobel_horizontal = cv2.convertScaleAbs(sobel_horizontal)
    sobel_color = cv2.cvtColor(abs_sobel_horizontal, cv2.COLOR_GRAY2BGR)
    return sobel_color

def applyVerticalEdgeDetection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_vertical = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    abs_sobel_vertical = cv2.convertScaleAbs(sobel_vertical)
    sobel_color = cv2.cvtColor(abs_sobel_vertical, cv2.COLOR_GRAY2BGR)
    return sobel_color

def applyTotalEdgeDetection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
    abs_sobel = cv2.convertScaleAbs(sobel)
    sobel_color = cv2.cvtColor(abs_sobel, cv2.COLOR_GRAY2BGR)
    return sobel_color

def visualizeEdges(original_frame, edges, color):
    _, binary_edges = cv2.threshold(edges, 10, 255, cv2.THRESH_BINARY)
    colored_edges = np.zeros_like(original_frame)
    colored_edges[:, :, color] = edges[:, :, 0] 
    overlay = cv2.addWeighted(original_frame, 1, colored_edges, 1, 0)
    return overlay

def applyHough(frame, param1, param2, minDist, minRadius, maxRadius):
    new_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0) 
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=minDist,
                       param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(new_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(new_frame, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    return new_frame

def padding(small_frame, full_frame):
    full_height, full_width = full_frame.shape[:2]
    small_height, small_width = small_frame.shape[:2]
    
    padding_top = padding_bottom = (full_height - small_height) // 2
    padding_left = padding_right = (full_width - small_width) // 2
    
    # For odd number differences
    if (full_height - small_height) % 2 != 0:
        padding_bottom += 1
    
    if (full_width - small_width) % 2 != 0:
        padding_right += 1
        
    padded_result = cv2.copyMakeBorder(small_frame, padding_top, padding_bottom,
                                    padding_left, padding_right, cv2.BORDER_CONSTANT,
                                    value=255)
    return padded_result

def templateMatch(frame, template):
    
    new_frame = frame.copy()
    gray_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_SQDIFF_NORMED)
    
    # Create heatmap
    graymap = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    logscale = np.log1p(graymap)
    final = cv2.normalize(logscale, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    
    resized_result = padding(final, frame)
    
    # Create rectangle
    w, h = gray_template.shape[::-1]
    
    # Best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    cv2.rectangle(new_frame, top_left, bottom_right, 255, 2)
    
    return resized_result, new_frame

def overlay_transparent(background, overlay, position):
        overlay_rgb = overlay[..., :3] 
        overlay_alpha = overlay[..., 3] / 255.0 

        x, y = position
        
        overlay_height, overlay_width = overlay.shape[:2]
        background_roi = background[y:y+overlay_height, x:x+overlay_width]
        
        x_end = min(x + overlay_width, background.shape[1])
        y_end = min(y + overlay_height, background.shape[0])
        x = max(x, 0)
        y = max(y, 0)

        background_roi = background[y:y_end, x:x_end]
    
        if x_end - x != overlay_width or y_end - y != overlay_height:
            overlay_rgb = cv2.resize(overlay_rgb, (x_end - x, y_end - y), interpolation=cv2.INTER_AREA)
            overlay_alpha = cv2.resize(overlay_alpha, (x_end - x, y_end - y), interpolation=cv2.INTER_AREA)
        
        blended_roi = overlay_rgb * overlay_alpha[..., None] + background_roi * (1 - overlay_alpha[..., None])
        
        background[y:y_end, x:x_end] = blended_roi.astype(np.uint8)




if __name__ == '__main__':
    
    input_video_file = "/Users/conor/Documents/College/Computer_Vision/IA1/last_vid.mp4"
    output_video_file = "output.mp4"
    template = cv2.imread("/Users/conor/Documents/College/Computer_Vision/IA1/eye.jpeg")
    
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    cap, out = openVideo(input_video_file, output_video_file)

    # To decide which code to use depending on video section
    part_one = True
    part_two = False
    part_three = False
    test = False

    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
          
            if part_one:
                if between(cap, g1, g2) or between(cap, g3, g4):
                    frame = grayscale(frame, cap)
                elif between(cap, b1, b2):
                    frame = cv2.GaussianBlur(frame, (11,11), 0)
                elif between(cap, b2, b3):
                    frame = cv2.GaussianBlur(frame, (31, 31), 0)
                elif between(cap,b3, b4):
                    frame = cv2.bilateralFilter(frame, 9, 100, 100)
                elif between(cap, b4, last):
                    frame = cv2.bilateralFilter(frame, 9, 500, 500)
                elif between(cap, last, 10000000):
                    frame, mask = binaryColorGrab(150, 20, 240, 20, 180, 50, frame)
                    if between(cap, morph, 1000000):
                        mask = objectGrabbing(mask)
                        frame = createOverlay(frame, mask)
            elif part_two:
                if between(cap,0, 2000):
                    edges = applyHorizontalEdgeDetection(frame)
                    frame = visualizeEdges(frame, edges, 0)
                elif between(cap, 2000, 3500):
                    edges = applyVerticalEdgeDetection(frame)
                    frame = visualizeEdges(frame, edges, 2)
                elif between(cap, 3500, 5000):
                    edges = applyTotalEdgeDetection(frame)
                    frame = visualizeEdges(frame, edges, 1)
                elif between(cap, 5000, 7500):
                    frame = applyHough(frame, 50, 30, 500, 20, 100)
                elif between(cap, 7500, 10000):
                    frame = applyHough(frame, 50, 30, 500, 20, 300)
                elif between(cap, 10000, 12500):
                    frame = applyHough(frame, 50, 30, 10, 20, 150)
                elif between(cap, 12500, 15000):
                    frame = applyHough(frame, 25, 15, 500, 20, 70)
                elif between(cap, 15000, 18000):
                    likelihood, box = templateMatch(frame, template)
                    frame = box
                elif between(cap, 18000, 1000000):
                    likelihood, box = templateMatch(frame, template)
                    # Convert grayscale result to BGR (color) if you want a colored overlay
                    frame = cv2.cvtColor(likelihood, cv2.COLOR_GRAY2BGR)
            
            elif part_three:
                if between(cap, 3000, 100000): 
                    overlay_image = cv2.imread('eyes.png', cv2.IMREAD_UNCHANGED)
                    original_overlay_height, original_overlay_width = overlay_image.shape[:2]
                    new_frame, mask = binaryColorGrab(106, 90, 240, 20, 180, 50, frame)
    
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)        
                    max_contour = max(contours, key=cv2.contourArea)
                    (x, y), radius = cv2.minEnclosingCircle(max_contour)
                    center = (int(x), int(y))
                    
                    # So eyes don't get too small
                    min_width = 150  
                    min_height = 150  

                    # Resize eyes
                    scale_factor = radius * 2 / original_overlay_width
                    new_width = int(original_overlay_width * scale_factor)
                    new_height = int(original_overlay_height * scale_factor)
                    

                    if new_width < min_width:
                        new_width = min_width
                        scale_factor = new_width / original_overlay_width
                        new_height = int(original_overlay_height * scale_factor)
                    
                    if new_height < min_height:
                        new_height = min_height
                        scale_factor = new_height / original_overlay_height
                        new_width = int(original_overlay_width * scale_factor)
                    
                    resized_overlay = cv2.resize(overlay_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    top_left_x = max(center[0] - new_width // 2, 0)
                    top_left_y = max(center[1] - new_height // 2, 0)
                    
                    top_left_x = min(top_left_x, frame.shape[1] - new_width)
                    top_left_y = min(top_left_y, frame.shape[0] - new_height)
                    
                    top_left = (top_left_x, top_left_y)

                    overlay_transparent(frame, resized_overlay, top_left)
                    
                    # Turn the rest of the fram black during certain intervals
                    if between(cap, 9050, 11060) or between(cap, 13470, 15300):
                        height, width = frame.shape[:2]
                        new_mask = np.zeros((height, width), dtype=np.uint8)
                        cv2.drawContours(new_mask, [max_contour], -1, color=255, thickness=-1)
                        
                        mask_3channel = cv2.cvtColor(new_mask, cv2.COLOR_GRAY2BGR)
                        isolated_image = cv2.bitwise_and(frame, mask_3channel)
                        frame = isolated_image
                            
        
            cv2.imshow('test', frame)                                               

            # write frame that you processed to output
            out.write(frame)
 
           #  Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
               break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and writing object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()


This was an individual assignment I completed for my Computer Vision class, which involved applying basic image processing techniques and object detection to create a one-minute video. The assignment was structured into three segments:

### 0-20 seconds
- The first part focused on basic image processing techniques. I alternated the video between color and grayscale, applied Gaussian and bi-lateral filters for blurring effects, and demonstrated object highlighting through binary frames in RGB and HSV color spaces, using thresholding and binary morphological operations.

### 20-40 seconds
- This segment concentrated on object detection. It featured the use of the Sobel edge detector for edge detection and the Hough transform for detecting circular shapes. The techniques were visualized by displaying the detected edges and circles with colorful contours overlaid on the original footage.

### 40-60 seconds
- The final section was open for creative exploration of advanced techniques related to object manipulation. In my project, this included detecting and tracking a colored object, overlaying images onto the moving object, and removing the background of the video.

The project utilized Python and the OpenCV library for processing the video, which was captured using standard video recording tools. The video was processed frame by frame and submitted in mp4 format, which can be seen below (compressed):

https://github.com/confinlay/computer-vision/assets/106957733/8947d7a8-2163-496d-a320-cf1df7aad588

The code can be found in project.py

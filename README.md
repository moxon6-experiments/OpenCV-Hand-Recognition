Based on original C++ Code From:
  https://github.com/jujojujo2003/OpenCVHandGuesture

Changed palm detection to maximum inscribed circle (improved accuracy)
Restructured into Python App using OpenCV 3.0


# Hand Detection Algorithm


## 1 Training

### 1.1 Background Removal

----



## 2 Detection

### 2.1 Foreground Detection

### 2.2 Contour Extraction

### 2.3 Foreground cleaning

### 2.4 Extract Convexity Defects

### 2.5 Determine Palm location and size

### 2.6 Determine finger points
- 2.6.1 Get All potential finger points
- 2.6.2 Select valid potential points
- 2.6.3 Merge fingertip points
 
### 2.7 Generate Hand using finger points and palm data

  
  

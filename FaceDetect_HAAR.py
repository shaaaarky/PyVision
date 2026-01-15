import cv2 as cv 

# Function to find all available camera indexes, we'll ask the user to pick one 
def FindCameraIndexes():
    IndexRange = 10
    # We find valid indexes by testing all in the range and seeing which ones are responding
    ValidIndexes = []
    for index in range(IndexRange):
        cap = cv.VideoCapture(index)    
        if cap.read()[0]:
            ValidIndexes.append(index)
            cap.release()
        index += 1

    return ValidIndexes

def GetDeviceProperties(cap):
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    return fps, width, height

def main():
    
    font = cv.FONT_HERSHEY_SIMPLEX 

    # We take the input of the camera 
    cap = cv.VideoCapture(0)

    # Show the properties 
    fps, width, height = GetDeviceProperties(cap)
    print(f"FPS: {fps} \nWidth: {width} \nHeight: {height}")

    # Load Haar cascade 
    face_cascade = cv.CascadeClassifier(
    'haarcascade_frontalface_default.xml'
    )


    while cap.isOpened(): 
        peopleInFrame = 0

        # Capture frame by frame 
        ret, frame = cap.read()

        # If the return value each time we call .read() isn't true, it means something went wrong. 
        if not ret: 
            print("Capture stream crashed\nExiting...")
            break
        
        # Convert to grayscale 
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            peopleInFrame += 1
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Add text to below the image 
            offset = 20 # Pixel offset from the square 
            
            cv.putText(frame, "Name", (x, y - offset), font, 3, (255, 255, 255), 3, cv.LINE_AA)

            # Blur the region with the squares 
            roi = frame[y:y+h, x:x+w]
            blurred_roi = cv.GaussianBlur(roi, (51, 51), 0)  # (51, 51) is kernel size
            frame[y:y+h, x:x+w] = blurred_roi


        cv.putText(frame, f"People in frame: {peopleInFrame}", (0, 50), font, 1, (255, 255, 255), 3, cv.LINE_AA)
        cv.imshow("Face Detection", frame)
        
        # Press 'q' to close the stream 
        if cv.waitKey(10) == ord("q"): 
            break
    
    # Close the window
    cap.release()

    # Deallocate memory used by the window 
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
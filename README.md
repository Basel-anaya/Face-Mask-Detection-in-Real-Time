# Face-Mask-Detection-in-Real-Time
This project implements real-time face mask detection using Haar Cascade classifiers. It aims to identify whether a person is wearing a face mask or not in a live video stream.
![image](https://github.com/Basel-anaya/Face-Mask-Detection-in-Real-Time/assets/81964452/aabcb7fb-4dd6-4f33-96b6-f8283355db08)


## Requirements
```
Python 3.x
OpenCV (cv2) library
Haar Cascade XML file for face detection
```

## Installation

1. Clone this repository to your local machine or download the project files.
2. Install the required dependencies using the following command:
```bash
pip install opencv-python
```

## Usage

1. Place the Haar Cascade XML file (`haarcascade_frontalface_default.xml`) in the project directory.
2. Open a terminal or command prompt and navigate to the project directory.
3. Run the following command to start the face mask detection:
```bash
python face_mask_detection.py
```
4. A new window will open displaying the live video stream from your default camera.
5. The program will detect faces in the video stream and label them as `"Mask"` or `"No Mask"` based on the presence of a face mask.

## Customization

- You can adjust the Haar Cascade detection parameters or use a different Haar Cascade XML file for face detection. Refer to the `OpenCV documentation` for more details on customizing the `Haar Cascade classifier`.
- If you have a trained model for face mask detection, you can integrate it into the project to improve the accuracy of mask detection. Modify the code accordingly to load and use your trained model.

## Contributing
Contributions are welcome! If you have any suggestions, bug reports, or enhancements, please open an issue or submit a pull request.

## License
This project is licensed under the [Apache 2.0 License](LICENSE).

## Acknowledgments
- The face detection functionality in this project is based on the Haar Cascade classifiers available in OpenCV.
- Thanks to the open-source community for providing valuable resources and tools.


## References
- [OpenCV documentation](https://docs.opencv.org/)
- [Haar Cascade classifiers](https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html)

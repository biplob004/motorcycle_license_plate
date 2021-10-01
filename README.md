# Motorcycle licence plate extraction when rider not wearing helmet 
This is project is extracting motorcycle license plates if the rider or back sitter is not wearing a helmet.

In this project i am using yolv5 algorithm to detect rider, licence plate and riders head. Then using resnet50 image classifier, i classify wheather the rider is wearing a helmet or not. If not wearing a helmet then, it will save his licence plate number in a image folder, and also will capture rider image in a separate folder.

<img
src="./bike_gif.gif"
/>



# Download Model and other necessary files from link below:
After downloading these files put them all in the main folder and now you can run the code.


yolov5 weight file for motorcycle rider, license plate and head detection: 

Small version:

https://drive.google.com/file/d/1LbX-YRBTgJZEIqQqmCpNq4CNQzoiA8p5/view?usp=sharing


Medium Version:

https://drive.google.com/file/d/16ZDjUcX7vXQYPJ557dooN0em4mJmbJt7/view?usp=sharing

Classification Model: (resnet50)

https://drive.google.com/file/d/1OuOpgq99E5VKPtwUuFZ0GGety3YWjIQn/view?usp=sharing







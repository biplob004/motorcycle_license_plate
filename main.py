
import time
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.general import non_max_suppression
from torchvision import models
import random
import math
from torchvision import transforms
from PIL import Image
import time
import os


# may need to modify these variables
yolov5_weight_file = 'rider_helmet_number_small.pt' # ... may need full path
conf_set = 0.35  # Confident score of detection
frame_size = (800, 480)# (1024, 576) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(yolov5_weight_file, map_location=device)
cudnn.benchmark = True 
names = model.module.names if hasattr(model, 'module') else model.names


def object_detection(frame):
	img = torch.from_numpy(frame)
	img = img.permute(2, 0, 1).float().to(device)
	img /= 255.0  
	if img.ndimension() == 3:
		img = img.unsqueeze(0)

	pred = model(img, augment=False)[0]
	pred = non_max_suppression(pred, conf_set, 0.30) # prediction, conf, iou

	detection_result = []
	for i, det in enumerate(pred):
		if len(det): 
			for d in det: # d = (x1, y1, x2, y2, conf, cls)
				x1 = int(d[0].item())
				y1 = int(d[1].item())
				x2 = int(d[2].item())
				y2 = int(d[3].item())
				conf = round(d[4].item(), 2)
				c = int(d[5].item())
				
				detected_name = names[c]
				detection_result.append([x1, y1, x2, y2, conf, c])
				
				frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 1) # box
				if c!=1: #head
					frame = cv2.putText(frame, f'{names[c]} {str(conf)}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

	return (frame, detection_result)



######################################################## Image classifier #############################

head_classification_threshold = 3.9
labels = ['helmet', 'no helmet']
model2 = torch.load('helment_no_helmet98.6.pth', map_location=device)  # ... may need full path
model2.eval()

transform = transforms.Compose([
			transforms.Resize(144),
			# transforms.CenterCrop(142),
			transforms.ToTensor(),
			transforms.Normalize([0.5], [0.5])
		  ]) 


def img_classify(frame):
	frame = transform(Image.fromarray(frame))
	frame = frame.unsqueeze(0)
	prediction = model2(frame)
	result_idx = torch.argmax(prediction).item()

	prediction_conf = sorted(prediction[0]) 
	cs = (prediction_conf[-1]-prediction_conf[-2]).item() # confident score
	print(cs) 
	# provide a threshold value of classification prediction as cs
	if cs > head_classification_threshold: #< --- Classification confident score. Need to adjust, this value
		return [True, cs] if result_idx == 0 else [False, cs]
	else:
		return [None, cs]

#########################################################
# Return::: Is this small box inside this big box?
def inside_box(big_box, small_box):
	x1 = small_box[0] - big_box[0]
	y1 = small_box[1] - big_box[1]
	x2 = big_box[2] - small_box[2]
	y2 = big_box[3] - small_box[3]
	return not bool(min([x1, y1, x2, y2, 0]))


######################## Main ####################################################################################

def main(source, show_img=False, save_img=False):
	cap = cv2.VideoCapture(source)

	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			# Some list of shape yolov5 except
			# 1824, 1376 
			# 1024, 576 # 4.0
			# 928, 544
			# 800, 480 # 
			# 416, 320
			# 384 288 
			# 320, 256
			# 256, 192
			frame = cv2.resize(frame, frame_size)  # resizing image
			orifinal_frame = frame.copy()
			frame, results = object_detection(frame) 

			rider_list = []
			head_list = []
			number_list = []

			for result in results:
				x1,y1,x2,y2,cnf, clas = result
				if clas == 0:
					rider_list.append(result)
				elif clas == 1:
					head_list.append(result)
				elif clas == 2:
					number_list.append(result)

			for rdr in rider_list:
				time_stamp = str(time.time())
				x1r, y1r, x2r, y2r, cnfr, clasr = rdr
				for hd in head_list:
					x1h, y1h, x2h, y2h, cnfh, clash = hd
					if inside_box([x1r,y1r,x2r,y2r], [x1h,y1h,x2h,y2h]): # if this head inside this rider bbox
						try:
							head_img = orifinal_frame[y1h:y2h, x1h:x2h]
							helmet_present = img_classify(head_img)
						except:
							helmet_present[0] = None

						if  helmet_present[0] == True: # if helmet present
							frame = cv2.rectangle(frame, (x1h, y1h), (x2h, y2h), (0,255,0), 1)
							frame = cv2.putText(frame, f'{round(helmet_present[1],1)}', (x1h, y1h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
						elif helmet_present[0] == None: # Poor prediction
							frame = cv2.rectangle(frame, (x1h, y1h), (x2h, y2h), (0, 255, 255), 1)
							frame = cv2.putText(frame, f'{round(helmet_present[1],1)}', (x1h, y1h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
						elif helmet_present[0] == False:
							frame = cv2.rectangle(frame, (x1h, y1h), (x2h, y2h), (0, 0, 255), 1)
							frame = cv2.putText(frame, f'{round(helmet_present[1],1)}', (x1h, y1h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
							try:
								cv2.imwrite(f'riders_pictures/{time_stamp}.jpg', frame[y1r:y2r, x1r:x2r])
							except:
								print('could not save rider')

							for num in number_list:
								x1_num, y1_num, x2_num, y2_num, conf_num, clas_num = num
								if inside_box([x1r,y1r,x2r,y2r], [x1_num, y1_num, x2_num, y2_num]):
									try:
										num_img = orifinal_frame[y1_num:y2_num, x1_num:x2_num]
										cv2.imwrite(f'number_plates/{time_stamp}_{conf_num}.jpg', num_img)
									except:
										print('could not save number plate')

			if save_img:
				cv2.imwrite('Entire frame.jpg', frame)
			if show_img:
				frame = cv2.resize(frame, (900, 450))  # resizing to fit on computer screen
				cv2.imshow('Entire frame.jpg', frame)

				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				
		else:
			break

	cap.release()
	cv2.destroyAllWindows()


# main('1.jpg',save_img=True) # from image
main('vv2.MOV', show_img=True) # from video
# main(0, show_img=True) # from webcam

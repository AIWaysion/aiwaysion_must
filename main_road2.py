import cv2
import numpy as np
from time import sleep
from requests_html import HTMLSession
import socket
# from udp import FrameSegment
import json
import time
import boto3
import os
from datetime import datetime
from tqdm import tqdm
import threading
import queue
import mysql.connector
import requests
import joblib

#TODO: change uuid
uuid = "UUID"
#TODO: set image freq (in seconds)
image_freq = 60

# In[1]: data transfer

s3 = boto3.resource('s3',
		aws_access_key_id='AKIAWK6XCFPJZGD3Q7W6',
         	aws_secret_access_key='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')

# if use udp
#s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#fs = FrameSegment(s, port=5555, addr="54.219.161.172")

# In[2]: var.

class VideoInputThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):

        global restart
        global getVideo
        global trigger

        while True:
            #cap = cv2.VideoCapture('rtsp://wowza01.bellevuewa.gov:1935/live/CCTV063.stream')
            #cap = cv2.VideoCapture('rtsp://10.147.18.39:8554/B3stream')
            cap = cv2.VideoCapture('rtsp://192.168.1.18:554/1/h264major/user=admin&password=admin&channel=1&stream=0.sdp?')
            #cap = cv2.VideoCapture('videos/out.avi')
            ret, inputframe = cap.read()
            if ret:
                break
        
        while True:
            if restart == True:
                break
            ret, inputframe = cap.read()
            trigger = False
            if trigger:
                videoInputQueue.append(inputframe)
                if len(videoInputQueue) > 1:
                    videoInputQueue.pop(0)
                time.sleep(0.04)
            else:
                videoInputQueue.append(inputframe)
                if len(videoInputQueue) > 1:
                    videoInputQueue.pop(0)
                time.sleep(0.05)

videoInputQueue = []
restart = False

bgsubtract = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=15)
time_send_image = time.time()
input_buffer = queue.Queue()


# In[3]: def func.
#count_path = './runs/track/exp/counts.txt'
def box_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy


def weatherinfo(query):
    session = HTMLSession()
    url = f'https://www.google.com/search?q={query}+weather+now'
    response = session.get(url, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36'})
    temp = response.html.find('span#wob_ttm', first=True).text
    h = response.html.find('span#wob_hm', first=True).text
    h = h[:2]
    print(temp, h)
    return temp, h

# IF Response 429 (in case server has received too many requests)

def get_weather(city, country_code, api_key):
    # API endpoint and parameters
    url = 'https://api.openweathermap.org/data/2.5/weather'
    params = {'q': f'{city},{country_code}', 'appid': api_key, 'units': 'imperial'}

    # Send request to API and parse JSON response
    response = requests.get(url, params=params)
    data = response.json()

    # Extract temperature and humidity from response
    if 'main' in data:
        temp_f = data['main']['temp']
        temp_c = (temp_f - 32) * 5/9
        humidity = data['main']['humidity']
        print(f"Temperature in {city}: {temp_c:.1f}°C")
        print(f"Humidity in {city}: {humidity}%")
        return temp_c, humidity
    else:
        print("Error: Could not extract weather data from API response")

# send image
def send_image():
    global time_send_image
    while True:
        frame = input_buffer.get()
        current_hour =  datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        current_date =  datetime.now().strftime("%Y_%m_%d")
        #file_name = 'G:\\pysource-vehcount\\source code\\image_cache\\' + current_hour + '.jpg'
        #file_path = 'G:\\pysource-vehcount\\source code\\image_cache'
        file_name = '/home/luyang/program/aiwaysion_must/image_cache/' + current_hour + '.jpg'
        file_path = '/home/luyang/program/aiwaysion_must/image_cache/'
        if time.time() - time_send_image > image_freq:

            for f in os.listdir(file_path):
                os.remove(os.path.join(file_path, f))
            
            cv2.imwrite(file_name, frame)
            
            for f in tqdm(os.listdir(file_path)):
                #s3.meta.client.upload_file(file_path + '\\' + f, 'mustdevices', f'yakama/{current_date}/'+ f)
                s3.meta.client.upload_file(file_path + '/' + f, 'mustdevices', f'images/{uuid}/{current_date}/'+ f)
                toImg(uuid, f'/images/{uuid}/{current_date}/'+ f)
        
            time_send_image = time.time()



# API 
def toData(device_id, hum, date, temp, rdCon, speed, downstream, upstream):
    url = "https://api.staging.aiwaysion.com/v1/remote/device/data"
    data = {"device_id": device_id,  "hum": hum, "date": date, "temp": temp, "rdCon": rdCon, "speed": speed, "downstream": downstream, "upstream": upstream}
    myResponse = requests.post(url , data=data)
    if(myResponse.ok):
        print('Success to data')

def toImg(device_id, path_to_image):
    url = "https://api.staging.aiwaysion.com/v1/remote/device/images3"
    data = {"device_id": device_id,  "image": path_to_image} 
    myResponse = requests.post(url , data=data)
    if(myResponse.ok):
        print('Success to img')


# In[4]: bgsubtract para.

time_send_mess = time.time()

width_min = 120 #Minimum rectangular width
height_min = 120 #Minimum rectangular height
width_max = 1000 #Maximum rectangular width
height_max = 1000 #Maximum rectangular height

offset = 20 #The allowable error between pixels
line_pos_x = 500 #x position of the counting line

delay = 20 #FPS on video

detect = []
counts = 0
RF = joblib.load('/home/luyang/program/aiwaysion_must/RF_model_02092023.sav')

# In[5]: temp file dir.

#path ='G:\\pysource-vehcount\\source code'
path ='/home/luyang/program/aiwaysion_must/'
if not os.path.exists(os.path.join(path, 'image_cache')):
    os.mkdir(os.path.join(path, 'image_cache'))


# In[7]: thread
thread1 = threading.Thread(target=send_image)
thread1.start()

# In[8]: main loop
if __name__ == '__main__':
    videoInputThread = VideoInputThread()
    videoInputThread.start()
    while True:
        try:
            frame = videoInputQueue[0]

            
            # feed frame to thread
            buffer_frame = cv2.resize(frame, (640,480))
            input_buffer.put(buffer_frame)


            # start bgsubtract count method
            temp = float(1/delay)
            sleep(temp)
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            b,g,r = cv2.split(frame)
            dark_channel = cv2.min(cv2.min(r,g), b)
            m_g = np.median(gray)
            v_g = np.var(gray)
            m_d = np.median(dark_channel)
            v_d = np.var(dark_channel)
            blur = cv2.GaussianBlur(gray,(3,3),5)
            img_sub = bgsubtract.apply(blur)
            dilation = cv2.dilate(img_sub,np.ones((5,5)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            morphological_gradient = cv2.morphologyEx (dilation, cv2. MORPH_CLOSE , kernel)
            morphological_gradient = cv2.morphologyEx (morphological_gradient, cv2. MORPH_CLOSE , kernel)
            contour,h=cv2.findContours(morphological_gradient,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            # change line position here
            cv2.line(frame, (line_pos_x, 50), (line_pos_x, 650), (255,127,0), 3) 
            
            for(i,c) in enumerate(contour):
                (x,y,w,h) = cv2.boundingRect(c)
                valid_contour = (width_min <= w <= width_max) and (height_min<= h <= height_max)
                if not valid_contour:
                    continue

                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)        
                center = box_center(x, y, w, h)
                detect.append(center)
                cv2.circle(frame, center, 4, (0, 0,255), -1)

                for (x,y) in detect:
                    if x<(line_pos_x+offset) and x>(line_pos_x-offset):
                        counts+=1
                        cv2.line(frame, (line_pos_x, 50), (line_pos_x, 650), (0,127,255), 3)    
                        detect.remove((x,y))
                        print("car is detected : "+str(counts))
        except:
                #print('error')
                #time.sleep(5)
                continue

        # 15min Send to EC2 
        if time.time() - time_send_mess > 120:

            
            speed = 45.0 
            s1 = np.random.normal(speed,5, 5) 
            s2 = np.random.normal(speed-10, 5, 5) 
            s33 = np.random.normal(speed+10, 5, 5) 
            speed_queue = np.concatenate((s1, s2, s33)) 
            speed = str(np.around(np.mean(speed_queue), decimals =1)) 
            # time_send_speed = time.time() 
            # print("Speed is:", speed) 
        
            #temp_hum = weatherinfo('Yakima')
            temp_hum = get_weather('Bellevue', 'US', 'a0f784a6ad9973efd444a2615c6f7ad3')
            temperature = int(temp_hum[0])
            humidity = float(temp_hum[1])
            #udpstr = str(counts*4) + ',' + str(temp_hum[0]) + ',' + str(temp_hum[1])
            #print(udpstr)
            
            '''
            with open(count_path, 'r') as f:
                lines = f.readlines()
                last_line = lines[-1].split(',')
                up_count, down_count = last_line[0].split()[-1], last_line[1].split()[-1]
            '''
            # API/udp send str
            device_id = uuid 
            now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            y_pre = RF.predict([[temperature, humidity, m_g, v_g, m_d, v_d]])
            toData(device_id, temp_hum[1], now, int(temp_hum[0]), str(y_pre[0]), speed, counts*3, str(0))
            #fs.udp_event(bytes(udpstr, 'utf-8'))
            #s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            #s.sendto(udpstr.encode(), ("54.219.161.172", 5555))
            
            counts = 0

            time_send_mess = time.time()
                
        #cv2.putText(frame, "VEHICLE COUNT : "+str(counts), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
        #cv2.imshow("Video Original" , frame)
        #cv2.imshow("Detector",morphological_gradient)

        if cv2.waitKey(1) == ord('q'):
                    break

    
cv2.destroyAllWindows()
#cap.release()

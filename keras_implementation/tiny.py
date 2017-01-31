import numpy as np
from keras.models import Sequential, load_model, Model
from keras.layers import Convolution2D, Dense, Flatten, merge, MaxPooling2D, Input, ZeroPadding2D
from keras.optimizers import adam, rmsprop
from keras.initializations import uniform
from keras.layers.advanced_activations import LeakyReLU, ELU
import keras.backend.tensorflow_backend as K
import tensorflow as tf

# from utils import load_model_from_ckpt
from ipdb import set_trace

import cv2
import time
import sys

class YOLO_TF:
    fromfile = None
    # tofile_img = 'test/output.jpg'
    # tofile_txt = 'test/output.txt'
    imshow = True
    filewrite_img = False
    filewrite_txt = False
    disp_console = False
    # weights_file = 'weights/YOLO_tiny.ckpt'
    alpha = 0.1
    threshold = 0.1
    iou_threshold = 0.5
    num_class = 20
    num_box = 2
    grid_size = 7
    classes =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

    w_img = 640
    h_img = 480

    def __init__(self,argvs = []):
        # reducing tf GPU usage
        K.set_session(tf.Session(config=tf.ConfigProto(allow_soft_placement=True)))

        self.argv_parser(argvs)
        self.model = self.build_networks()
        set_trace()

    def argv_parser(self,argvs):
        for i in range(1,len(argvs),2):
            if argvs[i] == '-fromfile' : self.fromfile = argvs[i+1]
            if argvs[i] == '-tofile_img' : self.tofile_img = argvs[i+1] ; self.filewrite_img = True
            if argvs[i] == '-tofile_txt' : self.tofile_txt = argvs[i+1] ; self.filewrite_txt = True
            if argvs[i] == '-imshow' :
                if argvs[i+1] == '1' :self.imshow = True
                else : self.imshow = False
            if argvs[i] == '-disp_console' :
                if argvs[i+1] == '1' :self.disp_console = True
                else : self.disp_console = False
                

    def build_networks(self):
        model = Sequential()
        
        y = (Input(shape=(448,448,3)))
        # first layer
        x = (ZeroPadding2D(padding=(1,1)))(y)
        x = (Convolution2D(16, 3, 3, subsample=(1,1), activation='linear', name='conv1', border_mode='valid' ,init='uniform', trainable=True))(x)
        x = (LeakyReLU(alpha=0.1))(x)
        x = (MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool1', trainable=True, border_mode='same'))(x)

        for i in range(2,7):
            x = (ZeroPadding2D(padding=(1,1)))(x)
            x = (Convolution2D(16*(2**(i-1)), 3, 3, subsample=(1,1), activation='linear', border_mode='valid', name='conv' + str(i), init='uniform', trainable=True))(x)
            x = (LeakyReLU(alpha=0.1))(x)
            x = (MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool' + str(i), trainable=True, border_mode='same'))(x)

        # set_trace()
        for i in range(3):
            x = (ZeroPadding2D(padding=(1,1)))(x)
            x = (Convolution2D(1024, 3, 3, subsample=(1,1), activation='linear', name='conv' + str(i+7), border_mode='valid', init='uniform', trainable=True))(x)

        x = (Flatten())(x)
        x = (Dense(256, activation='linear', name='FC1', init='uniform'))(x)
        x = (LeakyReLU(alpha=0.1))(x)
        x = (Dense(4096, activation='linear', name='FC2', init='uniform'))(x)
        x = (LeakyReLU(alpha=0.1))(x)
        x = (Dense(7*7*self.num_class + 7*7*2 + 7*7*2*4, activation='linear', name='FC3', init='uniform'))(x)

        model = Model(input = y, output=x)
        model.compile(optimizer=adam(lr=10**-3), loss="mse")

        return model

    def detect_from_cvmat(self,img):
        s = time.time()
        self.h_img,self.w_img,_ = img.shape
        img_resized = cv2.resize(img, (448, 448))
        img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
        img_resized_np = np.asarray( img_RGB )
        inputs = np.zeros((1,448,448,3),dtype='float32')
        inputs[0] = (img_resized_np/255.0)*2.0-1.0
        # in_dict = {self.x: inputs}
        # net_output = self.sess.run(self.fc_19,feed_dict=in_dict)
        net_output = self.model.predict_on_batch(inputs)    
        # set_trace()
        self.result = self.interpret_output(net_output[0])
        self.show_results(img,self.result)
        strtime = str(time.time()-s)
        if self.disp_console : print 'Elapsed time : ' + strtime + ' secs' + '\n'

    def detect_from_file(self,filename):
        if self.disp_console : print 'Detect from ' + filename
        img = cv2.imread(filename)
        #img = misc.imread(filename)
        self.detect_from_cvmat(img)

    def interpret_output(self,output):
        probs = np.zeros((7,7,2,self.num_class))
        class_probs = np.reshape(output[0:980],(7,7,self.num_class))
        scales = np.reshape(output[980:1078],(7,7,2))
        boxes = np.reshape(output[1078:],(7,7,2,4))
        offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))

        boxes[:,:,:,0] += offset
        boxes[:,:,:,1] += np.transpose(offset,(1,0,2))
        boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
        boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
        boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])
        
        boxes[:,:,:,0] *= self.w_img
        boxes[:,:,:,1] *= self.h_img
        boxes[:,:,:,2] *= self.w_img
        boxes[:,:,:,3] *= self.h_img

        for i in range(2):
            for j in range(self.num_class):
                probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])

        set_trace()

        filter_mat_probs = np.array(probs>=self.threshold,dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]] 

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]
        
        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0 : continue
            for j in range(i+1,len(boxes_filtered)):
                if self.iou(boxes_filtered[i],boxes_filtered[j]) > self.iou_threshold : 
                    probs_filtered[j] = 0.0
        
        filter_iou = np.array(probs_filtered>0.0,dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

        return result

    def show_results(self,img,results):
        img_cp = img.copy()
        if self.filewrite_txt :
            ftxt = open(self.tofile_txt,'w')
        for i in range(len(results)):
            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3])//2
            h = int(results[i][4])//2
            if self.disp_console : print '    class : ' + results[i][0] + ' , [x,y,w,h]=[' + str(x) + ',' + str(y) + ',' + str(int(results[i][3])) + ',' + str(int(results[i][4]))+'], Confidence = ' + str(results[i][5])
            if self.filewrite_img or self.imshow:
                cv2.rectangle(img_cp,(x-w,y-h),(x+w,y+h),(0,255,0),2)
                cv2.rectangle(img_cp,(x-w,y-h-20),(x+w,y-h),(125,125,125),-1)
                cv2.putText(img_cp,results[i][0] + ' : %.2f' % results[i][5],(x-w+5,y-h-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
            if self.filewrite_txt :             
                ftxt.write(results[i][0] + ',' + str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h)+',' + str(results[i][5]) + '\n')
        if self.filewrite_img : 
            if self.disp_console : print '    image file writed : ' + self.tofile_img
            cv2.imwrite(self.tofile_img,img_cp)         
        if self.imshow :
            cv2.imshow('YOLO_tiny detection',img_cp)
            cv2.waitKey(1)
        if self.filewrite_txt : 
            if self.disp_console : print '    txt file writed : ' + self.tofile_txt
            ftxt.close()

    def iou(self,box1,box2):
        tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
        lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
        if tb < 0 or lr < 0 : intersection = 0
        else : intersection =  tb*lr
        return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)



if __name__ == "__main__":
    yolo = YOLO_TF()
    filename = "../weights/YOLO_tiny.ckpt"
    # yolo.model.load_weights("my_yolo_tiny.h5")
    yolo.model.load_weights("tiny_weights.h5")
    set_trace()
    img_file = "../test/person.jpg"
    yolo.detect_from_file(img_file)
    cv2.waitKey(0)
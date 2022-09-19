from pynq import Overlay, DefaultIP
import pynq
import numpy as np
from time import time

def Model(path):
    return Overlay(path).cnn_action_detection_0

class CNNDriver(DefaultIP):
    def __init__(self, description):
        super().__init__(description=description)
        
        self.data_in = pynq.allocate(shape=(90,), dtype=np.float32)
        self.register_map.data_in=self.data_in.device_address
        
        self.raw_outputs = pynq.allocate(shape=(3,), dtype=np.float32)
        self.register_map.raw_output=self.raw_outputs.device_address
        
        self.raw_cnn_outputs = pynq.allocate(shape=(130,), dtype=np.float32)
        self.register_map.cnn_output=self.raw_cnn_outputs.device_address
        
        self.debug = False
        
    bindto = ["xilinx.com:hls:cnn_action_detection:1.0"] 
    
    def inference(self, window: np.int16):
        """
        Feed a window of data into the model and get the current prediction
        and confidence by the model. The sliding window must have window size
        of 15 and window stride of 5.
        
        A window consists of 15 data, each 6 channels.
        The window format should be:
        d1c1, d1c2, d1c3, ..., d1c6, d2c1, d2c2,... ,d15c5, d15c6
        whereas d1c6 refers to the value of the 6th channel of the 1st data
        All data must be in np.int16 format.
        
        The function saves the raw(before softmax) outputs to self.raw_outputs
        
        Returns the index of the predicted type of action.
        The index and action types are as follows:
        +---+------------+
        | 0 | Shield     |
        +---+------------+
        | 1 | Reload     |
        +---+------------+
        | 2 | Grenade    |
        +---+------------+
        | 3 | Final move |
        +---+------------+
        | 4 | Idle       |
        +---+------------+
        
        Usually the inference can only give high confidence(e.g. >80%) after receiving 
        10 windows of data. 
        
        Sample usage:
        data=[d1, d2, ..., d150]
        predicted_class, confidence = IP.inference(data[0:15])
        predicted_class, confidence = IP.inference(data[5:20])
        predicted_class, confidence = IP.inference(data[10:25])
        ............
        predicted_class, confidence = IP.inference(data[135:150])
        
        Parameters:
        window: flattened input of 15 data, each 6 channels, in np.int16 format 
        
        Returns:
        predicted_class(int) : Index of the predicted type of action
        confidence(float     : Confidence that the action belong to the predicted class
        
        """
        
        # preparing input for the IP
        self.data_in[:] = np.float32(window[:]/4096.0)
        self.register_map.function_select=0
        
        # start inferencing
        if self.debug:
            start_time = time()
        self.register_map.function_select=0
        self.register_map.CTRL.AP_START=1
        while(self.register_map.CTRL.AP_DONE == 0):pass
        self.register_map.function_select=1
        self.register_map.CTRL.AP_START=1
        while(self.register_map.CTRL.AP_DONE == 0):pass
        if self.debug:
            print(f"time took for inference = {time() - start_time}")
        
        predicted_class = int(self.register_map.result_out)
        confidence = max(softmax(self.raw_outputs))
        return predicted_class, confidence
    
    def getResult(self):
        """
        (debug) Get raw CNN layer outputs and save to self.raw_cnn_outputs
        
        """
        self.register_map.function_select=1
        if self.debug:
            start_time = time()
        self.register_map.CTRL.AP_START=1
        while(self.register_map.CTRL.AP_DONE == 0):pass
        if self.debug:
            print(f"time took for reading raw result = {time() - start_time}")
    
    def resetBuffer(self):
        '''
        To reset the buffer in the network and get ready for new inference.
        '''
        self.register_map.function_select=2
        if self.debug:
            start_time = time()
        self.register_map.CTRL.AP_START=1
        while(self.register_map.CTRL.AP_DONE == 0):pass
        if self.debug:
            print(f"time took for resetting buffer = {time() - start_time}")
            
    def setCNNWeights(self, new_weights):
        setWeightsOrBias(new_weights, 3)
    
    def setCNNBias(self, new_weights):
        setWeightsOrBias(new_weights, 4)
        
    def setDenseWeights(self, new_weights):
        setWeightsOrBias(new_weights, 5)
        
    def setDenseBias(self, new_weights):
        setWeightsOrBias(new_weights, 6)
            
    def setWeightsOrBias(self, new_weights, weights_or_bias):
        buffer = pynq.allocate(shape=data.shape, dtype=data.dtype)
        buffer[:] = new_weights[:]
        self.register_map.weights_and_bias = buffer.device_address
        self.register_map.function_select=3
        self.register_map.CTRL.AP_START=1
        while(self.register_map.CTRL.AP_DONE == 0):pass
        buffer.freebuffer()
        
def writeToDDR(data):
    buffer = pynq.allocate(shape=data.shape, dtype=data.dtype)
    buffer[:] = data[:]
    return buffer

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
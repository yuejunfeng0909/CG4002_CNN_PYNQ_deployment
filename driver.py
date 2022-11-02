from pynq import Overlay, DefaultIP
import pynq
import numpy as np
from time import time
from typing import List

PRESET_THRESH = 0.95

def Model(path):
    return Overlay(path).cnn_action_detection_0

class CNNDriver(DefaultIP):
    def __init__(self, description):
        super().__init__(description=description)
        
        self.input = pynq.allocate(shape=(6,), dtype=np.float32)
        self.register_map.data = self.input.device_address
        
        self.raw_outputs = pynq.allocate(shape=(5,), dtype=np.float32)
        self.register_map.raw_output=self.raw_outputs.device_address
        self.threshold = PRESET_THRESH
        
        self.debug = True
        
    bindto = ["xilinx.com:hls:cnn_action_detection:1.0"] 
    
    def set_threshold(self, new_threshold):
        self.threshold = new_threshold
    
    def inference(self, data: List[int], user_number=0):
        """
        Feed data into the model and get the current prediction
        and confidence by the model.
        
        A data consists of 6 integer values.
        All data must be in np.int16 format.
        
        The function saves the raw(before softmax) outputs to self.raw_outputs
        
        Returns the index of the predicted type of action.
        The index and action types are as follows:
        +---+------------+
        |-1 | None       |
        +---+------------+
        | 0 | Shield     |
        +---+------------+
        | 1 | Reload     |
        +---+------------+
        | 2 | Grenade    |
        +---+------------+
        | 3 | Final move |
        +---+------------+
        
        Sample usage:
        d1 = [0, 0, 0, 0, 0, 0]
        data=[d1, d2, ..., d75]
        predicted_class, confidence = IP.inference(data[0])
        predicted_class, confidence = IP.inference(data[1])
        predicted_class, confidence = IP.inference(data[2])
        ............(until high confidence is reached)
        
        Parameters:
        window: np.array of 6 values, in np.int32 format. 
        
        Returns:
        predicted_class(int) : Index of the predicted type of action
        confidence(float     : Confidence that the action belong to the predicted class
        
        """
        if self.debug:
            start_time = time()
        
        # preparing input for the IP
        self.input[:] = np.float32([i/4096.0 for i in data])
        
        # start inferencing
        self.register_map.function_select=0
        self.register_map.user_number = user_number
        self.register_map.CTRL.AP_START=1
        while(self.register_map.CTRL.AP_DONE == 0):pass # mostly immediate

        # Confidence
        predicted_class = np.argmax(self.raw_outputs)
        confidence = max(softmax(self.raw_outputs))
        
        if self.debug:
            print(f"player {user_number}, predicted={["Shield", "Reload", "Grenade", "Logout"][predicted_class]}, confidence={confidence*100:.3f}%, time took for inference={(time() - start_time)*1000:.3f}ms")
            
        if predicted_class == 4 or confidence < self.threshold:
            return -1
        return predicted_class
    
    def resetBuffer(self, user_number=0):
        '''
        To reset the buffer in the network and get ready for new inference.
        '''
        if self.debug:
            start_time = time()
           
        self.register_map.function_select=1
        self.register_map.user_number = user_number
        self.register_map.CTRL.AP_START=1
        while(self.register_map.CTRL.AP_DONE == 0):pass
        
        if self.debug:
            print(f"time took for resetting buffer = {time() - start_time}")

    def setCNNWeights(self, new_weights):
        self.setWeightsOrBias(new_weights, 2)
    
    def setCNNBias(self, new_weights):
        self.setWeightsOrBias(new_weights, 3)
        
    def setDenseWeights(self, new_weights):
        self.setWeightsOrBias(new_weights, 4)
        
    def setDenseBias(self, new_weights):
        self.setWeightsOrBias(new_weights, 5)
            
    def setWeightsOrBias(self, new_weights, weights_or_bias):
        buffer = pynq.allocate(shape=new_weights.shape, dtype=np.float32)
        buffer[:] = new_weights[:]
        self.register_map.weights_and_bias = buffer.device_address
        self.register_map.function_select=weights_or_bias
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

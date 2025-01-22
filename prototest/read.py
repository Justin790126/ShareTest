#!/bin/python3.7
import tensorflow as tf
from tensorflow.core.util import event_pb2
path="/home/justin126/workspace/ShareTest/prototest/logs/train/events.out.tfevents.1737293120.justin126-VirtualBox.24119.0.v2"
serialized_examples = tf.data.TFRecordDataset(path)
for serialized_example in serialized_examples:
    event = event_pb2.Event.FromString(serialized_example.numpy())
    for value in event.summary.value:
        t = tf.make_ndarray(value.tensor)
        print(value.tag, event.step, t, type(t))
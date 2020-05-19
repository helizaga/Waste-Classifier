import tensorflow.compat.v1 as tf
import sys
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
Tk().withdraw()

image_path = askopenfilename(title="Choose a file",
    filetypes=[('image files', '.png'),
               ('image files', '.jpg'),
           ])

if image_path:
    
    image_data = tf.gfile.GFile(image_path, 'rb').read()

    label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile("tf_files/retrained_labels.txt")]

    with tf.gfile.GFile("tf_files/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (Prob: %.2f)' % (human_string, score))

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from flask import Flask, escape, request, jsonify

import argparse
import sys
import time
import pybase64

import numpy as np
import tensorflow as tf

app = Flask(__name__)

@app.route('/')
def mainServer():
    file_name = "tf_files/flower_photos/daisy/3475870145_685a19116d.jpg"
    model_file = "../tf_files/retrained_graph.pb"
    label_file = "../tf_files/retrained_labels.txt"
    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed")
    # parser.add_argument(request.args.get('image'))
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    # testString = request.args.get('str')
    # graph = "../tf_files/retrained_graph.pb"

    args = parser.parse_args()

    if args.graph:
      # model_file = args.graph
      "../tf_files/retrained_graph.pb"
    if request.args.get('img'):
      # file_name = args.image
      file_name = request.args.get('img')
    if args.labels:
      label_file = args.labels
    if args.input_height:
      input_height = args.input_height
    if args.input_width:
      input_width = args.input_width
    if args.input_mean:
      input_mean = args.input_mean
    if args.input_std:
      input_std = args.input_std
    if args.input_layer:
      input_layer = args.input_layer
    if args.output_layer:
      output_layer = args.output_layer

    graph = load_graph(model_file)
    # t = read_tensor_from_image_file(file_name,
    #                                 input_height=input_height,
    #                                 input_width=input_width,
    #                                 input_mean=input_mean,
    #                                 input_std=input_std)

    t = read_tensor_from_image_b64(file_name,
                                    input_height=input_height,
                                    input_width=input_width,
                                    input_mean=input_mean,
                                    input_std=input_std)

    print("SHAPE")
    print(t.shape)
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    with tf.Session(graph=graph) as sess:
      start = time.time()
      results = sess.run(output_operation.outputs[0],
                        {input_operation.outputs[0]: t})
      end=time.time()
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)

    print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
    template = '"name":"{}", "score":"{:0.5f}"'
    stringToReturn = '{"possible_pokemon": ['
    listOfPossiblePokes = []
    for i in top_k:
        listOfPossiblePokes.append({'name':labels[i],'score':str(results[i])})
      # print(template.format(labels[i], results[i]))
        stringToReturn += "{" + template.format(labels[i], results[i])+"},"
    stringToReturn = stringToReturn[:-1]#THIS REMOVES THE LAST COMMA
    stringToReturn += ']}'
    # print(jsonify(listOfPossiblePokes))
    return jsonify({'possible_pokemon':listOfPossiblePokes})
    # return (stringToReturn)
    # return "<h1>Label Image Server!</h1>" + "\n<h2>enter</h2>"


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def read_tensor_from_image_b64(b64String, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  # input_name = "file_reader"
  # output_name = "normalized"
  # file_reader = tf.read_file(file_name, input_name)
  # image_reader = tf.io.decode_base64(b64String, name='base64_reader')
  # print(image_reader)
  # image_reader = tf.image.decode_image(b64String, name='base64_reader')
  # image_reader = tf.image.convert_image_dtype(image_reader,dtype = tf.float32)
  # image_reader = tf.image.decode_png(b64, name = 'png_reader')

  # float_caster = tf.cast(image_reader, tf.float32)
  # dims_expander = tf.expand_dims(float_caster, 0);
  # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Dims expander is !!!!!!!!!!!!!!!!!!!!!!!!!!1"+str(dims_expander))
  # # resized = tf.image.resize(dims_expander, [input_height, input_width])
  # resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  # normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  # sess = tf.Session()
  # result = sess.run(normalized)
  #
  # return result


  buf = str(pybase64.b64decode(b64String))
  result = np.fromstring(buf, dtype=np.float32)
  result = result.reshape((1, -1, -1, 3))
  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__ == "__main__":
  app.run(debug=True)
  # file_name = "tf_files/flower_photos/daisy/3475870145_685a19116d.jpg"
  # model_file = "tf_files/retrained_graph.pb"
  # label_file = "tf_files/retrained_labels.txt"
  # input_height = 224
  # input_width = 224
  # input_mean = 128
  # input_std = 128
  # input_layer = "input"
  # output_layer = "final_result"
  #
  # parser = argparse.ArgumentParser()
  # parser.add_argument("--image", help="image to be processed")
  # parser.add_argument("--graph", help="graph/model to be executed")
  # parser.add_argument("--labels", help="name of file containing labels")
  # parser.add_argument("--input_height", type=int, help="input height")
  # parser.add_argument("--input_width", type=int, help="input width")
  # parser.add_argument("--input_mean", type=int, help="input mean")
  # parser.add_argument("--input_std", type=int, help="input std")
  # parser.add_argument("--input_layer", help="name of input layer")
  # parser.add_argument("--output_layer", help="name of output layer")
  # mainServer(parser)

def decode(base64_string):
    if isinstance(base64_string, bytes):
        base64_string = base64_string.decode("utf-8")

    imgdata = base64.b64decode(base64_string)
    img = skimage.io.imread(imgdata, plugin='imageio')
    return img

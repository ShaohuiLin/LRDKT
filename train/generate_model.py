#coding: utf-8

import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
from google.protobuf import text_format
import argparse

def convert_type(type):
    if type == 4:
        return "convolution"
    elif type == 14:
        return "innerproduct"
    else:
        return type


def read_caffemodel(PATH):
    net_data = caffe_pb2.NetParameter()
    caffe_data = open(PATH)
    net_data.MergeFromString(caffe_data.read())
    net_dict = {x.name:x for x in net_data.layer}
    return net_data,net_dict


def setLayerData(layers,name, new_W, new_b):
    del layers[name].blobs[0].data[:]
    del layers[name].blobs[1].data[:]
    layers[name].blobs[0].data.extend(new_W)
    layers[name].blobs[1].data.extend(new_b)


def saveModel(PATH,net_data):
    with open(PATH, 'w') as f:
        f.write(net_data.SerializeToString())


def read_prototxt(PATH):
    net_config = caffe.proto.caffe_pb2.NetParameter()
    file = open(PATH, "r")
    text_format.Merge(str(file.read()), net_config)
    file.close()

    net_msg = dict()
    if len(net_config.layer) == 0:
        layers = net_config.layers
    else:
        layers = net_config.layer
    for layer in layers:
        net_msg[layer.name] = dict()
        net_msg[layer.name]["type"] = convert_type(layer.type) if isinstance(layer.type, int) else layer.type.lower()
        net_msg[layer.name]["bottom"] = layer.bottom
        net_msg[layer.name]["top"] = layer.top

    return net_msg


def genLayer(name, type, param):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = type

    blobs = []

    for p in param:
        blob = caffe_pb2.BlobProto()
        blob.data.extend(p.data.flatten())
        blob.shape.dim.extend(p.data.shape)
        blobs.append(blob)
    
    layer.blobs.extend(blobs)

    return layer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='decom layer')
    parser.add_argument('--model', type=str, required=True, help='The original network model.')
    parser.add_argument('--weights', type=str, required=True, help='The original network weight.')
    parser.add_argument('--input', type=str, required=True, help='The decomposed network model.')
    parser.add_argument('--output', type=str, required=True, help='the directory of the output model.')
    args = parser.parse_args()

    net = caffe.Net(args.model, args.weights, caffe.TEST)
    lrNet, lrNet_dict = read_caffemodel(args.input)
    net_msg = read_prototxt(args.model)
    savePath = args.output

    for name in net.params.keys():
        layer = genLayer(name + "_temp", net_msg[name]["type"], net.params[name])
        print(name + "_temp")
        lrNet.layer.extend([layer])

    saveModel(savePath, lrNet)

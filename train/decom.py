#coding: utf-8

import numpy as np
from scipy import linalg as LA
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
from google.protobuf import text_format
import argparse

PCA = 0.9


def svd_decom(W, b, K, type):
    if type.lower() == "convolution":
        N, C, y, x = W.shape
        W = W.transpose((1, 2, 0, 3)).reshape((C*y, -1))
        U, D, Q = np.linalg.svd(W, full_matrices=False)
        sqrt_D = LA.sqrtm(np.diag(D))
        V = U[:, :K].dot(sqrt_D[:K, :K])
        H = Q.T[:, :K].dot(sqrt_D[:K, :K])
        V = V.T.reshape(K, C, y, 1)
        H = H.reshape(N, x, 1, K).transpose((0, 3, 2, 1))
        if not b is None:
            b_1 = np.zeros((K,))
            b_2 = b
        else:
            b_1 = None
            b_2 = None
        W1, b1, W2, b2 = V, b_1, H, b_2
    elif type.lower() == "innerproduct":
        U, D, Q = np.linalg.svd(W, full_matrices=False)
        sqrt_D = LA.sqrtm(np.diag(D))
        V = U[:, :K].dot(sqrt_D[:K, :K])
        H = Q.T[:, :K].dot(sqrt_D[:K, :K])
        H = H.T
        if not b is None:
            b_1 = np.zeros((K,))
            b_2 = b
        else:
            b_1 = None
            b_2 = None
        W1, b1, W2, b2 = H, b_1, V, b_2
    else:
        print("not support layer type")
        return

    return W1, b1, W2, b2


def read_caffemodel(PATH):
    net_data = caffe_pb2.NetParameter()
    caffe_data = open(PATH)
    net_data.MergeFromString(caffe_data.read())
    return net_data


def delLayer(oriNet, layerName):
    for i in range(len(oriNet.layers)):
        if oriNet.layers[i].name == layerName:
            del oriNet.layers[i]
            break

    return oriNet


def saveModel(PATH, net_data):
    with open(PATH, 'w') as f:
        f.write(net_data.SerializeToString())


def genLayer(name, type, weight, bias):
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = type

    if type.lower() == "convolution":
        blob1 = caffe_pb2.BlobProto()
        blob1.data.extend(weight.flatten())
        blob1.num = weight.shape[0]
        blob1.channels = weight.shape[1]
        blob1.height = weight.shape[2]
        blob1.width = weight.shape[3]
        if not bias is None:
            blob2 = caffe_pb2.BlobProto()
            blob2.data.extend(bias)
            blob2.shape.dim.extend(bias.shape)
    elif type.lower() == "innerproduct":
        blob1 = caffe_pb2.BlobProto()
        blob1.data.extend(weight.flatten())
        blob1.shape.dim.extend(weight.shape)
        if not bias is None:
            blob2 = caffe_pb2.BlobProto()
            blob2.data.extend(bias.flatten())
            blob2.shape.dim.extend(bias.shape)
    else:
        print(name, type)
        print("not support layer type")
        return

    if bias is None:
        layer.blobs.extend([blob1])
    else:
        layer.blobs.extend([blob1, blob2])

    return layer


def readProtoTxtFile(filepath):
    file = open(filepath, "r")
    net_config = caffe.proto.caffe_pb2.NetParameter()
    text_format.Merge(str(file.read()), net_config)
    file.close()
    return net_config


def writeProtoTxtFile(filepath, net):
    file = open(filepath, "w")
    if not file:
        raise IOError("ERROR (" + filepath + ")!")
    file.write(str(net))
    file.close()


def pca_cal(weight, type):
    if type == "Convolution":
        weight = weight.transpose((1, 2, 0, 3)).reshape(weight.shape[1] * weight.shape[2], -1)
    else:
        weight = weight.transpose((1, 0))
    cov = weight.T.dot(weight)
    cov = cov / (weight.shape[0] - 1)
    w, v = np.linalg.eig(cov)
    sort_w = sorted(w, reverse=True)
    sum = 0
    EN = np.sum(w)
    rank = 0
    for i in range(weight.shape[1]):
        sum += sort_w[i]
        if sum / EN > PCA:
            rank = i + 1
            break

    return rank

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='decom layer')
    parser.add_argument('--model', type=str, required=True, help="The original network structure.")
    parser.add_argument('--weights', type=str, required=True, help='The caffemodel to decom.')
    parser.add_argument('--outputmodel', type=str, required=True, help='The decom network structure.')
    parser.add_argument('--outputweights', type=str, required=True, help='Save the decom model of the directory')
    parser.add_argument('--pca', type=float, required=True, help='PCA energy percentage.')

    args = parser.parse_args()

    model = caffe.Net(args.model, args.weights, caffe.TEST)
    PCA = args.pca
    net = readProtoTxtFile(args.model)
    oriNet = read_caffemodel(args.weights)

    print(model.params.keys())
    print("read done")

    new_net = caffe.proto.caffe_pb2.NetParameter()

    for i, n in enumerate(net.layer):
        if n.type != "Convolution" and n.type != "InnerProduct":
            layer = caffe.proto.caffe_pb2.LayerParameter()
            layer.CopyFrom(n)
            new_net.layer.extend([layer])
        elif n.type == "Convolution":
            weight = model.params[n.name][0].data
            rank = pca_cal(weight, n.type)

            print(n.name, rank)

            cp = [v[0].name for v in n.convolution_param.ListFields()]
            layer_up = caffe.proto.caffe_pb2.LayerParameter()
            layer_up.CopyFrom(n)
            layer_up.name = n.name + "_up"
            layer_up.type = "Convolution"
            layer_up.top[0] = n.top[0] + "_up"
            layer_up.convolution_param.num_output = rank
            if 'pad' in cp:
                layer_up.convolution_param.pad_h = n.convolution_param.pad[0]
                layer_up.convolution_param.ClearField("pad")
            if 'stride' in cp:
                layer_up.convolution_param.stride_h = n.convolution_param.stride[0]
                layer_up.convolution_param.stride_w = 1
                layer_up.convolution_param.ClearField("stride")
            if 'kernel_size' in cp:
                layer_up.convolution_param.kernel_h = n.convolution_param.kernel_size[0]
                layer_up.convolution_param.kernel_w = 1
                layer_up.convolution_param.ClearField("kernel_size")

            layer_down = caffe.proto.caffe_pb2.LayerParameter()
            layer_down.CopyFrom(n)
            layer_down.name = n.name + "_down"
            layer_down.type = "Convolution"
            layer_down.bottom[0] = n.top[0] + "_up"
            layer_down.top[0] = n.top[0]
            layer_down.convolution_param.num_output = n.convolution_param.num_output
            if 'pad' in cp:
                layer_down.convolution_param.pad_w = n.convolution_param.pad[0]
                layer_down.convolution_param.ClearField("pad")
            if 'stride' in cp:
                layer_down.convolution_param.stride_w = n.convolution_param.stride[0]
                layer_down.convolution_param.stride_h = 1
                layer_down.convolution_param.ClearField("stride")
            if 'kernel_size' in cp:
                layer_down.convolution_param.kernel_w = n.convolution_param.kernel_size[0]
                layer_down.convolution_param.kernel_h = 1
                layer_down.convolution_param.ClearField("kernel_size")

            new_net.layer.extend([layer_up, layer_down])

            oriW = np.array(model.params[n.name][0].data)
            if n.convolution_param.HasField("bias_term"):
                if n.convolution_param.bias_term == True:
                    orib = np.array(model.params[n.name][1].data)
                else:
                    orib = None
            else:
                orib = None

            accW1, accb1, accW2, accb2 = svd_decom(oriW, orib, rank, n.type)

            layer_up = genLayer(n.name + "_up", n.type, accW1, accb1)
            layer_down = genLayer(n.name + "_down", n.type, accW2, accb2)

            oriNet.layer.extend([layer_up, layer_down])
            oriNet = delLayer(oriNet, n.name)

        else:
            weight = model.params[n.name][0].data
            rank = pca_cal(weight, n.type)
            print(n.name, rank)

            layer_up = caffe.proto.caffe_pb2.LayerParameter()
            layer_up.CopyFrom(n)
            layer_up.name = n.name + "_up"
            layer_up.type = "InnerProduct"
            layer_up.top[0] = n.top[0] + "_up"
            layer_up.inner_product_param.num_output = rank

            layer_down = caffe.proto.caffe_pb2.LayerParameter()
            layer_down.CopyFrom(n)
            layer_down.name = n.name + "_down"
            layer_down.type = "InnerProduct"
            layer_down.bottom[0] = n.top[0] + "_up"
            layer_down.top[0] = n.top[0]
            layer_down.inner_product_param.num_output = n.inner_product_param.num_output

            new_net.layer.extend([layer_up, layer_down])

            oriW = np.array(model.params[n.name][0].data)
            if n.inner_product_param.HasField("bias_term"):
                if n.inner_product_param.bias_term == True:
                    orib = np.array(model.params[n.name][1].data)
                else:
                    orib = None
            else:
                orib = None

            accW1, accb1, accW2, accb2 = svd_decom(oriW, orib, rank, n.type)

            layer_up = genLayer(n.name + "_up", n.type, accW1, accb1)
            layer_down = genLayer(n.name + "_down", n.type, accW2, accb2)

            oriNet.layer.extend([layer_up, layer_down])
            oriNet = delLayer(oriNet, n.name)

    writeProtoTxtFile(args.outputmodel, new_net)
    saveModel(args.outputweights, oriNet)
import random
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.models import Model


# util function to convert a tensor into a valid image
def deprocess_image(x):
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape(x.shape[1], x.shape[2])  # original shape (1,img_rows, img_cols,1)


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads


def constraint_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = np.mean(gradients)
    return grad_mean * new_grads


def constraint_black(gradients, rect_shape=(6, 6)):
    start_point = (
        random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads


def init_path_coverage_tables(model1, model2, model3):
    path_coverage_dict1 = defaultdict(bool)
    path_coverage_dict2 = defaultdict(bool)
    path_coverage_dict3 = defaultdict(bool)
    init_path_dict(model1, path_coverage_dict1)
    init_path_dict(model2, path_coverage_dict2)
    init_path_dict(model3, path_coverage_dict3)
    return path_coverage_dict1, path_coverage_dict2, path_coverage_dict3


def init_path_dict(model, path_coverage_dict):
    for i in range(len(model.layers) - 1):
        layer1 = model.layers[i]
        layer2 = model.layers[i + 1]
        
        if 'input' in layer1.name or 'block2_pool1' in layer1.name:
            continue
        
        layer1_neurons = layer1.output_shape[-1]
        layer2_neurons = layer2.output_shape[-1]
        
        if 'block1_conv1' in layer1.name or 'block2_conv1' in layer1.name or 'before_softmax' in layer1.name:
            for neuron_num in range(layer1_neurons):
                path_coverage_dict[((layer1.name, neuron_num), (layer2.name, neuron_num))] = False
        else:
            for neuron1 in range(layer1_neurons):
                for neuron2 in range(layer2_neurons):
                    path_coverage_dict[((layer1.name, neuron1), (layer2.name, neuron2))] = False
    


def path_to_cover(path_coverage_dict):
    not_covered = [((layer1_name, neuron1), (layer2_name, neuron2)) for ((layer1_name, neuron1), (layer2_name, neuron2)), v in path_coverage_dict.items() if not v]
    if not_covered:
        (layer1_name, neuron1), (layer2_name, neuron2) = random.choice(not_covered)
    else:
        (layer1_name, neuron1), (layer2_name, neuron2) = random.choice(path_coverage_dict.keys())
    return (layer1_name, neuron1), (layer2_name, neuron2)


def path_covered(path_coverage_dict):
    covered_paths = len([v for v in path_coverage_dict.values() if v])
    total_paths = len(path_coverage_dict)
    return covered_paths, total_paths, covered_paths / float(total_paths)


def update_path_coverage(input_data, model, path_coverage_dict, threshold=0):
    layer_names = [layer.name for layer in model.layers if 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i in range(len(intermediate_layer_outputs) - 1):
        current_layer_output = scale(intermediate_layer_outputs[i][0])
        next_layer_output = scale(intermediate_layer_outputs[i + 1][0])
        
        if layer_names[i] == 'block2_pool1':
            continue
        
        if layer_names[i] == 'block1_conv1' or layer_names[i] == 'block2_conv1' or layer_names[i] == 'before_softmax':
            for neuron_num in range(current_layer_output.shape[-1]):
                if np.mean(current_layer_output[..., neuron_num]) > threshold and np.mean(next_layer_output[..., neuron_num]) > threshold:
                    path = ((layer_names[i], neuron_num), (layer_names[i+1], neuron_num))
                    if not path_coverage_dict.get(path, False):
                        path_coverage_dict[path] = True
        else:
            for neuron_1 in range(current_layer_output.shape[-1]):
                if np.mean(current_layer_output[..., neuron_1]) > threshold:
                    for neuron_2 in range(next_layer_output.shape[-1]):
                        if np.mean(next_layer_output[..., neuron_2]) > threshold:
                            path = ((layer_names[i], neuron_1), (layer_names[i+1], neuron_2))
                            if not path_coverage_dict.get(path, False):
                                path_coverage_dict[path] = True


def full_coverage(model_layer_dict):
    if False in model_layer_dict.values():
        return False
    return True


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def fired(model, layer_name, index, input_data, threshold=0):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
    scaled = scale(intermediate_layer_output)
    if np.mean(scaled[..., index]) > threshold:
        return True
    return False


def diverged(predictions1, predictions2, predictions3, target):
    #     if predictions2 == predictions3 == target and predictions1 != target:
    if not predictions1 == predictions2 == predictions3:
        return True
    return False

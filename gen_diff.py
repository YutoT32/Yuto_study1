'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse
import ast

from keras.datasets import mnist
from keras.layers import Input
from scipy.misc import imsave

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from configs import bcolors
from utils import *

import csv

import time

def parse_tuple(input_str):
    try:
        return tuple(ast.literal_eval(input_str))
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Invalid tuple format: {}".format(input_str))

start = time.time()

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])
parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('target_model_diffact', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=0, type=int)
parser.add_argument('target_model_pc', help="target model that we want it increace path coverage",
                    choices=[0, 1, 2], default=0, type=int)
parser.add_argument('start_point', help="occlusion upper left corner coordinate", type=parse_tuple)
parser.add_argument('occlusion_size', help="occlusion size", type=parse_tuple)

args = parser.parse_args()

# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(_, _), (x_test, _) = mnist.load_data()

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
model1 = Model1(input_tensor=input_tensor)
model2 = Model2(input_tensor=input_tensor)
model3 = Model3(input_tensor=input_tensor)

# init coverage table
path_coverage_dict1, path_coverage_dict2, path_coverage_dict3 = init_path_coverage_tables(model1, model2, model3)

# ==============================================================================================
# start gen inputs

num_gen_img = 0
num_alr_img = 0

for no_seeds in xrange(args.seeds):
    gen_img = np.expand_dims(random.choice(x_test), axis=0)
    orig_img = gen_img.copy()
    
    # first check if input already induces differences
    label1, label2, label3 = np.argmax(model1.predict(gen_img)[0]), np.argmax(model2.predict(gen_img)[0]), np.argmax(
        model3.predict(gen_img)[0])

    if not label1 == label2 == label3:
        '''
        print(bcolors.OKGREEN + 'input already causes different outputs: {}, {}, {}'.format(label1, label2,
                                                                                            label3) + bcolors.ENDC)
        '''
        update_path_coverage(gen_img, model1, path_coverage_dict1, args.threshold)
        update_path_coverage(gen_img, model2, path_coverage_dict2, args.threshold)
        update_path_coverage(gen_img, model3, path_coverage_dict3, args.threshold)
        '''
        print(bcolors.OKGREEN + 'covered paths percentage %d paths %.3f, %d paths %.3f, %d paths %.3f'
              % (len(path_coverage_dict1), path_covered(path_coverage_dict1)[2], len(path_coverage_dict2),
                 path_covered(path_coverage_dict2)[2], len(path_coverage_dict3),
                 path_covered(path_coverage_dict3)[2]) + bcolors.ENDC)
        averaged_pc = (path_covered(path_coverage_dict1)[0] + path_covered(path_coverage_dict2)[0] +
                       path_covered(path_coverage_dict3)[0]) / float(
            path_covered(path_coverage_dict1)[1] + path_covered(path_coverage_dict2)[1] +
            path_covered(path_coverage_dict3)[1])
        print(bcolors.OKGREEN + 'averaged covered paths %.3f' % averaged_pc + bcolors.ENDC)
        '''
        gen_img_deprocessed = deprocess_image(gen_img)

        # save the result to disk
        imsave('./generated_inputs/' + 'already_differ_' + str(label1) + '_' + str(
            label2) + '_' + str(label3) + '.png', gen_img_deprocessed)
        
        num_alr_img += 1
        
        print(no_seeds)
        
        continue

    # if all label agrees
    orig_label = label1
    (layer1_name1, neuron11), (layer2_name1, neuron21) = path_to_cover(path_coverage_dict1)
    (layer1_name2, neuron12), (layer2_name2, neuron22) = path_to_cover(path_coverage_dict2)
    (layer1_name3, neuron13), (layer2_name3, neuron23) = path_to_cover(path_coverage_dict3)

    # construct joint loss function
    if args.target_model_diffact == 0:
        loss1 = -args.weight_diff * K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = K.mean(model3.get_layer('before_softmax').output[..., orig_label])
    elif args.target_model_diffact == 1:
        loss1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = -args.weight_diff * K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = K.mean(model3.get_layer('before_softmax').output[..., orig_label])
    elif args.target_model_diffact == 2:
        loss1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = -args.weight_diff * K.mean(model3.get_layer('before_softmax').output[..., orig_label])
    
    loss1_neuron_a = K.mean(model1.get_layer(layer1_name1).output[..., neuron11])
    loss1_neuron_b = K.mean(model1.get_layer(layer2_name1).output[..., neuron21])
    loss2_neuron_a = K.mean(model2.get_layer(layer1_name2).output[..., neuron12])
    loss2_neuron_b = K.mean(model2.get_layer(layer2_name2).output[..., neuron22])
    loss3_neuron_a = K.mean(model3.get_layer(layer1_name3).output[..., neuron13])
    loss3_neuron_b = K.mean(model3.get_layer(layer2_name3).output[..., neuron23])
    
    if args.target_model_pc == 0:
        layer_output = (loss1 + loss2 + loss3) + args.weight_nc * (loss1_neuron_a + loss1_neuron_b + loss2_neuron_a + loss2_neuron_b + loss3_neuron_a + loss3_neuron_b)
    elif args.target_model_pc == 1:
        layer_output = (loss1 + loss2 + loss3) + args.weight_nc * (loss1_neuron_a + loss1_neuron_b + loss3_neuron_a + loss3_neuron_b)
    elif args.target_model_pc == 2:
        layer_output = (loss1 + loss2 + loss3) + args.weight_nc * (loss1_neuron_a + loss1_neuron_b + loss2_neuron_a + loss2_neuron_b)

    # for adversarial image generation
    final_loss = K.mean(layer_output)

    # we compute the gradient of the input picture wrt this loss
    grads = normalize(K.gradients(final_loss, input_tensor)[0])

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_tensor], [loss1, loss2, loss3, loss1_neuron_a, loss1_neuron_b, loss2_neuron_a, loss2_neuron_b, loss3_neuron_a, loss3_neuron_b, grads])

    # we run gradient ascent for 20 steps
    for iters in xrange(args.grad_iterations):
        loss_value1, loss_value2, loss_value3, loss_neuron1_a, loss_neuron1_b, loss_neuron2_a, loss_neuron2_b, loss_neuron3_a, loss_neuron3_b, grads_value = iterate([gen_img])
        if args.transformation == 'light':
            grads_value = constraint_light(grads_value)  # constraint the gradients value
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point,
                                          args.occlusion_size)  # constraint the gradients value
        elif args.transformation == 'blackout':
            grads_value = constraint_black(grads_value)  # constraint the gradients value

        gen_img += grads_value * args.step
        predictions1 = np.argmax(model1.predict(gen_img)[0])
        predictions2 = np.argmax(model2.predict(gen_img)[0])
        predictions3 = np.argmax(model3.predict(gen_img)[0])

        if not predictions1 == predictions2 == predictions3:
            update_path_coverage(gen_img, model1, path_coverage_dict1, args.threshold)
            update_path_coverage(gen_img, model2, path_coverage_dict2, args.threshold)
            update_path_coverage(gen_img, model3, path_coverage_dict3, args.threshold)
            '''
            print(bcolors.OKGREEN + 'covered paths percentage %d paths %.3f, %d paths %.3f, %d paths %.3f'
                  % (len(path_coverage_dict1), path_covered(path_coverage_dict1)[2], len(path_coverage_dict2),
                     path_covered(path_coverage_dict2)[2], len(path_coverage_dict3),
                     path_covered(path_coverage_dict3)[2]) + bcolors.ENDC)
            averaged_pc = (path_covered(path_coverage_dict1)[0] + path_covered(path_coverage_dict2)[0] +
                           path_covered(path_coverage_dict3)[0]) / float(
                path_covered(path_coverage_dict1)[1] + path_covered(path_coverage_dict2)[1] +
                path_covered(path_coverage_dict3)[1])
            print(bcolors.OKGREEN + 'averaged covered paths %.3f' % averaged_pc + bcolors.ENDC)
            '''
            gen_img_deprocessed = deprocess_image(gen_img)
            orig_img_deprocessed = deprocess_image(orig_img)

            # save the result to disk
            imsave('./generated_inputs/' + args.transformation + '_' + str(predictions1) + '_' + str(
                predictions2) + '_' + str(predictions3) + '.png',
                   gen_img_deprocessed)
            imsave('./generated_inputs/' + args.transformation + '_' + str(predictions1) + '_' + str(
                predictions2) + '_' + str(predictions3) + '_orig.png',
                   orig_img_deprocessed)
            
            num_gen_img += 1
            
            gen_img_deprocessed = gen_img_deprocessed.astype(np.float32)
            orig_img_deprocessed = orig_img_deprocessed.astype(np.float32)
            
            l1_distance = np.sum(np.abs(gen_img_deprocessed -orig_img_deprocessed))
            l2_distance = np.sqrt(np.sum(np.square(gen_img_deprocessed - orig_img_deprocessed)))
            
            averaged_pc = (path_covered(path_coverage_dict1)[0] + path_covered(path_coverage_dict2)[0] +
                           path_covered(path_coverage_dict3)[0]) / float(
                path_covered(path_coverage_dict1)[1] + path_covered(path_coverage_dict2)[1] +
                path_covered(path_coverage_dict3)[1])
            
            new_data_1 = [args.seeds, num_gen_img, l1_distance, l2_distance, path_covered(path_coverage_dict1)[2], path_covered(path_coverage_dict2)[2], path_covered(path_coverage_dict3)[2], averaged_pc]
            csv_file = "results_extra.csv"
            with open(csv_file, "a") as file:
                writer = csv.writer(file)
                writer.writerow(new_data_1)
            
            print(no_seeds)
            
            break

end = time.time()

print('--end--')

new_data_2 = [args.seeds, num_gen_img, num_alr_img, end - start]
csv_file = "results.csv"
with open(csv_file, "a") as file:
    writer = csv.writer(file)
    writer.writerow(new_data_2)
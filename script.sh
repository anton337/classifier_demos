#export PROJECT=spiral_fitting
#export PROJECT=simple_ann_classification
#export PROJECT=xor_tutorial
#export PROJECT=signal_tracking
#export PROJECT=rbm_single_layer_example
#export PROJECT=auto_encoder_image_segmentation
#export PROJECT=mnist_classifier
#export PROJECT=convolutional_neural_network
#export PROJECT=sep_classifier
#export PROJECT=cifar_classifier
#export PROJECT=salt_classifier
#export PROJECT=convolutional_rbm
#export PROJECT=afi_convolutional_rbm
#export PROJECT=convolutional_rbm_auto_encoder
export PROJECT=cat_classifier

./clean.sh

./build.sh

./run_project.sh

#./debug_project.sh


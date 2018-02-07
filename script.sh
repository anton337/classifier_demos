#export PROJECT=spiral_fitting
#export PROJECT=simple_ann_classification
#export PROJECT=xor_tutorial
#export PROJECT=signal_tracking
#export PROJECT=rbm_single_layer_example
export PROJECT=auto_encoder_image_segmentation

./clean.sh

./build.sh

./run_project.sh

#./debug_project.sh


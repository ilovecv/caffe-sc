#!/bin/sh
curau=$1
setnum=$2
datadir=$3
resdir=$4
foldname=$5
FILE="$resdir/solver${curau}.prototxt"
/bin/cat <<EOM >$FILE
net: "$resdir/train_val${curau}.prototxt"
test_iter: 145
# Carry out testing every 500 training iterations.
test_interval: 200

test_compute_loss: true
# The base learning rate, momentum and the weight decay of the network.
base_lr: 0.0005
momentum: 0.9
weight_decay: 0.06
# The learning rate policy
lr_policy: "step"
gamma: 0.6
stepsize: 400
# Display every 100 iterations
display: 200
# The maximum number of iterations
max_iter: 3000
# snapshot intermediate results
snapshot: 200
snapshot_prefix: "$resdir/train${curau}_"

# solver mode: CPU or GPU
solver_mode: GPU
EOM


FILE="$resdir/train_val${curau}.prototxt"
/bin/cat <<EOM >$FILE
name: "CIFAR10_quick"
layer {
  name: "cifar"
  type: "ImageData"
  top: "data"
  top: "label"
  transform_param {
    mirror: 1
    crop_height: 122
    crop_width: 90
    rotation: 10
  }
  image_data_param {
    image_num: 1
    label_dim: 1
    #new_height: 128
    #new_width: 96
    root_folder: "$datadir/$foldname/"
    source: "$datadir/training_set${setnum}_AU${curau}.txt"
    balance: true
    balance_axis: 0
    balance_coeff: 0.2
    batch_size: 100 
    is_color: false
    shuffle: true
  }
  include: { phase: TRAIN }
}
layer {
  name: "cifar"
  type: "ImageData"
  top: "data"
  top: "label"
  transform_param {
    mirror: 0
    crop_height: 122
    crop_width: 90 
  }
  image_data_param {
    image_num: 1
    label_dim: 1
    #new_height: 128
    #ls new_width: 96
    root_folder: "$datadir/$foldname/"
    source: "$datadir/develop_set${setnum}_AU${curau}.txt"
    balance: false
    balance_axis: 0
    balance_coeff: 0.2
    batch_size: 100 
    is_color: false
    shuffle: false
  }

  include: { phase: TEST }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param { lr_mult: 1 decay_mult: 1 }
  param { lr_mult: 2 decay_mult: 0 }
  convolution_param {
    num_output: 32
    pad: 0
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu1"
  type: "PReLU"
  bottom: "conv1"
  top: "conv1"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: AVE
    kernel_size: 3
    stride: 3
  }
}
layer {
  name: "norm1"
  type: "LRN"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
  bottom: "pool1"
  top: "norm1"
}

layer {
  name: "conv2"
  type: "Convolution"
  bottom: "norm1"
  top: "conv2"
  param { lr_mult: 1 decay_mult: 1 }
  param { lr_mult: 2 decay_mult: 0 }
  convolution_param {
    num_output: 32
    pad: 0
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu2"
  type: "PReLU"
  bottom: "conv2"
  top: "conv2"
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: AVE
    kernel_size: 3
    stride: 3
  }
}
layer {
  name: "norm2"
  type: "LRN"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
  bottom: "pool2"
  top: "norm2"
}
layer {
  name: "conv3"
  type: "Convolution"
  bottom: "norm2"
  top: "conv3"
  param { lr_mult: 1 decay_mult: 1 }
  param { lr_mult: 2 decay_mult: 0 }
  convolution_param {
    num_output: 64
    pad: 0
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu3"
  type: "PReLU"
  bottom: "conv3"
  top: "conv3"
}
layer {
  name: "norm3"
  type: "LRN"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
  bottom: "conv3"
  top: "norm3"
}
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "norm3"
  top: "ip1"
  param { lr_mult: 1 decay_mult: 1 }
  param { lr_mult: 2 decay_mult: 0 }
  inner_product_param {
    num_output: 128
    weight_filler {
      type: "xavier"
    }
  }
}
#layer {
#  name: "relu4"
#  type: "PReLU"
#  bottom: "ip1"
#  top: "relu4"
#}
#layer {
#  name: "drop1"
#  type: "Dropout"
#  bottom: "relu4"
#  top: "drop1"
#  dropout_param {
#    dropout_ratio: 0.5
#  }
#}
layer {
  name: "ip2"
  type: "InnerProduct"
  bottom: "ip1"
  top: "ip2"
  param { lr_mult: 1 decay_mult: 1 }
  param { lr_mult: 2 decay_mult: 0 }
  inner_product_param {
    num_output: 1
    weight_filler {
      type: "xavier"
    }
  }
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "ip2"
  bottom: "label"
  top: "accuracy"
  #include: { phase: TEST }
}
layer {
  name: "loss"
  type: "SigmoidCrossEntropyLoss"
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}
layer {
  name: "score"
  type: "Score"
  bottom: "ip2"
  bottom: "label"
  top: "weight_rate"
  score_param {
      dest_file: "$resdir/outscore_baseline${curau}.txt"
  }
  include: { phase: TEST }

}
EOM

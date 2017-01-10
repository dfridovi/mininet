/*
 * Copyright (c) 2017, David Fridovich-Keil.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Please contact the author(s) of this library if you have any questions.
 * Authors: David Fridovich-Keil   ( dfk@eecs.berkeley.edu )
 */

///////////////////////////////////////////////////////////////////////////////
//
// Learns to classify a point in the plane according to which quadrant it
// lies in.
//
///////////////////////////////////////////////////////////////////////////////

#include <layer/relu.h>
#include <layer/layer_params.h>
#include <loss/cross_entropy.h>
#include <net/network.h>
#include <dataset/dataset.h>
#include <trainer/backprop_trainer.h>
#include <trainer/backprop_params.h>

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>

// Network size.
DEFINE_int32(num_hidden_layers, 1, "Number of hidden layers.");
DEFINE_int32(hidden_layer_size, 200, "Number of outputs of each hidden layer.");

// Dataset parameters.
DEFINE_double(training_fraction, 0.75,
              "Fraction of dataset to be used for training.");
DEFINE_string(data_file, "mnist.csv",
              "Name of the file containing the dataset.");

// Training parameters.
DEFINE_double(max_avg_loss, 1e-4, "Maximum allowable mean loss for a batch.");
DEFINE_int32(batch_size, 20, "Amount of data to be used per SGD iteration.");
DEFINE_int32(num_iters_per_epoch, 1000, "Number of iterations per epoch.");
DEFINE_int32(num_epochs, 10, "Number of epochs to run SGD.");
DEFINE_double(learning_rate, 0.2, "Initial learning rate.");
DEFINE_double(learning_rate_decay, 0.5, "Learning rate decay factor.");

using namespace mininet;

int main(int argc, char** argv) {
  // Set up logging and flags.
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  // Check valid parameters.
  CHECK(FLAGS_num_hidden_layers > 0);

  // Load the dataset.
  std::cout << "Loading the dataset..." << std::flush;

  std::vector<VectorXd> inputs, outputs;
  std::ifstream file(MININET_DATA_DIR + FLAGS_data_file);

  int label = -1;
  size_t position;
  std::vector<int> line;
  while (file.good()) {
    line.clear();

    // Get label.
    std::string label_string;
    std::getline(file, label_string, ',');
    try {
      label = std::stoi(label_string);
    } catch (...) {
      LOG(WARNING) << "Error parsing a label. String was: " << label_string;
      label = 0;
    }
    CHECK(label >= 0 && label <= 9);

    // Parse the rest of the line.
    std::string data_string;
    std::getline(file, data_string);
    while ((position = data_string.find(',')) != std::string::npos) {
      std::string field = data_string.substr(0, position);
      data_string = data_string.substr(position + 1);

      try {
        line.push_back(std::stoi(field));
      } catch (...) {
        LOG(WARNING) << "Error parsing data field. String was: " << field;
        line.push_back(0);
      }

    }

    // Catch remaining characters.
    if (data_string.length() > 0) {
      try {
        line.push_back(std::stoi(data_string));
      } catch (...) {
        LOG(WARNING) << "Error parsing data field. String was: " << data_string;
        line.push_back(0);
      }
    }

    // Catch empty line.
    if (line.size() == 0)
      continue;

    // Parse label and line into VectorXd's.
    VectorXd output = VectorXd::Zero(10);
    output(label) = 1.0;

    VectorXd input(line.size());
    for (size_t ii = 0; ii < line.size(); ii++)
      input(ii) = static_cast<double>(line[ii]);

    // Check that the input size matches the previous line.
    if (inputs.size() > 0)
      CHECK_EQ(inputs.back().rows(), input.rows());

    // Insert into inputs and outputs.
    outputs.push_back(output);
    inputs.push_back(input);
  }

  file.close();
  Dataset dataset(inputs, outputs, FLAGS_training_fraction);

  std::cout << "done. Found " << inputs.size() << " data points." << std::endl;

  // Set up the network.
  std::cout << "Setting up the network..." << std::flush;
  std::vector<LayerParams> layer_params;
  layer_params.push_back(LayerParams(RELU, inputs[0].rows(),
                                     FLAGS_hidden_layer_size));

  for (size_t ii = 1; ii < FLAGS_num_hidden_layers; ii++)
    layer_params.push_back(LayerParams(RELU, FLAGS_hidden_layer_size,
                                       FLAGS_hidden_layer_size));

  layer_params.push_back(LayerParams(SOFTMAX, FLAGS_hidden_layer_size, 10));

  const LossFunctor::ConstPtr loss = CrossEntropy::Create();
  Network net(layer_params, loss);

  std::cout << "done." << std::endl << std::flush;

  // Create a trainer.
  BackpropParams backprop_params;
  backprop_params.max_avg_loss_ = FLAGS_max_avg_loss;
  backprop_params.batch_size_ = FLAGS_batch_size;
  backprop_params.num_iters_per_epoch_ = FLAGS_num_iters_per_epoch;
  backprop_params.num_epochs_ = FLAGS_num_epochs;
  backprop_params.learning_rate_ = FLAGS_learning_rate;
  backprop_params.learning_rate_decay_ = FLAGS_learning_rate_decay;

  BackpropTrainer trainer(net, dataset, backprop_params);

  // Train.
  trainer.Train();

  // Test.
  const double final_loss = trainer.Test();
  std::printf("Final loss was %f.\n", final_loss);

  return 0;
}

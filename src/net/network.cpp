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
 * Author: David Fridovich-Keil   ( dfk@eecs.berkeley.edu )
 *         Sara Fridovich-Keil    ( saraf@princeton.edu )
 */

///////////////////////////////////////////////////////////////////////////////
//
// Defines the Network class.
//
///////////////////////////////////////////////////////////////////////////////

#include <net/network.h>

#include <glog/logging.h>

namespace mininet {

Network::~Network() {}
Network::Network(std::vector<LayerParams> params,
                 const LossFunctor::ConstPtr& loss)
  : loss_(loss) {
  CHECK_NOTNULL(loss.get());
  CHECK_GE(params.size(), 1);

  for (size_t ii = 0; ii < params.size(); ii++) {
    Layer::Ptr layer;

    // Create the new hidden layer.
    switch (params[ii].type_) {
    case LINEAR :
      layer = Linear::Create(params[ii].input_size_, params[ii].output_size_);
      break;
    case RELU :
      layer = ReLU::Create(params[ii].input_size_, params[ii].output_size_);
      break;
    case SIGMOID :
      layer = Sigmoid::Create(params[ii].input_size_, params[ii].output_size_);
      break;
    case SOFTMAX :
      layer = Softmax::Create(params[ii].input_size_, params[ii].output_size_);
      break;
    default:
      LOG(WARNING) << "Invalid hidden layer type. Using ReLU instead.";
      layer = ReLU::Create(params[ii].input_size_, params[ii].output_size_);
      break;
    }

    // Make sure its input size matches the previous output size.
    if (ii > 0)
      CHECK_EQ(params[ii].input_size_, params[ii - 1].output_size_);

    // Add to list.
    layers_.push_back(layer);
  }
}

// Copy constructor.
Network::Network(const Network& net)
  : loss_(net.loss_) {
  CHECK_NOTNULL(loss_.get());

  // Copy over layers.
  for (const auto& layer : net.layers_) {
    layers_.push_back(layer->Copy());
  }
}

// Treat the network as a functor. Computes the output of the net.
void Network::operator()(const VectorXd& input, VectorXd& output) const {
  CHECK_EQ(input.rows(), layers_.front()->InputSize());
  CHECK_EQ(output.rows(), layers_.back()->OutputSize());

  // Pass through the network.
  VectorXd current_output(input.size());
  VectorXd current_input = input;
  for (size_t ii = 0; ii < layers_.size(); ii++) {
    Layer::ConstPtr layer = layers_[ii];
    current_output.resize(layer->OutputSize());

    layer->Forward(current_input, current_output);
    current_input = current_output;
  }

  output = current_output;
}

// Compute average loss on a set of inputs and ground truths.
double Network::Loss(const std::vector<VectorXd>& inputs,
                     const std::vector<VectorXd>& ground_truths) const {
  CHECK_EQ(inputs.size(), ground_truths.size());

  // Forward and backward passes for each element of the inputs.
  double loss_sum = 0.0;
  for (size_t ii = 0; ii < inputs.size(); ii++) {
    std::vector<VectorXd> layer_inputs;
    Forward(inputs[ii], layer_inputs);

    std::vector<VectorXd> deltas;
    loss_sum += Backward(ground_truths[ii], layer_inputs, deltas);
  }

  // Return loss.
  return loss_sum /= static_cast<double>(inputs.size());
}

// Compute average layer inputs and deltas for a batch of inputs.
// Returns average loss.
double Network::RunBatch(const std::vector<VectorXd>& inputs,
                         const std::vector<VectorXd>& ground_truths,
                         std::vector<MatrixXd>& derivatives) const {
  // Keep track of running sums of derivatives and loss.
  derivatives.clear();
  double loss_sum = 0.0;

  // Set up derivative matrices.
  for (size_t ii = 0; ii < layers_.size(); ii++) {
    derivatives.push_back(MatrixXd::Zero(layers_[ii]->OutputSize(),
                                         layers_[ii]->InputSize() + 1));
  }

  // Forward and backward passes for each element of the inputs.
  for (size_t ll = 0; ll < inputs.size(); ll++) {
    std::vector<VectorXd> layer_inputs;
    Forward(inputs[ll], layer_inputs);

    std::vector<VectorXd> deltas;
    loss_sum += Backward(ground_truths[ll], layer_inputs, deltas);

    CHECK_EQ(layer_inputs.size(), layers_.size() + 1);
    CHECK_EQ(deltas.size(), layers_.size());

    // Compute derivatives with respect to each layer's weights.
    for (size_t kk = 0; kk < layers_.size(); kk++) {
      for (size_t jj = 0; jj < derivatives[kk].cols() - 1; jj++)
        for (size_t ii = 0; ii < derivatives[kk].rows(); ii++)
          derivatives[kk](ii, jj) += layer_inputs[kk](jj) * deltas[kk](ii);

      // Catch bias weights.
      for (size_t ii = 0; ii < derivatives[kk].rows(); ii++)
        derivatives[kk].rightCols(1)(ii) += deltas[kk](ii);

      // Average.
      derivatives[kk] /= static_cast<double>(inputs.size());
    }
  }

  return loss_sum /= static_cast<double>(inputs.size());
}

// Update weights.
void Network::UpdateWeights(const std::vector<MatrixXd>& derivatives,
                            double learning_rate, double momentum,
                            double decay) {
  CHECK_EQ(derivatives.size(), layers_.size());

  for (size_t ii = 0; ii < layers_.size(); ii++)
    layers_[ii]->UpdateWeights(derivatives[ii], learning_rate,
                               momentum, decay);
}

// Forward pass: compute the input of each layer. Last entry is output of final
// layer.
void Network::Forward(const VectorXd& input,
                      std::vector<VectorXd>& layer_inputs) const {
  CHECK_EQ(input.rows(), layers_.front()->InputSize());
  layer_inputs.clear();

  // Pass through the network.
  layer_inputs.push_back(input);
  for (size_t ii = 0; ii < layers_.size(); ii++) {
    Layer::ConstPtr layer = layers_[ii];
    VectorXd output(layer->OutputSize());

    layer->Forward(layer_inputs[ii], output);
    layer_inputs.push_back(output);
  }
}

// Backward pass: compute the 'deltas', i.e. the derivatives of loss by each
// successive layer's outputs. Returns loss.
double Network::Backward(const VectorXd& ground_truth,
                         const std::vector<VectorXd>& layer_inputs,
                         std::vector<VectorXd>& deltas) const {
  CHECK_EQ(ground_truth.rows(), layers_.back()->OutputSize());
  CHECK_EQ(layer_inputs.size(), layers_.size() + 1);
  deltas.clear();

  // Initialize 'deltas' to be the right size.
  deltas.resize(layers_.size());

  // Start at the output layer.
  VectorXd gamma(layers_.back()->InputSize());
  VectorXd delta(layers_.back()->OutputSize());

  const double loss =
    layers_.back()->Backward(loss_, ground_truth,
                             layer_inputs.back(), gamma, delta);
  deltas.back() = delta;

  // Propagate derivatives backward.
  for (int ii = layers_.size() - 2; ii >= 0; ii--) {
    Layer::ConstPtr layer = layers_[ii];
    VectorXd next_gamma(layer->InputSize());
    VectorXd next_delta(layer->OutputSize());

    layer->Backward(layer_inputs[ii + 1], gamma,
                    next_gamma, next_delta);
    deltas[ii] = next_delta;

    gamma.resize(next_gamma.rows());
    gamma = next_gamma;
  }

  // Return loss.
  return loss;
}

// Perturb a specific weight.
void Network::PerturbWeight(size_t layer_number, size_t ii, size_t jj,
                            double amount) {
  CHECK_LT(layer_number, layers_.size());

  layers_[layer_number]->PerturbWeight(ii, jj, amount);
}

} // namespace mininet

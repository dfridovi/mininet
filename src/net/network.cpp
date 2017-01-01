/*
 * Copyright (c) 2015, The Regents of the University of California (Regents).
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
Network::Network(std::vector<LayerParameters> params,
                 const LossFunctor::ConstPtr& loss)
  : loss_(loss) {
  CHECK_NOTNULL(loss.get());

  for (size_t ii = 0; ii < params.size() - 1; ii++) {
    HiddenLayer::Ptr layer;

    // Create the new hidden layer.
    switch (params[ii].type_) {
    case RELU :
      layer = ReLU::Create(params[ii].input_size_, params[ii].output_size_);
      break;
    default:
      VLOG(1) << "Invalid hidden layer type. Using ReLU instead.";
      layer = ReLU::Create(params[ii].input_size_, params[ii].output_size_);
      break;
    }

    // Make sure its input size matches the previous output size.
    if (ii > 0)
      CHECK(params[ii].input_size_ == params[ii - 1].output_size_);

    // Add to list.
    hidden_layers_.push_back(layer);
  }

  // Add output layer.
  switch (params.back().type_) {
  case SOFTMAX :
    output_layer_ = Softmax::Create(params.back().input_size_,
                                    params.back().output_size_);
    break;
  default:
    VLOG(1) << "Invalid output layer type. Using softmax instead.";
    output_layer_ = Softmax::Create(params.back().input_size_,
                                    params.back().output_size_);
    break;
  }

  // Make sure its input size matches the previous output size.
  if (params.size() > 1)
    CHECK(params.back().input_size_ == params[params.size() - 2].output_size_);
}

// Treat the network as a functor. Computes the output of the net.
void Network::operator()(const VectorXd& input, VectorXd& output) const {
  CHECK(input.rows() == hidden_layers_.front()->InputSize());
  CHECK(output.rows() == output_layer_->OutputSize());

  // Pass through the network.
  VectorXd current_output(input.size());
  VectorXd current_input = input;
  for (size_t ii = 0; ii < hidden_layers_.size(); ii++) {
    HiddenLayer::ConstPtr layer = hidden_layers_[ii];
    current_output.resize(layer->OutputSize());

    layer->Forward(current_input, current_output);
    current_input = current_output;
  }

  // Pass through output layer.
  output_layer_->Forward(current_input, output);
}

// Update weights. Returns current loss.
double Network::UpdateWeights(const VectorXd& input, const VectorXd& ground_truth,
                              double step_size) {
  // Foward and backward passes.
  std::vector<VectorXd> layer_inputs;
  Forward(input, layer_inputs);

  std::vector<VectorXd> deltas;
  const double loss = Backward(ground_truth, layer_inputs, deltas);

  CHECK(layer_inputs.size() == hidden_layers_.size() + 2);
  CHECK(deltas.size() == hidden_layers_.size() + 1);

  // Update all weights.
  for (size_t ii = 0; ii < hidden_layers_.size(); ii++)
    hidden_layers_[ii]->UpdateWeights(layer_inputs[ii], deltas[ii], step_size);

  output_layer_->UpdateWeights(layer_inputs.back(), deltas.back(), step_size);

  // Return loss.
  return loss;
}

// Forward pass: compute the input of each layer. Last entry is output of final
// layer.
void Network::Forward(const VectorXd& input,
                      std::vector<VectorXd>& layer_inputs) const {
  CHECK(input.rows() == hidden_layers_.front()->InputSize());
  layer_inputs.clear();

  // Pass through the network.
  layer_inputs.push_back(input);
  for (size_t ii = 0; ii < hidden_layers_.size(); ii++) {
    HiddenLayer::ConstPtr layer = hidden_layers_[ii];
    VectorXd output(layer->OutputSize());

    layer->Forward(layer_inputs[ii], output);
    layer_inputs.push_back(output);
  }

  VectorXd output(output_layer_->OutputSize());
  output_layer_->Forward(layer_inputs.back(), output);
  layer_inputs.push_back(output);
}

// Backward pass: compute the 'deltas', i.e. the derivatives of loss by each
// successive layer's outputs. Returns loss.
double Network::Backward(const VectorXd& ground_truth,
                         const std::vector<VectorXd>& layer_inputs,
                         std::vector<VectorXd>& deltas) const {
  CHECK(ground_truth.rows() == output_layer_->OutputSize());
  CHECK(layer_inputs.size() == hidden_layers_.size() + 2);
  deltas.clear();

  // Initialize 'deltas' to be the right size.
  deltas.resize(hidden_layers_.size() + 1);

  // Start at the output layer.
  VectorXd gamma(output_layer_->InputSize());
  VectorXd delta(output_layer_->OutputSize());
  const double loss = output_layer_->Backward(loss_, ground_truth,
                                              layer_inputs.back(),
                                              gamma, delta);
  deltas[hidden_layers_.size()] = delta;

  for (size_t ii = 0; ii < hidden_layers_.size(); ii++) {
    size_t reverse_ii = hidden_layers_.size() - 1 - ii;

    // Propagate derivatives backward.
    HiddenLayer::ConstPtr layer = hidden_layers_[reverse_ii];
    VectorXd next_gamma(layer->InputSize());
    VectorXd next_delta(layer->OutputSize());

    layer->Backward(layer_inputs[reverse_ii + 1], gamma, next_gamma, next_delta);
    deltas[reverse_ii] = next_delta;
    gamma = next_gamma;
  }

  // Return loss.
  return loss;
}

} // namespace mininet

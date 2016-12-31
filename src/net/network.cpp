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

namespace mininet {

Network::~Network() {}
Network::Network(std::vector<LayerParameters> params,
                 const LossFunctor& loss)
  : loss_(loss) {
  for (size_t ii = 0; ii < params.size(); ii++) {
    Layer::Ptr layer;

    // Create the new layer.
    switch (params[ii].type_) {
    case RELU :
      layer = ReLU::Create(params.input_size_, params.output_size_);
      break;
    case SOFTMAX :
      layer = Softmax::Create(params.input_size_, params.output_size_);
      break;
    default:
      VLOG(1) << "Invalid layer type. Using ReLU instead.";
      layer = ReLU::Create(params.input_size_, params.output_size_);
      break;
    }

    // Make sure its input size matches the previous output size.
    if (ii > 0)
      CHECK(params[ii].input_size_ == params[ii - 1].output_size_);

    // Add to list.
    layers_.push_back(layer);
  }
}

// Treat the network as a functor. Computes the output of the net.
void Network::operator()(const VectorXd& input, VectorXd& output) const {
  CHECK(input.rows() == layers_.back().InputSize());
  CHECK(output.rows() == layers_.back().OutputSize());

  // Pass through the network.
  VectorXd current_output(input.size());
  VectorXd current_input = input;
  for (size_t ii = 0; ii < layers_.size(); ii++) {
    Layer::ConstPtr layer = layers_[ii];
    current_output.resize(layer->OutputSize());

    layer->Forward(current_input, current_output);
    current_input = current_output;
  }

  // Set output.
  CHECK(output.rows() == current_output.rows());
  output = current_output;
}

// Update weights. Returns current loss.
double Network::UpdateWeights(const VectorXd& input, const VectorXd& ground_truth,
                              double step_size) {
  // Foward and backward passes.
  std::vector<VectorXd> layer_inputs;
  Forward(input, layer_inputs);

  std::vector<VectorXd> deltas;
  Backward(ground_truth, layer_inputs, deltas);

  // Update all weights.
  for (size_t ii = 0; ii < layers_.size(); ii++)
    layers_[ii]->UpdateWeights(layer_inputs[ii], deltas[ii], step_size);
}

// Forward pass: compute the output of each layer.
void Network::Forward(const VectorXd& input,
                      std::vector<VectorXd>& layer_inputs) const {
  CHECK(input.rows() == layers_.front().InputSize());
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

}

} // namespace mininet

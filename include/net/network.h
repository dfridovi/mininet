/*
 * Copyright (c) 2017. David Fridovich-Keil.
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

#ifndef MININET_NET_NETWORK_H
#define MININET_NET_NETWORK_H

#include <util/types.h>
#include <layer/softmax.h>
#include <layer/relu.h>
#include <layer/layer_params.h>
#include <loss/loss_functor.h>

#include <vector>

namespace mininet {

class Network {
public:
  explicit Network(std::vector<LayerParams> params,
                   const LossFunctor::ConstPtr& loss);
  ~Network();

  // Treat the network as a functor. Computes the output of the net.
  void operator()(const VectorXd& input, VectorXd& output) const;

  // Compute average loss on a set of inputs and ground truths.
  double Loss(const std::vector<VectorXd>& inputs,
              const std::vector<VectorXd>& ground_truths) const;

  // Compute average layer inputs and deltas for a batch of inputs.
  // Returns average loss.
  double RunBatch(const std::vector<VectorXd>& batch,
                  const std::vector<VectorXd>& ground_truth,
                  std::vector<VectorXd>& layer_inputs_avg,
                  std::vector<VectorXd>& deltas_avg) const;

  // Update weights.
  void UpdateWeights(const std::vector<VectorXd>& layer_inputs,
                     const std::vector<VectorXd>& deltas,
                     double step_size);

  // Perturb a specific weight.
  void PerturbWeight(size_t layer_number, size_t ii, size_t jj, double amount);

private:
  // Forward pass: compute the inputs of each layer (outputs of previous).
  void Forward(const VectorXd& input,
               std::vector<VectorXd>& layer_inputs) const;

  // Backward pass: compute the 'deltas', i.e. the derivatives of loss by each
  // successive layer's outputs. Returns loss.
  double Backward(const VectorXd& ground_truth,
                  const std::vector<VectorXd>& layer_inputs,
                  std::vector<VectorXd>& deltas) const;

  // Layers.
  std::vector<HiddenLayer::Ptr> hidden_layers_;
  OutputLayer::Ptr output_layer_;

  // Loss functor.
  const LossFunctor::ConstPtr loss_;
}; // class Network

} // namespace mininet

#endif

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
// Defines the BackpropTrainer class, which is derived from Trainer.
//
///////////////////////////////////////////////////////////////////////////////

#include <trainer/backprop_trainer.h>

#include <iostream>
#include <glog/logging.h>

namespace mininet {

BackpropTrainer::~BackpropTrainer() {}
BackpropTrainer::BackpropTrainer(const Network& network, const Dataset& dataset,
                                 const BackpropParams& params)
  : Trainer(network, dataset),
    params_(params) {}

// All trainers must implement this interface.
void BackpropTrainer::Train() {
  double learning_rate = params_.learning_rate_;
  double loss = std::numeric_limits<double>::infinity();

  for (size_t ii = 0; ii < params_.num_epochs_; ii++) {
    for (size_t jj = 0; jj < params_.num_iters_per_epoch_; jj++) {
      // Get a new batch.
      std::vector<VectorXd> input_samples, output_samples;
      if (!dataset_.Batch(params_.batch_size_, input_samples, output_samples))
        LOG(WARNING) << "Error while generating a batch.";

      // Compute average layer inputs and deltas.
      std::vector<MatrixXd> derivatives;
      loss = network_.RunBatch(input_samples, output_samples, derivatives);

      // Update weights.
      network_.UpdateWeights(derivatives, learning_rate);
    }

    // Print a message.
    std::printf("Epoch %zu: loss = %f\n", ii, loss);

    // Update learning rate.
    learning_rate *= params_.learning_rate_decay_;
  }
}

double BackpropTrainer::Test() const {
  return network_.Loss(dataset_.TestingInputs(), dataset_.TestingOutputs());
}

} // namespace mininet

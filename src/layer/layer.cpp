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
// Defines the Layer base class.
//
///////////////////////////////////////////////////////////////////////////////

#include <layer/layer.h>

#include <glog/logging.h>
#include <random>
#include <iostream>

namespace mininet {

// Initialize weights randomly, and add an extra input dimension for the
// input bias term.
Layer::~Layer() {}
Layer::Layer(size_t input_size, size_t output_size)
  : weights_(MatrixXd::Zero(output_size, input_size + 1)),
    weight_changes_(MatrixXd::Zero(output_size, input_size + 1)) {
  // Create a random number generator for a normal distribution of mean
  // 0.0 and standard deviation 0.1.
  std::random_device rd;
  std::default_random_engine rng(rd());
  std::normal_distribution<double> gaussian(0.0, 0.1);

  // Populate weights from this distribution.
  for (size_t ii = 0; ii < weights_.rows(); ii++)
    for (size_t jj = 0; jj < weights_.cols(); jj++)
      weights_(ii, jj) = gaussian(rng);
}

// Update weights by gradient descent.
// 'learning_rate' multiplies the derivative at each weight,
// 'momentum' multiplies the previous change in weight, and
// 'decay' multiplies the current value of the weight (L2 regularization).
void Layer::UpdateWeights(const MatrixXd& derivatives, double learning_rate,
                          double momentum, double decay) {
  CHECK_EQ(derivatives.rows(), weights_.rows());
  CHECK_EQ(derivatives.cols(), weights_.cols());

  weight_changes_ = -learning_rate * derivatives
    - decay * weights_ + momentum * weight_changes_;
  weights_ += weight_changes_;
}

// Perturb a single weight by a specified amount.
void Layer::PerturbWeight(size_t ii, size_t jj, double amount) {
  CHECK_LT(ii, weights_.rows());
  CHECK_LT(jj, weights_.cols());

  weights_(ii, jj) += amount;
}

} // namespace mininet

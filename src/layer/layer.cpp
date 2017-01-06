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

namespace mininet {

// Initialize weights randomly, and add an extra input dimension for the
// input bias term.
Layer::Layer(size_t input_size, size_t output_size)
  : weights_(MatrixXd::Random(output_size, input_size + 1)) {}
Layer::~Layer() {}

// Update weights by gradient descent.
void Layer::UpdateWeights(const VectorXd& inputs, const VectorXd& deltas,
                          double step_size) {
  CHECK(inputs.rows() == weights_.cols() - 1);
  CHECK(deltas.rows() == weights_.rows());

  for (size_t ii = 0; ii < weights_.rows(); ii++) {
    for (size_t jj = 0; jj < weights_.cols() - 1; jj++)
      weights_(ii, jj) -= step_size * inputs(jj) * deltas(ii);

    // Handle last column (bias).
    weights_(ii, weights_.cols() - 1) -= step_size * deltas(ii);
  }
}

// Perturb a single weight by a specified amount.
void Layer::PerturbWeight(size_t ii, size_t jj, double amount) {
  CHECK(ii < weights_.rows());
  CHECK(jj < weights_.cols());

  weights_(ii, jj) += amount;
}

} // namespace mininet
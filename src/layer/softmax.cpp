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
// Defines the softmax layer type.
//
///////////////////////////////////////////////////////////////////////////////

#include <layer/softmax.h>

#include <glog/logging.h>
#include <math.h>

namespace mininet {

// Factory method.
OutputLayer::Ptr Softmax::Create(size_t input_size, size_t output_size) {
  OutputLayer::Ptr ptr(new Softmax(input_size, output_size));
  return ptr;
}

// Private constructor. Use the factory method instead.
Softmax::Softmax(size_t input_size, size_t output_size)
  : OutputLayer(input_size, output_size) {}

// Activation and gradient. Implement these in derived classes.
void Softmax::Forward(const VectorXd& input, VectorXd& output) const {
  // Check that input and output are the correct sizes.
  CHECK(input.rows() == weights_.cols() - 1);
  CHECK(output.rows() == weights_.rows());

  // Compute linear transformation with bias.
  output = weights_.leftCols(input.rows()) * input + weights_.rightCols(1);

  // Compute non-linearity.
  double sum = 0.0;
  for (size_t ii = 0; ii < input.rows(); ii++) {
    output(ii) = std::exp(output(ii));
    sum += output(ii);
  }

  // Catch small sum.
  if (sum < 1e-16) {
    VLOG(1) << "Sum was too small in softmax layer. Did not normalize.";
  } else {
    output /= sum;
  }
}

double Softmax::Backward(const LossFunctor::ConstPtr& loss,
                         const VectorXd& ground_truth, const VectorXd& output,
                         VectorXd& gammas, VectorXd& deltas) const {
  CHECK_NOTNULL(loss.get());

  // Check that all dimensions line up.
  CHECK(output.rows() == weights_.rows());
  CHECK(gammas.rows() == weights_.cols() - 1);
  CHECK(deltas.rows() == weights_.rows());

  // Compute loss value and gradient with respect to 'output'.
  double loss_value = std::numeric_limits<double>::infinity();
  VectorXd loss_gradient(output.rows());
  CHECK(loss->Evaluate(ground_truth, output, loss_value, loss_gradient));

  // Use the chain rule to compute 'deltas'.
  for (size_t ii = 0; ii < deltas.rows(); ii++) {
    deltas(ii) = 0.0;

    for (size_t jj = 0; jj < loss_gradient.rows(); jj++) {
      if (ii == jj)
        deltas(ii) += loss_gradient(jj) * output(ii) * (1.0 - output(jj));
      else
        deltas(ii) -= loss_gradient(jj) * output(ii) * output(jj);
    }
  }

  // Compute the associated 'gammas'.
  for (size_t jj = 0; jj < gammas.rows(); jj++) {
    gammas(jj) = 0.0;

    for (size_t ii = 0; ii < deltas.rows(); ii++) {
      gammas(jj) += deltas(ii) * weights_(ii, jj);
    }
  }

  // Return loss.
  return loss_value;
}

} // namespace mininet

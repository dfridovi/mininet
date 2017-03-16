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
// Defines the Sigmoid layer type.
//
///////////////////////////////////////////////////////////////////////////////

#include <layer/sigmoid.h>

#include <glog/logging.h>
#include <math.h>

namespace mininet {

// Factory method.
Layer::Ptr Sigmoid::Create(size_t input_size, size_t output_size) {
  Layer::Ptr ptr(new Sigmoid(input_size, output_size));
  return ptr;
}

// Must implement a deep copy.
Layer::Ptr Sigmoid::Copy() const {
  Layer::Ptr ptr(new Sigmoid(*this));
  return ptr;
}

// Private constructor. Use the factory method instead.
Sigmoid::Sigmoid(size_t input_size, size_t output_size)
  : Layer(input_size, output_size) {}

// Activation and gradient. Implement these in derived classes.
void Sigmoid::Forward(const VectorXd& input, VectorXd& output) const {
  // Check that input and output are the correct sizes.
  CHECK(input.rows() == weights_.cols() - 1);
  CHECK(output.rows() == weights_.rows());

  // Compute linear transformation with bias.
  output = weights_.leftCols(input.rows()) * input + weights_.rightCols(1);

  // Apply nonlinearity.
  for (size_t ii = 0; ii < output.rows(); ii++)
    output(ii) = 1.0 / (1.0 + std::exp(-output(ii)));
}

void Sigmoid::Backward(const VectorXd& output, const VectorXd& upstream_gammas,
                    VectorXd& gammas, VectorXd& deltas) const {
  // Check that all dimensions line up.
  CHECK(upstream_gammas.rows() == weights_.rows());
  CHECK(output.rows() == weights_.rows());
  CHECK(deltas.rows() == weights_.rows());
  CHECK(gammas.rows() == weights_.cols() - 1);

  // Compute the 'deltas' from the 'upstream gammas'.
  for (size_t ii = 0; ii < deltas.rows(); ii++) {
    deltas(ii) = upstream_gammas(ii) * output(ii) * (1.0 - output(ii));
  }

  // Compute the associated 'gammas'.
  for (size_t jj = 0; jj < gammas.rows(); jj++) {
    gammas(jj) = 0.0;

    for (size_t ii = 0; ii < deltas.rows(); ii++) {
      gammas(jj) += deltas(ii) * weights_(ii, jj);
    }
  }
}

// Output layer version of backprop. This function computes the
// so-called 'deltas' and 'gammas', i.e. the derivative of loss with respect
// to the pre-nonlinearity values and layer inputs, respectively. Note that
// 'output' holds the output of the non-linearity.
double Sigmoid::Backward(const LossFunctor::ConstPtr& loss,
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
    deltas(ii) = loss_gradient(ii) * output(ii) * (1.0 - output(ii));
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

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
// Defines the ReLU layer type.
//
///////////////////////////////////////////////////////////////////////////////

#include <layer/relu.h>

#include <glog/logging.h>
#include <math.h>

namespace mininet {

// Activation and gradient. Implement these in derived classes.
void ReLU::Activation(const VectorXd& input, VectorXd& output) const {
  // Check that input and output are the correct sizes.
  CHECK(input.rows() == weights_.cols() - 1);
  CHECK(output.rows() == weights_.rows());

  // Compute linear transformation with bias.
  output = weights_.leftCols(input.rows()) * input + weights_.rightCols(1);

  // Apply hinge loss (ReLU) nonlinearity.
  for (size_t ii = 0; ii < output.rows(); ii++)
    output(ii) = max(output(ii), 0.0);
}

void ReLU::Gradient(const VectorXd& upstream_deltas,
                    const VectorXd& values,
                    MatrixXd& gradient, VectorXd& deltas) const {
  // Check that all dimensions line up.
  CHECK(upstream_deltas.rows() == weights_.rows());
  CHECK(values.rows() == weights_.cols() - 1);
  CHECK(deltas.rows() == weights_.cols());
  CHECK(gradient.rows() == weights_.rows());
  CHECK(gradient.cols() == weights_.cols());

  // Backpropagation.
  // TODO!
}

} // namespace mininet

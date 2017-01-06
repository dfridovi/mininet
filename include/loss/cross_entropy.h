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
 * Authors: David Fridovich-Keil   ( dfk@eecs.berkeley.edu )
 *          Sara Fridovich-Keil    ( saraf@princeton.edu )
 */

///////////////////////////////////////////////////////////////////////////////
//
// Defines the cross entropy loss functor.
//
// ////////////////////////////////////////////////////////////////////////////

#ifndef MININET_LOSS_CROSS_ENTROPY_H
#define MININET_LOSS_CROSS_ENTROPY_H

#include <loss/loss_functor.h>

#include <glog/logging.h>
#include <math.h>

namespace mininet {

struct CrossEntropy : public LossFunctor {
  // Factory method.
  static LossFunctor::Ptr Create() {
    LossFunctor::Ptr ptr(new CrossEntropy);
    return ptr;
  }

  // All loss functors must evaluate the loss and derivative with respect to
  // the input 'values' (which are the output of some 'OutputLayer').
  bool Evaluate(const VectorXd& ground_truth, const VectorXd& values,
                double& loss, VectorXd& gradient) const {
    // Check that 'ground truth' and 'values' are probability distributions
    // on equal-sized alphabets.
    if (ground_truth.rows() != values.rows()) {
      VLOG(1) << "Ground truth and values are not the same length.";
      return false;
    }

    if (ground_truth.sum() < 1.0 - 1e-16 || ground_truth.sum() > 1.0 + 1e-16) {
      VLOG(1) << "Ground truth vector does not sum to unity.";
      return false;
    }

    if (values.sum() < 1.0 - 1e-16 || values.sum() > 1.0 + 1e-16) {
      VLOG(1) << "Values vector does not sum to unity.";
      return false;
    }

    if (ground_truth.minCoeff() < 0.0) {
      VLOG(1) << "Ground truth vector contains a number less than 0.0.";
      return false;
    }

    if (ground_truth.maxCoeff() > 1.0) {
      VLOG(1) << "Ground truth vector contains a number greater than 1.0.";
      return false;
    }

    if (values.minCoeff() < 0.0) {
      VLOG(1) << "Values vector contains a number less than 0.0.";
      return false;
    }

    if (values.maxCoeff() > 1.0) {
      VLOG(1) << "Values vector contains a number greater than 1.0.";
      return false;
    }

    // Compute the loss and gradient.
    loss = 0.0;
    for (size_t ii = 0; ii < ground_truth.rows(); ii++) {
      loss -= ground_truth(ii) * std::log(values(ii));
      gradient(ii) = -ground_truth(ii) / values(ii);
    }

    return true;
  }
}; //\struct CrossEntropy

}  //\namespace mininet

#endif

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
 * Authors: David Fridovich-Keil   ( dfk@eecs.berkeley.edu )
 */

///////////////////////////////////////////////////////////////////////////////
//
// Unit tests for loss functors.
//
///////////////////////////////////////////////////////////////////////////////

#include <loss/loss_functor.h>
#include <loss/cross_entropy.h>
#include <loss/l2.h>

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <iostream>
#include <math.h>

using namespace mininet;

// Test that the gradient computation agrees with numerical differentiation.
TEST(CrossEntropy, TestGradient) {
  const size_t kNumLabels = 100;
  const double kPerturbation = 1e-8;
  const double kInversePerturbation = 1e8;
  const double kEpsilon = 1e-6;

  const LossFunctor::ConstPtr loss = CrossEntropy::Create();

  // Generate a random input/output pair.
  VectorXd input = VectorXd::Random(kNumLabels);
  for (size_t ii = 0; ii < kNumLabels; ii++)
    input(ii) -= input.minCoeff();
  input /= input.sum();

  VectorXd output = VectorXd::Zero(kNumLabels);

  std::random_device rd;
  std::default_random_engine rng(rd());
  std::uniform_int_distribution<size_t> label(0, kNumLabels - 1);
  output(label(rng)) = 1.0;

  // Compute loss and gradient.
  double loss_value;
  VectorXd loss_gradient(kNumLabels);
  ASSERT_TRUE(loss->Evaluate(output, input, loss_value, loss_gradient));

  // Check against numerical derivatives.
  for (size_t ii = 0; ii < kNumLabels; ii++) {
    const double original_input = input(ii);

    double forward_loss, backward_loss;
    VectorXd dummy_gradient(kNumLabels);

    input(ii) += kPerturbation;
    ASSERT_TRUE(loss->Evaluate(output, input, forward_loss, dummy_gradient));

    input(ii) -= 2.0 * kPerturbation;
    ASSERT_TRUE(loss->Evaluate(output, input, backward_loss, dummy_gradient));

    const double numerical_derivative =
      0.5 * kInversePerturbation * (forward_loss - backward_loss);
    //input(ii) = original_input;
    input(ii) += kPerturbation;

    // Make sure they are close.
    EXPECT_NEAR(numerical_derivative, loss_gradient(ii), kEpsilon);
  }
}

// Test that the gradient computation agrees with numerical differentiation.
TEST(L2, TestGradient) {
  const size_t kNumLabels = 100;
  const double kPerturbation = 1e-8;
  const double kInversePerturbation = 1e8;
  const double kEpsilon = 1e-6;

  const LossFunctor::ConstPtr loss = L2::Create();

  // Generate a random input/output pair.
  VectorXd input = VectorXd::Random(kNumLabels);
  for (size_t ii = 0; ii < kNumLabels; ii++)
    input(ii) -= input.minCoeff();
  input /= input.sum();

  VectorXd output = VectorXd::Zero(kNumLabels);

  std::random_device rd;
  std::default_random_engine rng(rd());
  std::uniform_int_distribution<size_t> label(0, kNumLabels - 1);
  output(label(rng)) = 1.0;

  // Compute loss and gradient.
  double loss_value;
  VectorXd loss_gradient(kNumLabels);
  ASSERT_TRUE(loss->Evaluate(output, input, loss_value, loss_gradient));

  // Check against numerical derivatives.
  for (size_t ii = 0; ii < kNumLabels; ii++) {
    const double original_input = input(ii);

    double forward_loss, backward_loss;
    VectorXd dummy_gradient(kNumLabels);

    input(ii) += kPerturbation;
    ASSERT_TRUE(loss->Evaluate(output, input, forward_loss, dummy_gradient));

    input(ii) -= 2.0 * kPerturbation;
    ASSERT_TRUE(loss->Evaluate(output, input, backward_loss, dummy_gradient));

    const double numerical_derivative =
      0.5 * kInversePerturbation * (forward_loss - backward_loss);
    //input(ii) = original_input;
    input(ii) += kPerturbation;

    // Make sure they are close.
    EXPECT_NEAR(numerical_derivative, loss_gradient(ii), kEpsilon);
  }
}

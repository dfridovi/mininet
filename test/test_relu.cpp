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
// Unit tests for the ReLU layer type. The main idea here is to test whether a
// network composed of only ReLU layers can compute derivatives properly.
//
///////////////////////////////////////////////////////////////////////////////

#include <layer/relu.h>
#include <layer/layer_params.h>
#include <loss/cross_entropy.h>
#include <net/network.h>

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <iostream>
#include <math.h>

using namespace mininet;

// Single layer test.
TEST(ReLU, TestSingleLayer) {
  const size_t kNumInputs1 = 100;
  const size_t kNumInputs2 = 100;
  const size_t kNumOutputs = 10;
  const size_t kNumChecks = 100;
  const double kPerturbation = 1e-8;
  const double kInversePerturbation = 1e8;
  const double kEpsilon = 1e-6;

  // Create a network.
  std::vector<LayerParams> params;
  params.push_back(LayerParams(RELU, kNumInputs1, kNumInputs2));
  params.push_back(LayerParams(SOFTMAX, kNumInputs2, kNumOutputs));;

  const LossFunctor::ConstPtr loss = CrossEntropy::Create();
  Network net(params, loss);

  // Generate a random input/output pair.
  const VectorXd input = VectorXd::Random(kNumInputs1);
  VectorXd output = VectorXd::Zero(kNumOutputs);

  std::random_device rd;
  std::default_random_engine rng(rd());
  std::uniform_int_distribution<size_t> label(0, kNumOutputs - 1);
  output(label(rng)) = 1.0;

  // Run this input/output pair through the network.
  std::vector<VectorXd> batch = { input };
  std::vector<VectorXd> ground_truth = { output };
  std::vector<MatrixXd> derivatives;

  const double empirical_loss =
    net.RunBatch(batch, ground_truth, derivatives);

  ASSERT_TRUE(derivatives.size() == 2);

  // For a bunch of random weights, compute the derivative two ways:
  // (1) with the 'derivatives' computed above, and
  // (2) with numerical differentiation.

  // Output layer check.
  std::uniform_int_distribution<size_t> output_jj(0, kNumInputs2);
  std::uniform_int_distribution<size_t> output_ii(0, kNumOutputs - 1);
  for (size_t kk = 0; kk < kNumChecks; kk++) {
    const size_t jj = output_jj(rng);
    const size_t ii = output_ii(rng);

    // Compute derivative with a symmetric difference.
    net.PerturbWeight(1, ii, jj, kPerturbation);
    const double forward_loss = net.Loss(batch, ground_truth);

    net.PerturbWeight(1, ii, jj, -2.0 * kPerturbation);
    const double backward_loss = net.Loss(batch, ground_truth);

    const double numerical_derivative =
      0.5 * kInversePerturbation * (forward_loss - backward_loss);
    net.PerturbWeight(1, ii, jj, kPerturbation);

    // Compute derivative using backprop.
    const double backprop_derivative = derivatives[1](ii, jj);

    // Make sure they are close.
    EXPECT_NEAR(numerical_derivative, backprop_derivative, kEpsilon);
  }

  // Hidden layer check.
  std::uniform_int_distribution<size_t> hidden_jj(0, kNumInputs1);
  std::uniform_int_distribution<size_t> hidden_ii(0, kNumInputs2 - 1);
  for (size_t kk = 0; kk < kNumChecks; kk++) {
    const size_t jj = hidden_jj(rng);
    const size_t ii = hidden_ii(rng);

    // Compute derivative with a symmetric difference.
    net.PerturbWeight(0, ii, jj, kPerturbation);
    const double forward_loss = net.Loss(batch, ground_truth);

    net.PerturbWeight(0, ii, jj, -2.0 * kPerturbation);
    const double backward_loss = net.Loss(batch, ground_truth);

    const double numerical_derivative =
      0.5 * kInversePerturbation * (forward_loss - backward_loss);
    net.PerturbWeight(0, ii, jj, kPerturbation);

    // Compute derivative using backprop.
    const double backprop_derivative = derivatives[0](ii, jj);

    // Filter out cases where either derivative is exactly 0.0.
    if (numerical_derivative == 0.0 || backprop_derivative == 0.0)
      continue;

    // Make sure they are close.
    EXPECT_NEAR(numerical_derivative, backprop_derivative, kEpsilon);
  }
}

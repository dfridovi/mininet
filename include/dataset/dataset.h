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
// Defines the dataset class.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef MININET_DATASET_DATASET_H
#define MININET_DATASET_DATASET_H

#include <util/types.h>

#include <vector>
#include <random>
#include <algorithm>

namespace mininet {

class Dataset {
public:
  explicit Dataset(const std::vector<VectorXd>& inputs,
                   const std::vector<VectorXd>& outputs,
                   double training_fraction = 0.75);
  ~Dataset();

  // Get a random sample from the training set. Returns false if there are not
  // enough training samples.
  bool Batch(size_t batch_size, std::vector<VectorXd>& input_samples,
             std::vector<VectorXd>& output_samples);

  // Get a const reference to the testing set.
  const std::vector<VectorXd>& TestingInputs() const;
  const std::vector<VectorXd>& TestingOutputs() const;

private:
  // Training and testing sets.
  std::vector<VectorXd> training_inputs_;
  std::vector<VectorXd> training_outputs_;
  std::vector<VectorXd> testing_inputs_;
  std::vector<VectorXd> testing_outputs_;
}; // class Dataset

} // namespace mininet

#endif

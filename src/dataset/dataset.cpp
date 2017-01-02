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
// Defines the dataset class.
//
///////////////////////////////////////////////////////////////////////////////

#include <dataset/dataset.h>

#include <glog/logging.h>

namespace mininet {

Dataset::~Dataset() {}
Dataset::Dataset(const std::vector<VectorXd>& inputs,
                 const std::vector<VectorXd>& outputs,
                 double training_fraction) {
  CHECK(inputs.size() == outputs.size());
  CHECK(training_fraction >= 0.0 && training_fraction <= 1.0);

  // Random number generation.
  std::random_device rd;
  std::default_random_engine rng(rd());

  // Randomly permute data indices.
  std::vector<size_t> indices(inputs.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), rng);

  // Parcel out the data into training and testing sets.
  const size_t kNumTrainingSamples =
    static_cast<size_t>(training_fraction * inputs.size());

  for (size_t ii = 0; ii < kNumTrainingSamples; ii++) {
    training_inputs_.push_back(inputs[ indices[ii] ]);
    training_outputs_.push_back(outputs[ indices[ii] ]);
  }

  for (size_t ii = kNumTrainingSamples; ii < inputs.size(); ii++) {
    testing_inputs_.push_back(inputs[ indices[ii] ]);
    testing_outputs_.push_back(outputs[ indices[ii] ]);
  }
}

// Get a random sample from the training set. Returns false if there are not
// enough training samples.
bool Dataset::Batch(size_t batch_size, std::vector<VectorXd>& input_samples,
                    std::vector<VectorXd>& output_samples) {
  input_samples.clear();
  output_samples.clear();

  // Check that batch size is smaller than training set.
  if (batch_size >= training_inputs_.size()) {
    VLOG(1) << "Batch size is larger than training set. Returning entire"
            << " training set.";
    batch_size = training_inputs_.size();
  }

  // Random number generation.
  std::random_device rd;
  std::default_random_engine rng(rd());

  // Generate a random subset of the training data.
  std::vector<size_t> indices(training_inputs_.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), rng);

  for (size_t ii = 0; ii < batch_size; ii++) {
    input_samples.push_back(training_inputs_[ indices[ii] ]);
    output_samples.push_back(training_outputs_[ indices[ii] ]);
  }

  return input_samples.size() < training_inputs_.size();
}

// Get a const reference to the testing set.
const std::vector<VectorXd>& Dataset::TestingInputs() const {
  return testing_inputs_;
}

const std::vector<VectorXd>& Dataset::TestingOutputs() const {
  return testing_outputs_;
}

} // namespace mininet

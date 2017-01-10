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

#include <dataset/dataset.h>

#include <glog/logging.h>
#include <unordered_set>

namespace mininet {

Dataset::~Dataset() {}
Dataset::Dataset(const std::vector<VectorXd>& training_inputs,
                 const std::vector<VectorXd>& training_outputs,
                 const std::vector<VectorXd>& testing_inputs,
                 const std::vector<VectorXd>& testing_outputs,
                 bool normalize)
  : training_inputs_(training_inputs),
    training_outputs_(training_outputs),
    testing_inputs_(testing_inputs),
    testing_outputs_(testing_outputs) {
  CHECK_EQ(training_inputs_.size(), training_outputs_.size());
  CHECK_EQ(testing_inputs_.size(),  testing_outputs_.size());
  CHECK(training_inputs_.size() > 0 && testing_inputs_.size() > 0);

  if (normalize) {
    Normalize(training_inputs_);
    Normalize(testing_inputs_);
  }
}

Dataset::Dataset(const std::vector<VectorXd>& inputs,
                 const std::vector<VectorXd>& outputs,
                 double training_fraction, bool normalize) {
  CHECK(inputs.size() == outputs.size());
  CHECK(training_fraction >= 0.0 && training_fraction <= 1.0);
  CHECK(inputs.size() > 0 && training_fraction * inputs.size() > 0.5);

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

  if (normalize) {
    Normalize(training_inputs_);
    Normalize(testing_inputs_);
  }
}

// Normalize a set of vectors, so that across all vectors each entry is zero
// mean and variance one.
void Dataset::Normalize(std::vector<VectorXd>& data) {
  if (data.size() <= 1) {
    LOG(WARNING) << "Tried to normalize a dataset of size 0 or 1.";
    return;
  }

  // Extract data dimension.
  const size_t dimension = data[0].rows();

  for (size_t ii = 0; ii < dimension; ii++) {
    double sum = 0.0;
    double sum_squares = 0.0;

    for (const auto& x : data) {
      // Assume all data has the correct dimension for speed.
      sum += x(ii);
      sum_squares += x(ii) * x(ii);
    }

    const double mean = sum / data.size();
    const double second_moment = sum_squares / data.size();
    const double stddev = std::sqrt(second_moment - mean * mean);

    if (stddev < 1e-8) {
      LOG(WARNING) << "Encountered a dimension of all zeros.";
      continue;
    }

    const double inv_stddev = 1.0 / stddev;

    // Loop back through the data and normalize this dimension.
    for (auto& x : data)
      x(ii) = (x(ii) - mean) * inv_stddev;
  }

}

// Get a random sample from the training set. Returns false if there are not
// enough training samples.
bool Dataset::Batch(size_t batch_size, std::vector<VectorXd>& input_samples,
                    std::vector<VectorXd>& output_samples) {
  input_samples.clear();
  output_samples.clear();

  size_t thresholded_batch_size = batch_size;

  // Check that batch size is smaller than training set.
  if (thresholded_batch_size > training_inputs_.size()) {
    VLOG(1) << "Batch size is larger than training set. Returning entire"
            << " training set.";
    thresholded_batch_size = training_inputs_.size();
  }

  // Random number generation.
  std::random_device rd;
  std::default_random_engine rng(rd());

  // Generate a random subset of the training data one of two ways:
  // (1) if batch size / training size >= 1 - 1/e, randomly shuffle,
  // (2) otherwise, draw random indices and check if they've been drawn.
  if (static_cast<double>(thresholded_batch_size) / training_inputs_.size() >=
      1.0 - 1.0 / M_E) {
    // (1) If batch size is large, do a random shuffle.
    std::vector<size_t> indices(training_inputs_.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    for (size_t ii = 0; ii < thresholded_batch_size; ii++) {
      input_samples.push_back(training_inputs_[ indices[ii] ]);
      output_samples.push_back(training_outputs_[ indices[ii] ]);
    }
  } else {
    // (2) Batch size is small, so choose random indices.
    std::uniform_int_distribution<size_t> unif(0, training_inputs_.size() - 1);
    std::unordered_set<size_t> sampled_indices;

    while (input_samples.size() < thresholded_batch_size) {
      // Pick a random index in the training set that we have not seen yet.
      const size_t ii = unif(rng);
      if (sampled_indices.count(ii) > 0)
        continue;

      sampled_indices.insert(ii);

      // Insert the corresponding samples.
      input_samples.push_back(training_inputs_[ii]);
      output_samples.push_back(training_outputs_[ii]);
    }
  }

  return input_samples.size() < batch_size;
}

// Get a const reference to the testing set.
const std::vector<VectorXd>& Dataset::TestingInputs() const {
  return testing_inputs_;
}

const std::vector<VectorXd>& Dataset::TestingOutputs() const {
  return testing_outputs_;
}

} // namespace mininet

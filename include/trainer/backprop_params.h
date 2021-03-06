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
 *          Sara Fridovich-Keil    ( saraf@princeton.edu )
 */

///////////////////////////////////////////////////////////////////////////////
//
// Defines the backprop parameters base struct.
//
// ////////////////////////////////////////////////////////////////////////////

#ifndef MININET_TRAINER_BACKPROP_PARAMS_H
#define MININET_TRAINER_BACKPROP_PARAMS_H

#include "../util/types.h"

namespace mininet {

struct BackpropParams {
  // Maximum allowable average loss (on the validation set) for a batch.
  // Used as a stopping criterion.
  double max_avg_loss_ = 1e-4;

  // Batch size. Amount of data to be used in each SGD iteration.
  size_t batch_size_ = 20;

  // Number of iterations per epoch.
  size_t num_iters_per_epoch_ = 10;

  // Number of epochs.
  size_t num_epochs_ = 10;

  // Initial learning rate.
  double learning_rate_ = 0.1;

  // Learning rate decay factor (multiply by this each new epoch).
  double learning_rate_decay_ = 0.5;

  // Momentum multiplier.
  double momentum_ = 0.0;

  // Weight decay rate (L2 regularizer).
  double weight_decay_ = 0.0;
}; //\struct BackpropParameters

}  //\namespace mininet

#endif

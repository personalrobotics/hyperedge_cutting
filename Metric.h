/***********************************************************************

Copyright (c) 2014, Carnegie Mellon University
Author: Shervin Javdani <sjavdani@gmail.com>

Near Optimal Bayesian Active Learning for Decision Making
Shervin Javdani, Yuxin Chen, Amin Karbasi, Andreas Krause, J. Andrew (Drew) Bagnell, and Siddhartha Srinivasa
Proceedings of the 17th International Conference on Artificial Intelligence and Statistics (AISTATS), April, 2014.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

  Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*************************************************************************/


#ifndef METRIC_H
#define METRIC_H


#include <vector>
#include "MetricUtils.h"


class Metric
{
  public:
    Metric();
    Metric(std::vector<RegionList>& particle_regs);
    ~Metric();

    virtual void updateSubregionMap(std::vector<RegionList>& particle_regs);
    virtual void updateSubregionWeights(std::vector<double>& particle_weights);

    double computeExpectedValue_oneTest(std::vector<double>& particle_weights, std::vector<int>& observation_matrix);
    double computeExpectedValue_oneTest(const std::vector<double>& particle_weights, const std::vector< std::vector<size_t> >& outcome_matrix);
    void computeExpectedValues(std::vector<double>& particle_weights, std::vector< std::vector<int > >& observation_matrix, std::vector<double>& expected_values);

    virtual double computeCurrentValue(){return 0;};

    bool isDone(std::vector<double>& weights);
    int workingRegion(std::vector<double>& weights);

  protected:
    std::vector<size_t> _particle_to_subreg_map;
    std::vector<SubRegion> _subregions;

    int indexFromSubRegionPointer(const SubRegion* ptr);


};

#endif

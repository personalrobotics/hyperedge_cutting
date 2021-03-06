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


#ifndef METRIC_HYPEREDGE_FIXEDCARD_H
#define METRIC_HYPEREDGE_FIXEDCARD_H


#include <vector>
#include "Metric.h"
#include "MetricUtils.h"


class Metric_Hyperedge_Fixedcard: public Metric
{
  public:
    Metric_Hyperedge_Fixedcard();
    Metric_Hyperedge_Fixedcard(std::vector<RegionList>& particle_regs);
    ~Metric_Hyperedge_Fixedcard();

    void updateSubregionMap(std::vector<RegionList>& particle_regs);
    void precomputeRemovalValueTreeNodes();

    double sumSubRegionWeights(std::vector<SubRegionList>& permutations);

    double computeCurrentValue();
    double computeCurrentValueRemoval(SubRegionTreeNode* node, double weight_so_far, std::vector<size_t>& inds_so_far);

    double computeCurrentValue_bruteForce();

  private:
    int _edge_cardinality;
    SubRegionTreeNode* _removalTreeRoot;

    void findEdgeCardinality();

    //for complete homogeneous symmetric poly calculations
    double computeLastCompleteHomogeneousPolynomials(const std::vector<size_t>& subregion_indices, int order);
    void computeAllCompleteHomogeneousPolynomials(const std::vector<size_t>& subregion_indices, int order, std::vector<double>& compl_homo_polys);
    void computeAllElementarySymmetric(const std::vector<size_t>& subregion_indices, int order, std::vector<double>& elem_symm_polys);
    void computeAllPowersums(const std::vector<size_t>& subregion_indices, int order, std::vector<double>& power_sums);

    void generateAllPermutations(std::vector<SubRegionList >& allPermutations);
    void removeSharedRegionPermutations(std::vector<SubRegionList >& allPermutations, std::vector<SubRegionList>& prunedPermutations);


};

#endif

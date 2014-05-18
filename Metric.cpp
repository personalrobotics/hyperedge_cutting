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


#include "Metric.h"
#include "MetricUtils.h"
#include <numeric>


Metric::Metric()
{
}

Metric::Metric(std::vector<RegionList>& particle_regs)
{
  updateSubregionMap(particle_regs);
}


Metric::~Metric()
{

}

/****************************************************
 * Update the subregion map which includes:
 * 1. Construct the list of subregions
 * 2. A mapping from particle index to subregion index
****************************************************/
void Metric::updateSubregionMap(std::vector<RegionList>& particle_regs)
{
  _particle_to_subreg_map.resize(particle_regs.size());
  _subregions.clear();

  for(size_t i=0; i < particle_regs.size(); i++)
  {
    size_t subreg_ind = _subregions.size()+1;
    for (size_t j=0; j < _subregions.size(); j++)
    {
      if (particle_regs[i].isEquiv(_subregions[j]._regions))
      {
        subreg_ind = j;
        break;
      }
    }
    if (subreg_ind < _subregions.size())
    {
      _particle_to_subreg_map[i] = subreg_ind;
    } else {
      _particle_to_subreg_map[i] = _subregions.size();
      SubRegion newSubReg;
      newSubReg._regions = particle_regs[i];
      _subregions.push_back(newSubReg);
    }

  }

  
  /*
  for(size_t i=0; i < _particle_to_subreg_map.size(); i++)
  {
    std::cout << _particle_to_subreg_map[i] << std::endl;
  }
  for(size_t i=0; i < _subregions.size(); i++)
  {
    std::cout << _subregions[i]._regions << std::endl;
  }
  */
}


void Metric::updateSubregionWeights(std::vector<double>& particle_weights)
{
  if (particle_weights.size() != _particle_to_subreg_map.size())
  {
    std::cerr << "ERROR: weight list not the same size as particle map. Perhaps update map?" << std::endl;
    std::cerr << "particle size: " << particle_weights.size() << "\t\t subreg map size: " << _particle_to_subreg_map.size() << std::endl;
  }

  //set all weights to 0
  for (size_t i=0; i < _subregions.size(); i++)
  {
    _subregions[i]._weight = 0.0;
  }

  for (size_t i=0; i < _particle_to_subreg_map.size(); i++)
  {
    _subregions[_particle_to_subreg_map[i]]._weight += particle_weights[i];
  }
}



double Metric::computeExpectedValue_oneTest(std::vector<double>& particle_weights, std::vector<int>& observation_matrix)
{
  std::vector< std::vector<size_t> >outcome_matrix; //matrix where the i'th row contains the index of all particles that agree with observation i
  for (size_t particle_index = 0; particle_index < observation_matrix.size(); particle_index++)
  {
    int observation_ind = observation_matrix[particle_index];
    if (observation_ind >= outcome_matrix.size())
    {
      outcome_matrix.resize(observation_ind+1);
    }
    outcome_matrix[observation_ind].push_back(particle_index);
  }

  return computeExpectedValue_oneTest(particle_weights, outcome_matrix);
}

/***********************************************************
 * Computes the expected value of one test, given the current set of weights
 * and the outcome matrix
 * outcome matrix: for row [i], the vector contains all indices of particles
 * that agree with that observation
***********************************************************/
double Metric::computeExpectedValue_oneTest(const std::vector<double>& particle_weights, const std::vector< std::vector<size_t> >& outcome_matrix)
{
  double total_weight_particles = 0.;

  //go through each outcome
  double expected_val_test = 0.0;
  for (std::vector<std::vector<size_t> >::const_iterator it = outcome_matrix.begin(); it < outcome_matrix.end(); ++it)
  {
    std::vector<double> weights_after_obs(particle_weights.size(), 0.0);
    double weight_thisobs = 0.0;
    for (std::vector<size_t>::const_iterator it_particles = (*it).begin(); it_particles < (*it).end(); ++it_particles)
    {
      double weight_thispart = particle_weights[*it_particles];
      weights_after_obs[*it_particles] = weight_thispart;
      weight_thisobs += weight_thispart;
    }

    //compute subregion hyperedge weights
    updateSubregionWeights(weights_after_obs);
    double val_HE = computeCurrentValue();

    expected_val_test += val_HE*weight_thisobs;
    total_weight_particles += weight_thisobs;
  }

  if (total_weight_particles == 0.)
  {
      std::cerr << "total weight of all observations is zero?" << std::endl;
      return 0.;
  }

  return (expected_val_test / total_weight_particles);
}

void Metric::computeExpectedValues(std::vector<double>& particle_weights, std::vector< std::vector<int > >& observation_matrix, std::vector<double>& expected_values)
{
  expected_values.resize(observation_matrix.size());
  for (size_t i = 0; i < observation_matrix.size(); i++)
  {
    expected_values[i] = computeExpectedValue_oneTest(particle_weights, observation_matrix[i]);
  }

}


bool Metric::isDone(std::vector<double>& weights)
{
    updateSubregionWeights(weights);
    SubRegionList remaining_subregions;
    for (size_t i=0; i < _subregions.size(); i++)
    {
        if (_subregions[i].getWeight() > 0.)
        {
            remaining_subregions.addSubregion(_subregions[i]);
            //std::cout << "remaining possible regs: " << remaining_subregions._region_intersections << std::endl;
        }
    }

    return (remaining_subregions.numIntersections() > 0);
}

int Metric::workingRegion(std::vector<double>& weights)
{
    updateSubregionWeights(weights);
    SubRegionList remaining_subregions;
    for (size_t i=0; i < _subregions.size(); i++)
    {
        if (_subregions[i].getWeight() > 0.)
        {
            remaining_subregions.addSubregion(_subregions[i]);
            //std::cout << "remaining possible regs: " << remaining_subregions._region_intersections << std::endl;
        }
    }

    if (remaining_subregions.numIntersections() <= 0)
    {
        return -1;
    } else {
        return remaining_subregions._region_intersections._regions[0];
    }
}

int Metric::indexFromSubRegionPointer(const SubRegion* ptr)
{
  if (_subregions.size() == 0)
    return 0;

  return ptr -&_subregions[0];
}

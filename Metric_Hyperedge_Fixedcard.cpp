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


#include <algorithm>
#include <vector>
#include <queue>
#include <cctype>
#include <math.h>
#include "Metric_Hyperedge_Fixedcard.h"


Metric_Hyperedge_Fixedcard::Metric_Hyperedge_Fixedcard()
  : Metric(), _removalTreeRoot(NULL)
{
}

//is there some way to have parent call the child's updateSubregionMap?
Metric_Hyperedge_Fixedcard::Metric_Hyperedge_Fixedcard(std::vector<RegionList>& particle_regs)
  : Metric(particle_regs), _removalTreeRoot(NULL)
{
  updateSubregionMap(particle_regs);
}



Metric_Hyperedge_Fixedcard::~Metric_Hyperedge_Fixedcard()
{
  delete _removalTreeRoot;
}


void Metric_Hyperedge_Fixedcard::updateSubregionMap( std::vector<RegionList>& particle_regs)
{
  Metric::updateSubregionMap(particle_regs);
  findEdgeCardinality();
  precomputeRemovalValueTreeNodes();
}

void Metric_Hyperedge_Fixedcard::findEdgeCardinality()
{
  _edge_cardinality = 0;
  for (size_t i=0; i < _subregions.size(); i++)
  {
    _edge_cardinality = std::max(_edge_cardinality, _subregions[i].numRegions());
  }

  _edge_cardinality += 1;

}



double Metric_Hyperedge_Fixedcard::computeCurrentValue()
{
  std::vector<size_t> subregion_indices(_subregions.size());
  for (size_t i=0; i < _subregions.size(); i++)
  {
    subregion_indices[i] = i;
  }
  double eff_val = computeLastCompleteHomogeneousPolynomials(subregion_indices, _edge_cardinality);


  std::vector<size_t> inds_for_removal;
  inds_for_removal.reserve(_edge_cardinality);
  eff_val -= computeCurrentValueRemoval(_removalTreeRoot, 1.0, inds_for_removal);
  
  //double brute_val = computeCurrentValue_bruteForce();
  //std::cout << "value: " << eff_val << "   brute force: " << brute_val << std::endl;
  
  return eff_val;
  
}


/****************************************************
  * Precomputes tree nodes to be used during calculation
****************************************************/
void Metric_Hyperedge_Fixedcard::precomputeRemovalValueTreeNodes()
{
  //create the root node which is empty, and connected to each single region
  delete _removalTreeRoot;
  
  _removalTreeRoot = new SubRegionTreeNode();
  _removalTreeRoot->_prev = NULL;

  //std::cout << "num sub regions: " << _subregions.size() << std::endl;

  std::queue<SubRegionTreeNode*> check_for_removal;
  std::queue<SubRegionTreeNode*> check_for_removal_next;
  //add individual element, remove n_i^\alpha from sum
  for (int i = 0; i < _subregions.size(); ++i)
  {
    SubRegionTreeNode* nxt = new SubRegionTreeNode();
    nxt->_prev = _removalTreeRoot;
    nxt->_subregion = i;

    _removalTreeRoot->_nexts.push_back(nxt);

    check_for_removal.push(nxt);
  }

  int curr_order = 1;

  while (!check_for_removal.empty() && curr_order < _edge_cardinality )
  {
    curr_order += 1;

    SubRegionTreeNode* prev_last = NULL;
    std::vector<SubRegionTreeNode*>::iterator it_start;
    while(!check_for_removal.empty())
    {
      SubRegionTreeNode* tocheck = check_for_removal.front();
      check_for_removal.pop();

      if (tocheck->_prev != prev_last)
      {
        prev_last = tocheck->_prev;
        it_start = tocheck->_prev->_nexts.begin();
        it_start++;
      }

      //compute region intersections up until this point
      std::vector<size_t> curr_elems;
      tocheck->getAllSubregionIndsUntil(curr_elems, _removalTreeRoot);
      RegionList region_intersections_sofar = _subregions[tocheck->_subregion]._regions;
      for (std::vector<size_t>::iterator it = curr_elems.begin(); it != curr_elems.end(); ++it)
      {
        region_intersections_sofar.intersectRegions(_subregions[*it]._regions, region_intersections_sofar);
      }
      
      //try adding each next node
      //eligible regions are everything prev points to
      for (std::vector<SubRegionTreeNode*>::iterator it = it_start; it != tocheck->_prev->_nexts.end(); ++it)
      {
        //make sure lists are in increasing order of index
        if ( (*it)->_subregion <= tocheck->_subregion)
        {
          it_start++;
          continue;
        }

        //add if this new region has intersection with previous
        RegionList region_intersections_with_next;
        region_intersections_sofar.intersectRegions( _subregions[(*it)->_subregion]._regions, region_intersections_with_next);
          
        if (region_intersections_with_next.numRegions() > 0)
        {
          SubRegionTreeNode* next = new SubRegionTreeNode(tocheck, (*it)->_subregion);
          tocheck->_nexts.push_back(next);
          check_for_removal_next.push(next);
          
          //curr_elems.back() = (*it)->_subregion;
          //sum_to_remove += (next->_weight_so_far*computeLastCompleteHomogeneousPolynomials(curr_elems, _edge_cardinality-curr_order));
        }
      }
      // Shrink vector to minimum required size
      
      //std::cout << "capacity before: " << tocheck->_nexts.capacity() << std::endl;
      std::vector<SubRegionTreeNode*>(tocheck->_nexts.begin(), tocheck->_nexts.end()).swap(tocheck->_nexts);
      //std::cout << "capacity after " << tocheck->_nexts.capacity() << std::endl;

    }

    std::swap(check_for_removal, check_for_removal_next);
  }

}


/****************************************************
 * Computes how much we need to remove from the last complete homogeneous symmetric polynomial
 * Returns the sum of all elements which do share a region
 * Uses precomputed tree nodes
****************************************************/
double Metric_Hyperedge_Fixedcard::computeCurrentValueRemoval(SubRegionTreeNode* node, double weight_so_far, std::vector<size_t>& inds_so_far)
{
  //std::cout << node->_subregion << std::endl;
  double sum;
  if (node->_prev != NULL)
  {
    weight_so_far *= _subregions[node->_subregion]._weight;
    if (weight_so_far <= 1e-15)
      return 0.0;

    inds_so_far.push_back(node->_subregion);
    sum = weight_so_far*computeLastCompleteHomogeneousPolynomials(inds_so_far, _edge_cardinality - inds_so_far.size());
  } else {
    sum = 0.0;
  }

  for (std::vector<SubRegionTreeNode*>::iterator it = node->_nexts.begin(); it != node->_nexts.end(); ++it)
  {
    sum += computeCurrentValueRemoval(*it, weight_so_far, inds_so_far);
  }

  inds_so_far.pop_back();

  return sum;
}



/****************************************************
 * Computes all complete homogeneous symmetric polynomials up to specified order
 * eg compl_homo_polys[3] = \sum n_i n_j n_k   n_i doesn't need to be unique
****************************************************/
double Metric_Hyperedge_Fixedcard::computeLastCompleteHomogeneousPolynomials(const std::vector<size_t>& subregion_indices, int order)
{
  if (order == 0)
    return 1.0;
  std::vector<double> compl_homo_polys;
  computeAllCompleteHomogeneousPolynomials(subregion_indices, order, compl_homo_polys);
  
  return compl_homo_polys.back();
}

/****************************************************
 * Computes all complete homogeneous symmetric polynomials up to specified order
 * eg compl_homo_polys[3] = \sum n_i n_j n_k   n_i doesn't need to be unique
****************************************************/

void Metric_Hyperedge_Fixedcard::computeAllCompleteHomogeneousPolynomials(const std::vector<size_t>& subregion_indices, int order, std::vector<double>& compl_homo_polys)
{
  /*
  std::vector<double> power_sums;
  computeAllPowersums(subregion_indices, order, power_sums);

  compl_homo_polys.resize(order);
  for (int curr_order=0; curr_order < order; curr_order++)
  {
    compl_homo_polys[curr_order] = 0;
    for (int sum_iterator=0; sum_iterator < curr_order; sum_iterator++)
    {
      compl_homo_polys[curr_order] += power_sums[sum_iterator] * compl_homo_polys[curr_order-sum_iterator-1];
    }
    compl_homo_polys[curr_order] += power_sums[curr_order];

    compl_homo_polys[curr_order] /= (double)(curr_order+1);
  }
  */
  
  std::vector<double> elem_symm_polys;
  computeAllElementarySymmetric(subregion_indices, order, elem_symm_polys);

  compl_homo_polys.resize(order);
  for (int curr_order=0; curr_order < order; curr_order++)
  {
    double odd_even_switch = 1.0;
    compl_homo_polys[curr_order] = 0;
    for (int sum_iterator=0; sum_iterator < curr_order; sum_iterator++)
    {
      compl_homo_polys[curr_order] += odd_even_switch * elem_symm_polys[sum_iterator] * compl_homo_polys[curr_order-sum_iterator-1];
      odd_even_switch *= -1.0;
    }
    compl_homo_polys[curr_order] += odd_even_switch * elem_symm_polys[curr_order];

    //std::cout << compl_homo_polys[curr_order] << "  ";
  }
  //std::cout << std::endl;
  
}


/****************************************************
 * Computes all elementary symmetric polynomials up to specified order
 * eg elem_symm_polys[3] = \sum n_i n_j n_k   each n_i is unique
****************************************************/
void Metric_Hyperedge_Fixedcard::computeAllElementarySymmetric(const std::vector<size_t>& subregion_indices, int order, std::vector<double>& elem_symm_polys)
{
  std::vector<double> power_sums;
  computeAllPowersums(subregion_indices, order, power_sums);

  elem_symm_polys.resize(order);
  for (int curr_order=0; curr_order < order; curr_order++)
  {
    double odd_even_switch = 1.0;
    elem_symm_polys[curr_order] = 0;
    for (int sum_iterator=0; sum_iterator < curr_order; sum_iterator++)
    {
      elem_symm_polys[curr_order] += odd_even_switch * power_sums[sum_iterator] * elem_symm_polys[curr_order-sum_iterator-1];
      odd_even_switch *= -1.0;
    }
    elem_symm_polys[curr_order] += odd_even_switch * power_sums[curr_order];

    elem_symm_polys[curr_order] /= (double)(curr_order+1);

    //std::cout << elem_symm_polys[curr_order] << "  ";
  }
  //std::cout << std::endl;
}


/****************************************************
 * Computes all power sums up to specified order
 * power_sums[i] = n_1^i + n_2^i + ...
****************************************************/
void Metric_Hyperedge_Fixedcard::computeAllPowersums(const std::vector<size_t>& subregion_indices, int order, std::vector<double>& power_sums)
{
  power_sums.resize(order);
  for (std::vector<double>::iterator it_powersum = power_sums.begin() ; it_powersum != power_sums.end(); ++it_powersum)
  {
    *it_powersum = 0;
  }
  for (std::vector<size_t>::const_iterator it = subregion_indices.begin() ; it != subregion_indices.end(); ++it)
  {
    double val = _subregions[*it]._weight;
    for (std::vector<double>::iterator it_powersum = power_sums.begin() ; it_powersum != power_sums.end(); ++it_powersum)
    {
      *it_powersum += val;
      val *= _subregions[*it]._weight;
    }
  }
/*
  for (std::vector<double>::iterator it_powersum = power_sums.begin() ; it_powersum != power_sums.end(); ++it_powersum)
  {
    std::cout << (*it_powersum) << "  ";
  }
  std::cout << std::endl;
  */
}


double Metric_Hyperedge_Fixedcard::computeCurrentValue_bruteForce()
{
  std::cout << "computing brute force" << std::endl;
  //RegionList is actually a list of SUBREGIONS, not of the original regions
  std::vector<SubRegionList > allPermutations;
  generateAllPermutations(allPermutations);
  
  //prune permutations, such that the remaining lists of subregions do not share any regions
  std::vector<SubRegionList > prunedPermutations;
  removeSharedRegionPermutations(allPermutations, prunedPermutations);
  
  //sum weight of permutations
  return sumSubRegionWeights(prunedPermutations);
  //return sumSubRegionWeights(allPermutations);
}



/****************************************************
 * Generates all permutations of fixed cardinality
****************************************************/
void Metric_Hyperedge_Fixedcard::generateAllPermutations(std::vector<SubRegionList >& allPermutations)
{
  std::queue< SubRegionList > currPermutations;
  std::queue< SubRegionList > nextPermutations;
  SubRegionList empt;
  currPermutations.push(empt);
  //generate list of all region subsets
  for (size_t edgeCard_ind=0; edgeCard_ind < _edge_cardinality; edgeCard_ind++)
  {
    while (!currPermutations.empty())
    {
      SubRegionList currPerm = currPermutations.front();
      currPermutations.pop();
      
      //add next subregion index
      //only add from last index onward, to ensure no repeats
//      std::vector<SubRegion>::iterator it = (std::vector<SubRegion>::iterator) currPerm.lastSubReg();
      for (int subreg_ind= ( (currPerm.numSubRegions() == 0) ? 0 : indexFromSubRegionPointer(currPerm.lastSubReg())); subreg_ind < _subregions.size(); subreg_ind++)
      {
        SubRegionList newPerm = currPerm;
        newPerm.addSubregion(_subregions[subreg_ind]);
        nextPermutations.push(newPerm);
      }
    }
    std::swap(currPermutations, nextPermutations);
  }

  //copy everything into vector
  allPermutations.clear();
  while (!currPermutations.empty())
  {
    allPermutations.push_back(currPermutations.front());
    SubRegionList a = currPermutations.front();
    currPermutations.pop();
  }


  
}


/****************************************************
 * Removes permutations which correspond to shared regions
****************************************************/
void Metric_Hyperedge_Fixedcard::removeSharedRegionPermutations(std::vector<SubRegionList >& allPermutations, std::vector<SubRegionList>& prunedPermutations)
{
  prunedPermutations.resize(0);
  for (std::vector<SubRegionList>::iterator it = allPermutations.begin() ; it != allPermutations.end(); ++it)
  {
    if ((*it).numIntersections() == 0)
      prunedPermutations.push_back(*it);
  }

//  std::cout << "printing pruned " << std::endl;
//  for (int i=0; i < prunedPermutations.size(); i++)
//  {
//    SubRegionList a = prunedPermutations[i];
//    for (int subreg_ind = 0; subreg_ind < a.numSubRegions(); subreg_ind++)
//    {
//      std::cout << indexFromSubRegionPointer(a._subregions[subreg_ind]) << "  "; 
//    }
//    std::cout << std::endl;
//  }

}


/****************************************************
 * Sum over all weights in permutations
****************************************************/
double Metric_Hyperedge_Fixedcard::sumSubRegionWeights(std::vector<SubRegionList>& permutations)
{
  double weight_sum = 0.0;
  for (std::vector<SubRegionList>::iterator it = permutations.begin() ; it != permutations.end(); ++it)
  {
    weight_sum += (*it).productWeights();
  }

  return weight_sum;
}

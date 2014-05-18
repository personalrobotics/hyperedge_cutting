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


#ifndef METRIC_UTILS_H
#define METRIC_UTILS_H

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <list>

struct RegionList
{
  std::vector<int> _regions;

  void addRegion(int newreg);

  void getRegions(std::vector<int>& regions) const {regions = _regions;}

  bool containsRegion(int reg){return (std::find(_regions.begin(), _regions.end(), reg) != _regions.end());}

  bool isEquiv(RegionList& tocmp);
  inline int numRegions(){ return _regions.size();};
  inline int lastReg(){return _regions.back();};
  void intersectRegions(RegionList& otherList, RegionList& intersect);
  inline void clear(){_regions.clear();};
  //inline int lastReg(){return _regions[_regions.size()-1];};

};

struct SubRegion
{
  inline double getWeight(){return _weight;};
  inline int numRegions(){ return _regions.numRegions();};
  inline RegionList getRegions(){ return _regions;};

  RegionList _regions;
  double _weight;

  void intersectRegions(SubRegion& subreg, RegionList& intersect);
};


//list of subregion pointers that maintains a region intersection list
struct SubRegionList
{
  //WARNING: risky to use as pointer. Be careful that the regions never get deleted!
  std::vector<SubRegion*> _subregions;
  RegionList _region_intersections;

  inline int numSubRegions(){ return _subregions.size();};
  inline int numIntersections(){ return _region_intersections.numRegions();};
  inline SubRegion* lastSubReg(){ return _subregions.back();};
  double productWeights();

  void addSubregion(SubRegion& toAdd);

};

//Tree of subregions that maintains current weights/region lists
struct SubRegionTreeNode
{
  SubRegionTreeNode(){};
  SubRegionTreeNode(SubRegionTreeNode* prev, size_t index);

  ~SubRegionTreeNode();

  SubRegionTreeNode* _prev;
  std::vector<SubRegionTreeNode*> _nexts;

  void getAllSubregionIndsUntil(std::vector<size_t>& inds, const SubRegionTreeNode* stop_at_node);

  size_t _subregion;
};



//converts vector of region lists (one for each hypothesis) to a matrix
//matrix rows are regions, columns are particles
//so mat[i][j] is = if particle j is in region i, and 0 otherwise
void regionListsToMat(const std::vector<RegionList>& lists, std::vector< std::vector<int> >& mat);

std::ostream& operator<<(std::ostream& os, const RegionList& list);


#endif

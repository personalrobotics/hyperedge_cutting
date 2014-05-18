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
#include "MetricUtils.h"


/****************************************************
 * Adds a region, and sorts list
****************************************************/
void RegionList::addRegion(int newreg)
{
  _regions.push_back(newreg);
  std::sort(_regions.begin(), _regions.end());
}


bool RegionList::isEquiv(RegionList& tocmp)
{
  if (tocmp._regions.size() != _regions.size())
  {
    return false;
  }

  for (size_t i=0; i < _regions.size(); i++)
  {
    if (_regions[i] != tocmp._regions[i])
    {
      return false;
    }
  }
  return true;
}


void RegionList::intersectRegions(RegionList& otherList, RegionList& output)
{
  //NOTE: std::set_intersection only works if the lists are sorted!!
  //since we sort when we add, we're ok
  output._regions.resize(std::max(_regions.size(), otherList._regions.size()));
  std::vector<int>::iterator it = std::set_intersection (_regions.begin(), _regions.end(), otherList._regions.begin(), otherList._regions.end(), output._regions.begin());

  output._regions.resize(it-output._regions.begin());

  //std::cout << "size v after: " << output._regions.size() << std::endl;
//   for (it=output._regions.begin(); it!=output._regions.end(); ++it)
//     std::cout << ' ' << *it;
//   std::cout << '\n'; 
}


void SubRegion::intersectRegions(SubRegion& subreg, RegionList& intersect)
{
  _regions.intersectRegions(subreg._regions, intersect);
}


void SubRegionList::addSubregion(SubRegion& toAdd)
{
  _subregions.push_back(&toAdd);
  //WARNING can i pass in region intersections as ouput? or perhaps i need to copy
  if (_subregions.size() == 1)
    _region_intersections = _subregions.front()->_regions;
  else
  {
    RegionList tmp = _region_intersections;
    tmp.intersectRegions(toAdd._regions, _region_intersections);
  }
}


double SubRegionList::productWeights()
{
  double prod=1.0;
  for (std::vector<SubRegion*>::iterator it=_subregions.begin() ; it != _subregions.end(); ++it)
  {
    prod *= (*it)->getWeight();
  }

  return prod;

}


SubRegionTreeNode::~SubRegionTreeNode()
{
  for( std::vector<SubRegionTreeNode*>::iterator it = _nexts.begin(); it != _nexts.end(); it++)
  {
    delete (*it);
  }

}


/****************************************************
 * Goes **UP** the tree and returns all subregion_inds until the stop node
****************************************************/
void SubRegionTreeNode::getAllSubregionIndsUntil(std::vector<size_t>& inds, const SubRegionTreeNode* stop_at_node)
{
  inds.resize(0);
  inds.push_back(_subregion);
  SubRegionTreeNode* check_next = _prev;
  while (check_next != stop_at_node && check_next != NULL)
  {
    inds.push_back(check_next->_subregion);
    check_next = check_next->_prev;
  }
  std::reverse(inds.begin(), inds.end());
}

SubRegionTreeNode::SubRegionTreeNode(SubRegionTreeNode* prev, size_t index)
  : _prev(prev), _subregion(index)
{
  //_prev->_region_intersections_so_far.intersectRegions(subreg._regions, _region_intersections_so_far);
  //_weight_so_far = _prev->_weight_so_far*subreg.getWeight();
}




void regionListsToMat(const std::vector<RegionList>& lists, std::vector< std::vector<int> >& mat)
{
    //find how many regions there are by max region number
    int total_regions = 0;
    for (size_t i=0; i < lists.size(); i++)
    {
        total_regions = std::max(total_regions, *std::max_element(lists[i]._regions.begin(), lists[i]._regions.end()));
    }
    total_regions++; //add one, since the first region index is zero. So we have one more region than max region index


    //resize each row to be number of particles
    mat.resize(total_regions);
    for (size_t i=0; i < mat.size(); i++)
    {
        mat[i].resize(lists.size());
        std::fill (mat[i].begin(),mat[i].end(),0);
    }



    for (size_t part_i=0; part_i < lists.size(); part_i++)
    {
        std::vector<int> regions;
        lists[part_i].getRegions(regions);
        for (size_t region_i=0; region_i < regions.size(); region_i++)
        {
            mat[regions[region_i]][part_i] = 1;
        }
    }

}

std::ostream& operator <<(std::ostream& os, const RegionList& list)
{
  for (size_t i=0; i < list._regions.size(); i++)
  {
    os << list._regions[i] << "  ";
  }
  return os;
}

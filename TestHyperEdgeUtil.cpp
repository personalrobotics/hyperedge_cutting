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


#include <vector>
#include "Metric.h"
#include "MetricUtils.h"
#include "Metric_Hyperedge_Fixedcard.h"
#include <cstdlib> 
#include <time.h>


void testRandom();


double diffclock(clock_t clock_end,clock_t clock_begin)
{
    double diffticks=clock_end-clock_begin;
    double diffms=(diffticks)/(CLOCKS_PER_SEC/1000.0);
    return diffms;
}

void testExpectation()
{
    size_t NUM_PARTICLES_ORIG = 6;
    size_t NUM_REGIONS = 4;
    std::vector<double> particle_weights(NUM_PARTICLES_ORIG, 0.0);
    particle_weights[0] = 1.0;
    particle_weights[1] = 1.0;
    particle_weights[2] = 0.0;
    particle_weights[3] = 0.0;
    particle_weights[4] = 1.0;
    particle_weights[5] = 0.0;

    //particle_weights[2] = 1.0/((double)NUM_PARTICLES_ORIG);
    //particle_weights[3] = 1.0/((double)NUM_PARTICLES_ORIG);
    //particle_weights[5] = 1.0/((double)NUM_PARTICLES_ORIG);
    //particle_weights[6] = 1.0/((double)NUM_PARTICLES_ORIG);


    std::vector< std::vector< int> > observation_matrix;

    //two observations. Obs 1: particles 0, 1, 4. Obs 2: particles 2, 3, 5
    static const int t0_obs_arr[] = {0,0,1,1,0,1};  //two observations. Obs 1: particles 0, 1, 4. Obs 2: particles 2, 3, 5
    std::vector<int> t0_obs(t0_obs_arr, t0_obs_arr + sizeof(t0_obs_arr) / sizeof(t0_obs_arr[0]) );
    observation_matrix.push_back(t0_obs);

    //three observations. Obs 1: particles 0, 2, 5. Obs 2: particles 1, 3. Obs 3: Particle 4
    static const int t1_obs_arr[] = {0,1,0,1,2,0};
    std::vector<int> t1_obs(t1_obs_arr, t1_obs_arr + sizeof(t1_obs_arr) / sizeof(t1_obs_arr[0]) );
    observation_matrix.push_back(t1_obs);

    //three observations. Obs 1: particles 0, 3. Obs 2: particles 1, 4. Obs 3: Particles 2, 5
    static const int t2_obs_arr[] = {0,1,2,0,1,2};
    std::vector<int> t2_obs(t2_obs_arr, t2_obs_arr + sizeof(t2_obs_arr) / sizeof(t2_obs_arr[0]) );
    observation_matrix.push_back(t2_obs);


    std::vector<double> expected_values;

    std::vector<RegionList> particles(NUM_PARTICLES_ORIG);

    particles[0].addRegion(0);

    particles[1].addRegion(1);

    particles[2].addRegion(2);

    particles[3].addRegion(3);

    particles[4].addRegion(4);
    particles[4].addRegion(0);

    particles[5].addRegion(5);



    Metric_Hyperedge_Fixedcard met(particles);
    met.updateSubregionWeights(particle_weights);
    met.computeExpectedValues(particle_weights, observation_matrix, expected_values);

    std::cout << "expected values: ";
    for (std::vector<double>::iterator it = expected_values.begin(); it < expected_values.end(); it++)
    {
        std::cout << *it << "   ";
    }
    std::cout << std::endl;




}

void testRandom()
{
    size_t NUM_REGIONS_TOTAL = 8;
    size_t NUM_REGIONS_EACH_MAX = 4;
    size_t NUM_PARTICLES = 400;

    std::vector<RegionList> particles(NUM_PARTICLES);
    std::vector<double> particle_weights(NUM_PARTICLES);

    std::vector<size_t> all_reg_inds(NUM_REGIONS_TOTAL);
    for (size_t i=0; i < NUM_REGIONS_TOTAL; i++)
    {
        all_reg_inds[i] = i;
    }

    for (size_t i=0; i < NUM_PARTICLES; i++)
    {
        particle_weights[i] = 1.0/((double)NUM_PARTICLES);

        size_t last_ind_perm = (size_t)(rand()%(int)(NUM_REGIONS_EACH_MAX))+1;

        std::random_shuffle(all_reg_inds.begin(), all_reg_inds.end());
        std::cout << "particle " << i << " adding regions  ";
        for (size_t j=0; j < last_ind_perm; j++)
        {
            particles[i].addRegion(all_reg_inds[j]);
            std::cout << all_reg_inds[j] << "  ";
        }
        std::cout << std::endl;

    }


    std::cout << "creating metric: " << std::endl;
    clock_t before = clock();
    Metric_Hyperedge_Fixedcard met(particles);
    met.updateSubregionWeights(particle_weights);
    clock_t after = clock();

    std::cout << "metric created. Took " << diffclock(after, before) << " ms" << std::endl;
    std::cout << "computing value " << std::endl;

    before = clock();
    double val = met.computeCurrentValue();
    after = clock();
    std::cout << "value: " << val << std::endl;
    std::cout << "value computed. Took " << diffclock(after, before) << " ms" << std::endl;
}



int main (int argc, char *argv[])
{
    size_t NUM_REGIONS = 3;
    size_t NUM_PARTICLES = 4;

    std::vector<RegionList> particles(NUM_PARTICLES);
    particles[0].addRegion(0);

    particles[1].addRegion(1);
    particles[1].addRegion(0);

    particles[2].addRegion(1);
    particles[2].addRegion(2);

    //particles[3].addRegion(3);
    //particles[3].addRegion(2);

    particles[3].addRegion(2);
    particles[3].addRegion(1);

    //particles[5].addRegion(3);


    //particles can have arbitrary weights
    //the algorithm is equivalent if all weights sum to 1 (and are equal) or if all weights are 1. Use either
    std::vector<double> particle_weights(NUM_PARTICLES);
    for (size_t i=0; i < particles.size(); i++)
    {
        particle_weights[i] = 1.0;///((double)NUM_PARTICLES);
    }


    //initialize the metric
    std::cout << "creating subregion map " << std::endl;

    Metric_Hyperedge_Fixedcard met(particles);
    met.updateSubregionWeights(particle_weights); 

    std::cout << "computing value" << std::endl;
    double val = met.computeCurrentValue();
    std::cout << "value: " << val << std::endl;


    testExpectation();

    return 0;
}

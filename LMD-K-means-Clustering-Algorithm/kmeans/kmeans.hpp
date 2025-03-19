//
//  kmeans.hpp
//  LMD-K-means-Clustering-Algorithm
//
//  Created by Tommaso Palmisano on 3/17/25.
//

#ifndef kmeans_hpp
#define kmeans_hpp

#include <iostream>
#include <vector>
#include <cmath>
#include <arm_neon.h>
#include <arm_sve.h>


class KMeans {
private:
  float euclideanDistance(std::vector<float> v1, std::vector<float> v2);
  float optimizedEuclideanDistance(const std::vector<float>& v1, const std::vector<float>& v2);
  std::vector<float> calculateCentroid(const std::vector< std::vector<float> >& vectors);
  std::vector<float> optimizedCalculateCentroid(const std::vector< std::vector<float> >& vectors);
  
public:
  void test();
};

#endif /* kmeans_hpp */

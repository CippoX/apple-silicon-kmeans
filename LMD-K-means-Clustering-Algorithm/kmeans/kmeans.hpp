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

class KMeans {
public:
  float euclideanDistance(std::vector<float> v1, std::vector<float> v2);
};

#endif /* kmeans_hpp */

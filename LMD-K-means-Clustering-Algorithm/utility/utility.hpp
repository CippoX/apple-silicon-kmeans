//
//  utility.hpp
//  LMD-K-means-Clustering-Algorithm
//
//  Created by Tommaso Palmisano on 3/10/25.
//

#ifndef utility_hpp
#define utility_hpp

#include <iostream>
#include <fstream>
#include <vector>

void load_MNIST(    const char* images_file, const char* labels_file,
                std::vector< std::vector<float> > &images,
                std::vector<int> &labels );

#endif /* utility_hpp */

//
//  timer.hpp
//  LMD-K-means-Clustering-Algorithm
//
//  Created by Tommaso Palmisano on 3/17/25.
//

#ifndef timer_hpp
#define timer_hpp

#include <iostream>
#include <chrono>

struct Timer {
public:
  Timer();
  Timer(std::string name);
  ~Timer();
  
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimepoint;
  void Stop();
  std::string name;
};

#endif /* timer_hpp */

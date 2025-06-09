//
//  timer.cpp
//  LMD-K-means-Clustering-Algorithm
//
//  Created by Tommaso Palmisano on 3/17/25.
//

#include "timer.hpp"

Timer::Timer() {
  m_StartTimepoint = std::chrono::high_resolution_clock::now();
  name = "";
}

Timer::Timer(std::string timerName) {
  m_StartTimepoint = std::chrono::high_resolution_clock::now();
  name = timerName;
}

Timer::~Timer() {
  Stop();
}

void Timer::Stop() {
  auto endTimepoint = std::chrono::high_resolution_clock::now();
  
  auto start = std::chrono::time_point_cast<std::chrono::nanoseconds>(m_StartTimepoint).time_since_epoch().count();
  auto end = std::chrono::time_point_cast<std::chrono::nanoseconds>(endTimepoint).time_since_epoch().count();
  
  auto duration = end - start;
  auto ms = duration * 0.000001;
  
  if(name == "") {
    std::cout<<ms<<"ms\n";
  } else {
    std::cout<< name << ": " << ms <<"ms\n";
  }
}

std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimepoint;

#pragma once
#include <unordered_map>
#include <vector>
#include <fstream>
#include <memory>
#include <string>
#include <chrono>
#include "MxNetCpp.h"

std::unordered_map < std::string, std::vector <float>> get_embeddings(const std::string &path)
{
  std::unordered_map < std::string, std::vector <float>> dict;
  std::ifstream fin(path, std::ios::binary);
  size_t rows, cols;
  fin >> rows >> cols;
  LOG(INFO) << "Total rows: " << rows;
  const size_t ROW_SIZE = cols * sizeof(float);
  std::unique_ptr<char> row_buffer(new char[ROW_SIZE]);
  std::string word;
  auto start = std::chrono::steady_clock::now();
  while (!fin.eof())
  {
    fin >> word;
    char space;
    fin.get(space); // Eat space
    fin.read(row_buffer.get(), ROW_SIZE); // Extract vec
    const float *vec = reinterpret_cast<const float*>(row_buffer.get());
    dict.emplace(std::piecewise_construct,
        std::forward_as_tuple(word),
        std::forward_as_tuple(vec, vec + cols));
  }
  auto end = std::chrono::steady_clock::now();
  LG << "Read embeedings in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/1000.0 << 's';
  return dict;
}
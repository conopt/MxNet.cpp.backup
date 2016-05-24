#pragma once
#include <unordered_map>
#include <vector>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <chrono>
#include <random>
#include "MxNetCpp.h"
#include "dmlc/io.h"

class Embeddings
{
  public:
    Embeddings(const std::string &path)
    {
      std::unique_ptr<dmlc::SeekStream> fin(dmlc::SeekStream::CreateForRead(path.c_str()));
      //std::ifstream fin(path, std::ios::binary);
      size_t rows, cols;
      char ch = 'a';
      std::string tmp;
      while (ch != ' ')
      {
        fin->Read(&ch, 1);
        tmp += ch;
      }
      rows = std::stoi(tmp);
      tmp.clear();
      while (ch != '\n')
      {
        fin->Read(&ch, 1);
        tmp += ch;
      }
      cols = std::stoi(tmp);
      tmp.clear();

      cols_ = cols;
      std::cerr << "Total rows: " << rows << std::endl;
      zero_ = std::vector<float>(cols, 0.0f);
      /*
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dis(-0.25f, 0.25f);
      for (size_t i = 0; i < cols; ++i)
        zero_[i] = dis(gen);
        */


      const size_t ROW_SIZE = cols * sizeof(float);
      std::unique_ptr<char> row_buffer(new char[ROW_SIZE]);
      std::string word;

      auto start = std::chrono::steady_clock::now();
      for (size_t i = 0; i < rows; ++i)
      {
        while (true)
        {
          fin->Read(&ch, 1);
          if (ch == ' ')
            break;
          word += ch;
        }
        fin->Read(row_buffer.get(), ROW_SIZE); // Extract vec
        const float *vec = reinterpret_cast<const float*>(row_buffer.get());
        dict_.emplace(std::piecewise_construct,
          std::forward_as_tuple(word),
          std::forward_as_tuple(vec, vec + cols));
        word.clear();
      }
      auto end = std::chrono::steady_clock::now();
      std::cerr << "Read embeedings in " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0 << 's';
      std::cerr << std::endl;
    }

    size_t length() const
    {
      return cols_;
    }

    const std::vector<float>& get(const std::string &word) const
    {
      auto it = dict_.find(word);
      if (it != dict_.end())
        return it->second;
      return zero_;
    }

    mxnet::cpp::NDArray get_overlap(const std::string &overlap_word) const
    {
    }

  private:
    std::unordered_map<std::string, std::vector<float>> dict_;
    std::vector<float> zero_;
    size_t cols_;
};
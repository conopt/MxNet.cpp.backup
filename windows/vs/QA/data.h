#pragma once

#include <condition_variable>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <memory>
#include <algorithm>

#include "logging.h"
#include "dmlc/io.h"

class DataReader {
public:
  DataReader(dmlc::SeekStream* stream, int batchSize)
    : stream_(stream), batchSize_(batchSize), reset_(false),
    eof_(false), exit_(false) {
    ioThread_ = std::thread([this](){this->IOThread(); });
  }

  ~DataReader() {
    ioThread_.join();
  }

  bool Eof() {
    std::lock_guard<std::mutex> l(mutex_);
    return eof_;
  }

  void Reset() {
    reset_ = true;
    eof_ = false;
  }

  std::vector<std::tuple<std::string, std::string, float>> ReadBatch() {
    std::unique_lock<std::mutex> l(mutex_);
    if (eof_) return {};
    if (dataBuffer_.size() == 0) {
      condReady_.wait(l, [this]{ return dataBuffer_.size() > 0; });
    } 
    auto data = std::move(dataBuffer_.front());
    dataBuffer_.pop_front();
    return std::move(data);
  }

private:
  std::tuple<std::string, std::string, float> parse(std::string line) {
    // Query URL PassageID Passage Rating1 Rating2
    std::transform(line.begin(), line.end(), line.begin(), ::tolower);
    std::istringstream iss(line);
    std::vector<std::string> columns;
    for (std::string segment; std::getline(iss, segment, '\t');) {
      columns.push_back(std::move(segment));
    }
    CHECK_EQ(columns.size(), 6);
    float rating = (columns[5] == "good" || columns[5] == "perfect") ? 1 : 0;
    return std::make_tuple(columns[0], columns[3], rating);
  }

  void IOThread() {
    std::unique_ptr<char> buffer_(new char[BUFFER_CAPACITY+1]);
    eof_ = false;
    reset_ = false;
    bool first = true;
    std::string line;
    // vector of (Query, Answer, Rating)
    std::vector<std::tuple<std::string, std::string, float>> data;
    // Read until eof / reset / exit
    while (true) {
      if (reset_ || exit_) break;
      size_t bytesRead = stream_->Read(buffer_.get(), BUFFER_CAPACITY);
      if (bytesRead == 0) {
        // eof
        break;
      }
      buffer_.get()[bytesRead] = 0;
      const char *line_start = buffer_.get();
      // Handle the bytes read
      while (*line_start) {
        const char *line_end = strchr(line_start, '\n');
        if (line_end == nullptr) {
          line += line_start;
          break;
        }
        if (line_end[-1] == '\r') {
          line.insert(line.end(), line_start, line_end - 1);
        }
        else {
          line.insert(line.end(), line_start, line_end);
        }
        if (!first) {
          data.push_back(std::move(parse(line)));
          if (data.size() == batchSize_) {
            addToBuffer(std::move(data));
          }
        }
        else {
          // Ignore tsv header
          first = false;
        }
        line.clear();
        line_start = line_end + 1;
      }
    }
    eof_ = true;
  }

  void addToBuffer(std::vector<std::tuple<std::string, std::string, float>> data) {
    // Push the batch data and notify consumer.
    {
      std::unique_lock<std::mutex> l(mutex_);
      dataBuffer_.push_back(std::move(data));
    }
    condReady_.notify_one();
  }

  std::thread ioThread_;
  std::mutex mutex_;
  std::condition_variable condReady_;
  bool reset_;
  bool eof_;
  bool exit_;

  std::list<std::vector<std::tuple<std::string, std::string, float>>> dataBuffer_;
  dmlc::SeekStream* stream_;
  const int batchSize_;
  const static int BUFFER_CAPACITY = 1<<16;
};

void testDataReader(const std::string &path) {
  DataReader reader(dmlc::SeekStream::CreateForRead(path.c_str()), 300);
  while (true) {
    auto data = std::move(reader.ReadBatch());
    if (data.size() == 0) {
      break;
    }
  }
}
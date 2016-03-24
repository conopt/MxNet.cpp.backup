#pragma once

#include <condition_variable>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <memory>

#include "logging.h"
#include "dmlc/io.h"

class DataReader {
public:
  DataReader(dmlc::SeekStream* stream,
    size_t streamSize,
    int recordSize,
    int rank,
    int nsplit,
    int batchSize)
    : stream_(stream),
    streamSize_(streamSize),
    recordSize_(recordSize),
    rank_(rank),
    nsplit_(nsplit),
    batchSize_(batchSize),
    reset_(false),
    eof_(false),
    exit_(false) {
    ioThread_ = std::thread([this](){this->IOThread(); });
  }
  ~DataReader() {
    {
      std::unique_lock<std::mutex> l(mutex_);
      exit_ = true;
      condEmpty_.notify_one();
    }
    ioThread_.join();
  }

  bool Eof() {
    std::lock_guard<std::mutex> l(mutex_);
    return eof_;
  }

  void Reset() {
    std::lock_guard<std::mutex> l(mutex_);
    reset_ = true;
    eof_ = false;
    if (!buffer_.empty()) buffer_.clear();
    condEmpty_.notify_one();
  }

  std::vector<float> ReadBatch() {
    std::unique_lock<std::mutex> l(mutex_);
    std::vector<float> r;
    if (eof_) return r;
    while (buffer_.empty()) {
      condReady_.wait(l);
    }
    r.swap(buffer_);
    condEmpty_.notify_one();
    return r;
  }

private:
  void IOThread() {
    std::unique_lock<std::mutex> l(mutex_);
    size_t recordByteSize = sizeof(float)*recordSize_;
    int totalRecords = streamSize_ / recordByteSize;
    int recordCount = totalRecords / nsplit_;
    stream_->Seek(recordCount * recordByteSize * rank_);
    if (rank_ == nsplit_ - 1 && totalRecords % nsplit_ != 0) {
      recordCount += totalRecords % nsplit_;
    }
    LG << "Stream size = " << streamSize_;
    LG << "record size = " << recordByteSize;
    LG << "record count = " << recordCount;
    LG << "Nsplit = " << nsplit_;
    LG << "rank = " << rank_;
    LG << "Seeking offset = " << (recordCount * recordByteSize * rank_);
    while (!exit_) {
      eof_ = false;
      reset_ = false;
      while (recordCount > 0) {
        while (!buffer_.empty()) {
          if (reset_ || exit_) break;
          condEmpty_.wait(l);
        }
        if (reset_ || exit_) break;
        buffer_.resize(recordSize_ * batchSize_);
        size_t bytesToRead = recordByteSize * min(batchSize_, recordCount);
        LG << "Bytes to read = " << bytesToRead;
        size_t bytesRead = stream_->Read(buffer_.data(), bytesToRead);
        if (bytesRead == 0) {
          break;
        }
        CHECK_EQ(bytesRead % recordByteSize, 0);
        buffer_.resize(bytesRead / sizeof(float));
        recordCount -= bytesRead / recordByteSize;
        condReady_.notify_one();
      }
      eof_ = true;
      while (!exit_ && !reset_) condEmpty_.wait(l);
    }
  }

  std::thread ioThread_;
  std::mutex mutex_;
  std::condition_variable condReady_;
  std::condition_variable condEmpty_;
  std::vector<float> buffer_;
  bool reset_;
  bool eof_;
  bool exit_;

  const size_t streamSize_;
  const int recordSize_;
  const int rank_;
  const int nsplit_;
  const int batchSize_;
  dmlc::SeekStream* stream_;
};
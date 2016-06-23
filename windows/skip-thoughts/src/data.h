#pragma once

#include <condition_variable>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <regex>
#include <memory>
#include <algorithm>

#include "utils.hpp"
#include "mxnet-cpp/logging.h"
#include "dmlc/io.h"

typedef std::tuple<std::vector<std::string>, std::vector<std::string>,
  std::vector<int>, std::vector<int>, float> QAItem;

class DataReader
{
public:
  static DataReader* Create(const std::string &filename, size_t batchSize);

  DataReader(const std::string &filename, size_t batchSize)
    : stream_(dmlc::SeekStream::CreateForRead(filename.c_str())),
      batchSize_(batchSize),
      reset_(false),
      eof_(false),
      exit_(false)
  { }

  ~DataReader()
  {
    exit_ = true;
    ioThread_.join();
  }

  bool Eof()
  {
    std::lock_guard<std::mutex> l(mutex_);
    return eof_;
  }

  void Reset()
  {
    std::unique_lock<std::mutex> l(mutex_);
    reset_ = true;
    eof_ = false;
  }

  std::vector<QAItem> ReadBatch()
  {
    //std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::unique_lock<std::mutex> l(mutex_);
    if (dataBuffer_.size() == 0)
    {
      if (eof_) return {};
      condReady_.wait(l, [this]{ return dataBuffer_.size() > 0; });
    } 
    auto data = std::move(dataBuffer_.front());
    dataBuffer_.pop_front();
    return std::move(data);
  }

protected:
  virtual void clean_() = 0;

  QAItem add_overlap(std::vector<std::string> question, std::vector<std::string> answer, float rating)
  {
    std::vector<int> q_overlap(question.size());
    std::vector<int> a_overlap(answer.size());
    std::set<std::string> answer_set(answer.begin(), answer.end());
    std::set<std::string> question_set(question.begin(), question.end());
    for (size_t i = 0; i < question.size(); ++i)
      q_overlap[i] = answer_set.count(question[i]) > 0 ? 1 : 0;
    for (size_t i = 0; i < answer.size(); ++i)
      a_overlap[i] = question_set.count(answer[i]) > 0 ? 1 : 0;
    return make_tuple(std::move(question), std::move(answer),
      std::move(q_overlap), std::move(a_overlap), rating);
  }

  virtual void HandleBytes(const char*, size_t) = 0;

  void IOThread()
  {
    std::unique_ptr<char> buffer_(new char[BUFFER_CAPACITY+1]);
    eof_ = false;
    reset_ = false;
    while (!exit_)
    {
      if (eof_) continue; // Idle
      if (reset_)
      {
        stream_->Seek(0);
        dataBuffer_.clear();
        clean_();
        reset_ = false;
      }
      size_t len = stream_->Read(buffer_.get(), BUFFER_CAPACITY);
      buffer_.get()[len] = 0;
      HandleBytes(buffer_.get(), len);
      if (len == 0)
        eof_ = true;
    }
  }

  void addToBuffer(std::vector<QAItem>&& data)
  {
    // Push the batch data and notify consumer.
    {
      std::unique_lock<std::mutex> l(mutex_);
      dataBuffer_.push_back(data);
    }
    condReady_.notify_one();
  }

  std::thread ioThread_;
  std::list<std::vector<QAItem>> dataBuffer_;
  const size_t batchSize_;
  int BUFFER_CAPACITY = 1<<16;

private:
  std::unique_ptr<dmlc::SeekStream> stream_;
  std::mutex mutex_;
  std::condition_variable condReady_;
  bool reset_;
  bool eof_;
  bool exit_;
};

class LineDataReader : public DataReader
{
  public:
    LineDataReader(const std::string &filename, size_t batchSize) :
      DataReader(filename, batchSize)
    {
    }

  protected:
    virtual void addLine(const std::string& line) = 0;
    std::string line_;

    // vector of (Query, Answer, Rating)
    std::vector<QAItem> data_;

    void clean_() override
    {
      line_ = "";
      data_.clear();
    }

    void HandleBytes(const char* bytes, size_t len) override
    {
      if (len == 0)
      {
        // Extra data
        if (!line_.empty())
          addLine(line_);
        addLine(""); // to signal eof
        if (!data_.empty())
          addToBuffer(std::move(data_));
      }
      const char *line_start = bytes;
      // Extract lines from bytes read, leave the remaining part to next read
      while (*line_start)
      {
        const char *line_end = strchr(line_start, '\n');
        if (line_end == nullptr)
        {
          line_ += line_start;
          break;
        }
        if (line_end[-1] == '\r')
          line_.insert(line_.end(), line_start, line_end - 1);
        else
          line_.insert(line_.end(), line_start, line_end);
        line_start = line_end + 1;
        if (!line_.empty())
          addLine(line_);
        line_.clear();
      }
    }
};

class TSVDataReader : public LineDataReader
{
  public:
    TSVDataReader(const std::string &filename, size_t batchSize) :
      LineDataReader(filename, batchSize)
    {
      ioThread_ = std::thread([this]()
      {
        this->IOThread();
      });
    }
  private:
    QAItem parse(std::string line)
    {
      // Query URL PassageID Passage Rating1 Rating2
      std::transform(line.begin(), line.end(), line.begin(), ::tolower);
      std::istringstream iss(line);
      std::vector<std::string> columns;
      for (std::string segment; std::getline(iss, segment, '\t');)
      {
        columns.push_back(std::move(segment));
      }
      CHECK_EQ(columns.size(), 6);
      float rating = columns[5] == "bad" ? 0 : 1;
      return add_overlap(split_words(columns[0]), split_words(columns[3]), rating);
    }
  private:
    bool first_ = true;
    void addLine(const std::string &line)
    {
      if (first_)
      {
        first_ = false;
        return;
      }
      if (line.empty()) // eof
      {
        first_ = true;
        return;
      }
      data_.push_back(std::move(parse(line)));
      if (data_.size() == batchSize_)
      {
        addToBuffer(std::move(data_));
        data_.clear();
      }
    };
};

class XMLDataReader : public LineDataReader
{
  public:
    XMLDataReader(const std::string &filename, size_t batchSize):
      LineDataReader(filename, batchSize)
    {
      ioThread_ = std::thread([this]()
      {
        this->IOThread();
      });
    }
  private:
    std::string prev_;
    std::vector<std::string> current_question_;
  protected:
    void addLine(const std::string &line) override
    {
      if (line.empty())
      {
        prev_ = "";
        return;
      }
      if (prev_ == "<question>")
      {
        std::string lower_line = line;
        std::transform(line.begin(), line.end(), lower_line.begin(), ::tolower);
        current_question_ = split(lower_line, '\t');
      }
      int rating = -1;
      if (prev_ == "<positive>")
        rating = 1;
      if (prev_ == "<negative>")
        rating = 0;
      if (rating >= 0)
      {
        std::string lower_line = line;
        std::transform(line.begin(), line.end(), lower_line.begin(), ::tolower);
        data_.push_back(std::move(
          add_overlap(current_question_, std::move(split(lower_line, '\t')), rating)));
        if (data_.size() == batchSize_)
        {
          addToBuffer(std::move(data_));
          data_.clear();
        }
      }
      prev_ = line;
    };
};

DataReader* DataReader::Create(const std::string &filename, size_t batchSize) {
  std::string ext = filename.substr(filename.find_last_of('.') + 1);
  if (ext == "xml")
    return new XMLDataReader(filename, batchSize);
  if (ext == "tsv")
    return new TSVDataReader(filename, batchSize);
  return nullptr;
}

void testDataReader(const std::string &path) {
  using namespace std;
  auto reader = DataReader::Create(path.c_str(), 300);
  size_t total = 0;
  while (true) {
    auto data = std::move(reader->ReadBatch());
    total += data.size();
    if (data.size() == 0) {
      break;
    }
    auto item = data[0];
    for (auto w : get<0>(item)) cerr << w << ' ';
    cerr << endl;
    for (auto w : get<1>(item)) cerr << w << ' ';
    cerr << endl;
    for (auto w : get<2>(item)) cerr << w << ' ';
    cerr << endl;
    for (auto w : get<3>(item)) cerr << w << ' ';
    cerr << endl << get<4>(item) << endl;
  }
  LOG(INFO) << "Total Data Read: " << total;
}
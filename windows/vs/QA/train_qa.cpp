#include <cstdint>
#include <tuple>
#include <utility>
#include "utils.hpp"
#include "MxNetCpp.h"
#include "data.h"
#include "embedding.hpp"
#include "dmlc/io.h"

using namespace std;
using namespace mxnet::cpp;

const int batch_size = 300;

pair<float, float> get_overlap(vector<string> question, vector<string> answer,
  const unordered_map<string, float>& word2df, const set<string>& stopwords)
{
  auto q_wordset = set<string>(question.begin(), question.end());
  auto a_wordset = set<string>(answer.begin(), answer.end());
  set<string> intersection;
  set_intersection(q_wordset.begin(), q_wordset.end(), a_wordset.begin(), a_wordset.end(),
    back_inserter(intersection));
  float overlap = intersection.size() * 1.0f / (q_wordset.size() * a_wordset.size());

  float df_overlap = 0.0f;
  for (const auto &word : intersection)
  {
    auto it = word2df.find(word);
    df_overlap += it != word2df.end() ? it->second : 0;
  }
  df_overlap /= q_wordset.size() + a_wordset.size();
  return make_pair(overlap, df_overlap);
}

void handle_batch(vector<QAItem> batch)
{
  for (const QAItem& item : batch)
  {
    vector<string> question = split_words(get<0>(item));
    vector<string> answer = split_words(get<1>(item));
    float rating = get<2>(item);
  }
}

// argv[1]: data path, argv[2]: embedding path
int main(int argc, char *argv[])
{
  dmlc::SeekStream* data_stream = dmlc::SeekStream::CreateForRead(argv[1]);
  DataReader reader(data_stream, batch_size);
  auto word2vec = get_embeddings(argv[2]);
}
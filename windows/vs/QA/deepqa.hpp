#pragma once
#include <array>
#include <cstdint>
#include <tuple>
#include <utility>
#include <numeric>
#include "MxNetCpp.h"
#include "data.h"
#include "dmlc/io.h"
#include "embeddings.hpp"
#include "utils.hpp"
#include "embeddings.hpp"
using namespace std;
using namespace mxnet::cpp;

class DeepQA
{
public:
  DeepQA(KVStore kv, const string& data_path, const string &embeddings_path) :
    kv_(move(kv)),
    reader_(DataReader::Create(data_path, BATCH_SIZE)),
    word2vec_(embeddings_path),
    context_(Context(DeviceType::kCPU, 0))
  {
    std::unique_ptr<Optimizer> opt(new Optimizer("ccsgd", 1e-2, 1e-5));
    (*opt).SetParam("momentum", 0.9)
      .SetParam("rescale_grad", 1.0 / (kv_.GetNumWorkers() * BATCH_SIZE));
    kv_.SetOptimizer(std::move(opt));
  }
      /*
      auto sim_weight = Symbol::Variable("sim_weight"); // NUM_FILTER * NUM_FILTER
      auto sim = FullyConnected("sim", FullyConnected("qM", q_pooling_2d, sim_weight, NUM_FILTER),
        a_pooling_2d_as_weight, BATCH_SIZE);
      //auto sim = FullyConnected("sim", q_pooling_2d, sim_weight, NUM_FILTER);
      */

      //auto join_layer = Concat("q_sim_a", { q_pooling_2d, sim, a_pooling_2d }, 3, 1);
  void run(const string& validation_path = "")
  {
    std::vector<mx_float> questions, answers, q_overlap, a_overlap, ratings, predictions;
    questions.reserve(BATCH_SIZE * SENTENCE_LENGTH * word2vec_.length());
    answers.reserve(BATCH_SIZE * SENTENCE_LENGTH * word2vec_.length());
    q_overlap.reserve(BATCH_SIZE * SENTENCE_LENGTH * 3);
    a_overlap.reserve(BATCH_SIZE * SENTENCE_LENGTH * 3);
    ratings.reserve(BATCH_SIZE * SENTENCE_LENGTH);

    vector<NDArray> q_array_v, a_array_v, q_overlap_array_v, a_overlap_array_v, r_array_v;
    NDArray r_array_v_merge;

    if (!validation_path.empty())
    {
      unique_ptr<DataReader> validation_reader(DataReader::Create(validation_path, BATCH_SIZE));
      vector<mx_float> all_ratings;
      while (true)
      {
        auto validation_data = validation_reader->ReadBatch();
        if (validation_data.size() != BATCH_SIZE)
          break;
        questions.clear(); answers.clear(); q_overlap.clear(); a_overlap.clear(); ratings.clear();
        for (const auto &item : validation_data)
        {
          seq2vec(questions, get<0>(item), SENTENCE_LENGTH);
          seq2vec(answers, get<1>(item), SENTENCE_LENGTH);
          overlap2vec(q_overlap, get<2>(item), SENTENCE_LENGTH);
          overlap2vec(a_overlap, get<3>(item), SENTENCE_LENGTH);
          ratings.push_back(get<4>(item));
          all_ratings.push_back(get<4>(item));
        }
        NDArray q_array(vector<mx_uint>{
          BATCH_SIZE,
            1,
            SENTENCE_LENGTH,
            (mx_uint)word2vec_.length()}, context_, false);
        q_array.SyncCopyFromCPU(questions);
        q_array_v.push_back(move(q_array));

        NDArray q_overlap_array(vector<mx_uint>{
          BATCH_SIZE * SENTENCE_LENGTH,
            3,
        }, context_, false);
        q_overlap_array.SyncCopyFromCPU(q_overlap);
        q_overlap_array_v.push_back(move(q_overlap_array));

        NDArray a_array(vector<mx_uint>{
          BATCH_SIZE,
            1,
            SENTENCE_LENGTH,
            (mx_uint)word2vec_.length()}, context_, false);
        a_array.SyncCopyFromCPU(answers);
        a_array_v.push_back(move(a_array));

        NDArray a_overlap_array(vector<mx_uint>{
          BATCH_SIZE * SENTENCE_LENGTH,
            3
        }, context_, false);
        a_overlap_array.SyncCopyFromCPU(a_overlap);
        a_overlap_array_v.push_back(move(a_overlap_array));

        NDArray r_array(Shape(BATCH_SIZE), context_, false);
        r_array.SyncCopyFromCPU(ratings);
        r_array_v.push_back(move(r_array));
      }
      r_array_v_merge = NDArray(Shape(q_array_v.size() * BATCH_SIZE), context_, false);
      r_array_v_merge.SyncCopyFromCPU(all_ratings);
      predictions.reserve(q_array_v.size() * BATCH_SIZE);
    }
    cerr << "Validation Size = " << q_array_v.size() << endl;

    bool init_kv = false;

    const ActivationActType ActivationType = ActivationActType::tanh;

    map<string, NDArray> args;
    // Net
    NDArray overlap_emb_array = generate_overlap_emb();
    auto overlap_emb = Symbol::Variable("overlap_emb");
    args.emplace(overlap_emb.name(), overlap_emb_array);

    auto q_input = Symbol::Variable("q_input");
    auto q_overlap_input = Symbol::Variable("q_overlap_input");
    auto q_embedded_overlap = OverlapEmbeddingLayer("q", q_overlap_input, overlap_emb, args);
    auto q_conv = ConvLayer("q", q_input, q_embedded_overlap,
      ActivationType, PoolingPoolType::max, false, args);

    auto a_input = Symbol::Variable("a_input");
    auto a_overlap_input = Symbol::Variable("a_overlap_input");
    auto a_embedded_overlap = OverlapEmbeddingLayer("a", a_overlap_input, overlap_emb, args);
    auto a_conv = ConvLayer("a", a_input, a_embedded_overlap,
      ActivationType, PoolingPoolType::max, false, args);

    auto labels = Symbol::Variable("labels");
    auto output = ClassificationLayer(q_conv, a_conv, labels, ActivationType, args);

    map<string, OpReqType> reqtypes;
    reqtypes[q_input.name()] = OpReqType::kNullOp;
    reqtypes[a_input.name()] = OpReqType::kNullOp;
    reqtypes[q_overlap_input.name()] = OpReqType::kNullOp;
    reqtypes[a_overlap_input.name()] = OpReqType::kNullOp;
    reqtypes[labels.name()] = OpReqType::kNullOp;

    auto time_start = chrono::high_resolution_clock::now();
    size_t processed = 0;
    auto t1 = chrono::high_resolution_clock::now();

    for (size_t epoch = 0; epoch < 5; ++epoch)
    {
      reader_->Reset();
      size_t batch_count = 0;
      while (true)
      {
        cerr << '.';
        ++batch_count;
        auto batch = reader_->ReadBatch();
        //LG << "Destruct in" << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - t1).count() / 1000.0 << "s";
        t1 = chrono::high_resolution_clock::now();
        //LG << "Read data batch in " << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - t1).count() / 1000.0 << "s";
        if (batch.size() != BATCH_SIZE)
          break;
        processed += BATCH_SIZE;
        t1 = chrono::high_resolution_clock::now();
        questions.clear(); answers.clear(); q_overlap.clear(); a_overlap.clear(); ratings.clear();
        //LG << "clear last vector in " << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - t1).count() / 1000.0 << "s";
        for (const auto &item : batch)
        {
          seq2vec(questions, get<0>(item), SENTENCE_LENGTH);
          seq2vec(answers, get<1>(item), SENTENCE_LENGTH);
          overlap2vec(q_overlap, get<2>(item), SENTENCE_LENGTH);
          overlap2vec(a_overlap, get<3>(item), SENTENCE_LENGTH);
          ratings.push_back(get<4>(item));
        }
        NDArray q_array(vector<mx_uint>{ BATCH_SIZE, 1, SENTENCE_LENGTH,
          (mx_uint)word2vec_.length()}, context_, false);
        NDArray q_overlap_array(vector<mx_uint>{ BATCH_SIZE * SENTENCE_LENGTH, 3 }, context_, false);
        NDArray a_array(vector<mx_uint>{ BATCH_SIZE, 1, SENTENCE_LENGTH, 
          (mx_uint)word2vec_.length()}, context_, false);
        NDArray a_overlap_array(vector<mx_uint>{ BATCH_SIZE * SENTENCE_LENGTH, 3 }, context_, false);
        NDArray r_array(Shape(BATCH_SIZE), context_, false);
        q_array.SyncCopyFromCPU(questions);
        q_overlap_array.SyncCopyFromCPU(q_overlap);
        a_array.SyncCopyFromCPU(answers);
        a_overlap_array.SyncCopyFromCPU(a_overlap);
        r_array.SyncCopyFromCPU(ratings);
        //LG << "preprocess data in " << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - t1).count() / 1000.0 << "s";

        t1 = chrono::high_resolution_clock::now();
        args[q_input.name()] = q_array;
        args[q_overlap_input.name()] = q_overlap_array;
        args[a_input.name()] = a_array;
        args[a_overlap_input.name()] = a_overlap_array;
        args[labels.name()] = r_array;

        unique_ptr<Executor> exe(output.SimpleBind(context_, args, {}, reqtypes));
        vector<int> indices(exe->arg_arrays.size());
        iota(indices.begin(), indices.end(), 0);
        if (!init_kv) {
          kv_.Init(indices, exe->arg_arrays);
          kv_.Pull(indices, &exe->arg_arrays);
          init_kv = true;
        }
        exe->Forward(true);
        exe->Backward();
        kv_.Push(indices, exe->grad_arrays);
        if (false)
        {
          for (const auto& params : args)
          {
            const mx_float *data = params.second.GetData();
            auto shape = params.second.GetShape();
            size_t length = accumulate(shape.begin(), shape.end(), 1, [](size_t a, size_t b) {
              return a*b;
            });
            //cerr << params.first << "=" << data[0] << endl;
            cerr << params.first << "=" << accumulate(data, data + length, 0.0f) << endl;
          }
          cerr << "----------------------------------------" << endl;
        }
        kv_.Pull(indices, &exe->arg_arrays);
        //LG << "ff bp in" << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - t1).count() / 1000.0 << "s";
        t1 = chrono::high_resolution_clock::now();
      }
      cerr << endl << batch_count << " batches" << endl;
      // Validation
      if (q_array_v.size() > 0)
      {
        t1 = chrono::high_resolution_clock::now();
        predictions.clear();
        for (size_t i = 0; i < q_array_v.size(); ++i)
        {
          args[q_input.name()] = q_array_v[i];
          args[q_overlap_input.name()] = q_overlap_array_v[i];
          args[a_input.name()] = a_array_v[i];
          args[a_overlap_input.name()] = a_overlap_array_v[i];
          args[labels.name()] = r_array_v[i];
          unique_ptr<Executor> exe(output.SimpleBind(context_, args, {}, reqtypes));
          exe->Forward(false);
          exe->outputs[0].WaitToRead();
          const mx_float *result = exe->outputs[0].GetData();
          predictions.insert(predictions.end(), result, result + BATCH_SIZE);
        }
        NDArray result(Shape(q_array_v.size() * BATCH_SIZE), context_, false);
        result.SyncCopyFromCPU(predictions);
        //LG << "Validate in" << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - t1).count() / 1000.0 << "s";
        cerr << "Epoch " << epoch << ", Dev Auc: " << auc(result, r_array_v_merge) << " Samples/s: " <<
          processed * 1000.0 / chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - time_start).count() << endl;
      }
    }
    cerr << "Training Ends";
  }

private:
  static const mx_uint BATCH_SIZE = 100;
  static const mx_uint NUM_FILTER = 100;
  static const mx_uint FILTER_WIDTH = 5;
  static const mx_uint OVERLAP_LENGTH = 5;
  static const mx_uint SENTENCE_LENGTH = 70;
  unique_ptr<DataReader> reader_;
  Embeddings word2vec_;
  Context context_;
  KVStore kv_;

  Symbol ConvLayer(string prefix, Symbol input, Symbol overlap,
      ActivationActType ActivationType, PoolingPoolType PoolType, bool padding,
      map<string, NDArray>& args)
  {
    Symbol input_overlap;
    if (padding)
    {
      auto pad = Symbol::Variable(prefix + "_pad");
      NDArray pad_array(Shape(BATCH_SIZE, 1, FILTER_WIDTH - 1, word2vec_.length()), context_);
      NDArray::SampleUniform(0, 0, &pad_array);
      args.emplace(pad.name(), pad_array);

      auto overlap_pad = Symbol::Variable(prefix + "_overlap_pad");
      NDArray overlap_pad_array(Shape(BATCH_SIZE, 1, FILTER_WIDTH - 1, OVERLAP_LENGTH), context_);
      NDArray::SampleUniform(0, 0, &overlap_pad_array);
      args.emplace(overlap_pad.name(), overlap_pad_array);

      auto input_padding = Concat(prefix + "_padding", { pad, input, pad }, 3, 2);
      auto overlap_padding = Concat(prefix + "_overlap_padding",
        { overlap_pad, overlap, overlap_pad }, 3, 2);
      input_overlap = Concat(prefix + "_concat", { input_padding, overlap_padding }, 2, 3);
    }
    else
    {
      input_overlap = Concat(prefix + "_concat", { input, overlap }, 2, 3);
    }

    auto weight = Symbol::Variable(prefix + "_weight");
    NDArray weight_array(vector<mx_uint>{NUM_FILTER, 1, FILTER_WIDTH,
      (mx_uint)word2vec_.length() + OVERLAP_LENGTH}, context_);
    //NDArray::SampleGaussian(0, 1, &weight_array);
    float bound = sqrt(1.0f / FILTER_WIDTH * (word2vec_.length() + OVERLAP_LENGTH));
    NDArray::SampleUniform(-bound, bound, &weight_array);
    args.emplace(weight.name(), weight_array);

    auto bias = Symbol::Variable(prefix + "_bias");
    NDArray bias_array(vector<mx_uint>{FILTER_WIDTH, (mx_uint)word2vec_.length() + OVERLAP_LENGTH},
      context_);
    NDArray::SampleGaussian(0, 1, &bias_array);
    args.emplace(bias.name(), bias_array);

    auto conv = Convolution(prefix + "_conv", input_overlap, weight, bias,
      Shape(FILTER_WIDTH, word2vec_.length() + OVERLAP_LENGTH), NUM_FILTER,
      Shape(1, 1), Shape(1, 1), Shape(0, 0), 1, 512, true);
    // Output Shape (with padding): batch, 1, l+w-1, 1
    //           (without padding): batch, 1, l-w+1, 1
    size_t output_length = SENTENCE_LENGTH + (padding?1:-1)*(FILTER_WIDTH - 1);
    auto act = Activation(prefix + "_activation", conv, ActivationType);
    auto pooling = Pooling(prefix + "_pooling", act,
      Shape(output_length, 1), PoolType);
    return Reshape(prefix + "_pooling_2d", pooling, Shape(BATCH_SIZE, NUM_FILTER));
  }

  Symbol ClassificationLayer(Symbol q_input, Symbol a_input, Symbol labels,
      ActivationActType ActivationType, map<string, NDArray>& args)
  {
    const mx_uint join_length = NUM_FILTER * 2;
    auto join_layer = Concat("q_sim_a", { q_input, a_input }, 2, 1);
    auto hidden_weight = Symbol::Variable("hidden_weight");
    NDArray hidden_weight_array(vector<mx_uint>{ join_length, join_length}, context_);
    float bound = sqrt(6.0f / (join_length * 2));
    NDArray::SampleUniform(-bound, bound, &hidden_weight_array);
    //NDArray::SampleGaussian(0, 1, &hidden_weight_array);
    args.emplace(hidden_weight.name(), hidden_weight_array);

    auto hidden_bias = Symbol::Variable("hidden_bias");
    NDArray hidden_bias_array(vector<mx_uint>{join_length}, context_);
    NDArray::SampleUniform(0, 0, &hidden_bias_array);
    //NDArray::SampleGaussian(0, 1, &hidden_bias_array);
    args.emplace(hidden_bias.name(), hidden_bias_array);

    auto hidden_layer = Activation("hidden_layer_act",
      FullyConnected("hidden_layer", join_layer, hidden_weight, join_length, hidden_bias),
      ActivationType);
    auto lr_weight = Symbol::Variable("lr_weight");
    NDArray lr_weight_array(vector<mx_uint>{1, join_length}, context_);
    NDArray::SampleGaussian(0, 1, &lr_weight_array);
    args.emplace(lr_weight.name(), lr_weight_array);

    auto lr_bias = Symbol::Variable("lr_bias");
    NDArray lr_bias_array(vector<mx_uint>{1}, context_);
    NDArray::SampleGaussian(0, 1, &lr_bias_array);
    args.emplace(lr_bias.name(), lr_bias_array);

    auto lr = FullyConnected("lr", hidden_layer, lr_weight, 1, lr_bias);
    return LogisticRegressionOutput("sigmoid", lr, labels);
  }

  Symbol OverlapEmbeddingLayer(string prefix, Symbol overlap, Symbol embedding,
      map<string, NDArray>& args)
  {
    return Reshape(prefix + "_overlap_emb_reshape",
      FullyConnected(prefix + "_overlap_emb", overlap, embedding, OVERLAP_LENGTH),
      Shape(BATCH_SIZE, 1, SENTENCE_LENGTH, OVERLAP_LENGTH));
  }

  mx_float auc(NDArray results, NDArray labels)
  {
    results.WaitToRead();
    const mx_float *result_data = results.GetData();
    const mx_float *label_data = labels.GetData();
    const int n = labels.GetShape()[0];
    vector<pair<mx_float, mx_float>> samples;
    for (int i = 0; i < n; ++i)
      samples.emplace_back(result_data[i], label_data[i]);
    sort(samples.begin(), samples.end());

    mx_float ones = 0;
    mx_float total = 0;
    for (const auto &s : samples)
    {
      if (s.second > 0.5)
        ones += 1;
      total += 1;
    }

    if (0 == ones || total == ones) return 1;

    double tp0, tn;
    double truePos = tp0 = ones;
    double accum = tn = 0;
    double threshold = result_data[0];

    for (const auto &s : samples)
    {
      if (s.first != threshold)
      { // threshold changes
        threshold = s.first;
        accum += tn * (truePos + tp0); //2* the area of  trapezoid
        tp0 = truePos;
        tn = 0;
      }
      if (s.second > 0.5)
        truePos -= 1;
      else
        tn += 1;
    }
    accum += tn * (truePos + tp0); // 2 * the area of trapezoid
    return accum / (2.0 * ones * (total - ones));
  }

  float aucroc(NDArray results, NDArray labels)
  {
    results.WaitToRead();
    const mx_float *result_data = results.GetData();
    const mx_float *label_data = labels.GetData();
    const int l = labels.GetShape()[0];
    const float STEP = 0.01;
    vector<pair<float, float>> coords;
    for (float threshold = STEP; threshold < 1; threshold += STEP)
    {
      int true_positive = 0;
      int false_positive = 0;
      int positives = 0;
      int negatives = 0;
      for (int i = 0; i < l; ++i)
      {
        if (label_data[i] < threshold)
          ++negatives;
        else
          ++positives;
        if (result_data[i] > threshold && label_data[i] < threshold)
          ++false_positive;
        if (result_data[i] > threshold && label_data[i] > threshold)
          ++true_positive;
      }
      coords.emplace_back(1.0*true_positive / positives, 1.0*false_positive / negatives);
    }
    return integral(coords);
  }

  float accuracy(NDArray results, NDArray labels)
  {
    results.WaitToRead();
    const mx_float *result_data = results.GetData();
    const mx_float *label_data = labels.GetData();
    size_t correct = 0;
    size_t total = labels.GetShape()[0];
    for (int i = 0; i < total; ++i)
    {
      float label = label_data[i];
      float p_label = result_data[i];
      if (label == (p_label >= 0.5))
        ++correct;
    }
    return correct * 1.0 / total;
  }

  pair<float, float> get_overlap(
    const vector<string> &question,
    const vector<string> &answer,
    const unordered_map<string, float>& word2df,
    const set<string>& stopwords)
  {
    auto q_wordset = set<string>(question.begin(), question.end());
    auto a_wordset = set<string>(answer.begin(), answer.end());
    vector<string> intersection;
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

  void seq2vec(vector<float>& result, const vector<string> &seq, size_t length)
  {
    for (size_t i = 0; i < length; ++i)
    {
      if (i < seq.size())
      {
        const auto& vec = word2vec_.get(seq[i]);
        result.insert(result.end(), vec.begin(), vec.end());
      }
      else
      {
        const auto& vec = word2vec_.get("<padding>");
        result.insert(result.end(), vec.begin(), vec.end());
      }
    }
  }

  void overlap2vec(vector<float>& result, const vector<int> &seq, size_t length)
  {
    for (size_t i = 0; i < length; ++i)
    {
      size_t value = 2;
      if (i < seq.size())
        value = seq[i];
      for (size_t j = 0; j < 3; ++j)
        if (value == j)
          result.push_back(1);
        else
          result.push_back(0);
    }
  }

  NDArray generate_overlap_emb()
  {
    NDArray overlap_emb(Shape(OVERLAP_LENGTH, 3), context_, false);
    NDArray::SampleGaussian(0, 1, &overlap_emb);
    return overlap_emb;
  }
    //vector<string> question = split_words(get<0>(item));
    //vector<string> answer = split_words(get<1>(item));
    //string rating = get<2>(item);
};
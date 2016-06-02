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

Symbol Multiply(const string& name, Symbol data, Symbol weight, Symbol bias,
    mx_uint N, mx_uint K, mx_uint M)
{
  auto swapped_weight = SwapAxis(name + "_weight_transpose", weight, 0, 1);
  return FullyConnected(name, data, swapped_weight, M, bias);
}

Symbol Multiply(const string& name, Symbol data, Symbol weight,
  mx_uint N, mx_uint K, mx_uint M)
{
  auto swapped_weight = SwapAxis(name + "_weight_transpose", weight, 0, 1);
  return FullyConnected(name, data, swapped_weight, M);
}

class DeepQA
{
public:
  DeepQA(const string& kv_mode, const string& data_path, const string &embeddings_path) :
    kv_(kv_mode),
#ifdef MXNET_USE_CUDA
    context_(Context(DeviceType::kGPU, 0))
#else
    context_(Context(DeviceType::kCPU, 0))
#endif
  {
#ifdef USE_CHANA
    if (kv_.GetType() != "local")
      kv_.RunServer();
#endif
    if (kv_.GetRank() == 0)
    {
      std::unique_ptr<Optimizer> opt(new Optimizer("adadelta", 0.1, 0));
      //(*opt).SetParam("rescale_grad", 1.0 / (kv_.GetNumWorkers()));
      kv_.SetOptimizer(std::move(opt));
    }
    word2vec_.init(embeddings_path);
    kv_.Barrier();

#define GET_ENV(entry, def) do{if (buf = getenv(#entry)) entry = stoi(buf); else entry = (def);}while(false)
    const char *buf;
    GET_ENV(BATCH_SIZE, 50);
    GET_ENV(NUM_FILTER, 100);
    GET_ENV(FILTER_WIDTH, 5);
    GET_ENV(OVERLAP_LENGTH, 5);
    GET_ENV(SENTENCE_LENGTH, 70);
    GET_ENV(NUM_EPOCH, 20);
#undef GET_ENV
    if (getenv("LOCAL"))
    {
      std::cerr << "Local: " << data_path << std::endl;
      reader_.reset(DataReader::Create(data_path, BATCH_SIZE));
    }
    else
    {
      string suffix = ".part" + to_string(kv_.GetRank()) + ".tsv";
      reader_.reset(DataReader::Create(data_path + suffix, BATCH_SIZE));
    }
  }
      /*
      auto sim_weight = Symbol::Variable("sim_weight"); // NUM_FILTER * NUM_FILTER
      auto sim = FullyConnected("sim", FullyConnected("qM", q_pooling_2d, sim_weight, NUM_FILTER),
        a_pooling_2d_as_weight, BATCH_SIZE);
      //auto sim = FullyConnected("sim", q_pooling_2d, sim_weight, NUM_FILTER);
      */

      //auto join_layer = Concat("q_sim_a", { q_pooling_2d, sim, a_pooling_2d }, 3, 1);
  void run(const string& validation_path = "", const string& weights_path = "",
      bool train_mode = true)
  {
    std::vector<mx_float> questions, answers, q_overlap, a_overlap, ratings, predictions;
    questions.reserve(BATCH_SIZE * SENTENCE_LENGTH * word2vec_.length());
    answers.reserve(BATCH_SIZE * SENTENCE_LENGTH * word2vec_.length());
    q_overlap.reserve(BATCH_SIZE * SENTENCE_LENGTH * 3);
    a_overlap.reserve(BATCH_SIZE * SENTENCE_LENGTH * 3);
    ratings.reserve(BATCH_SIZE * SENTENCE_LENGTH * 2);

    vector<NDArray> q_array_v, a_array_v, q_overlap_array_v, a_overlap_array_v, r_array_v;
    NDArray r_array_v_merge;

    if (!validation_path.empty())
    {
      unique_ptr<DataReader> validation_reader(DataReader::Create(validation_path, BATCH_SIZE));
      vector<mx_float> all_ratings;
      size_t dev_size = 0;
      while (true)
      {
        auto validation_data = validation_reader->ReadBatch();
        dev_size += validation_data.size();
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
      cerr << "Validation Size = " << dev_size << endl;
    }

    bool init_kv = false;

    const ActivationActType ActivationType = ActivationActType::tanh;

    map<string, NDArray> args;
    // Net
    auto q_input = Symbol::Variable("q_input");
    auto q_overlap_input = Symbol::Variable("q_overlap_input");

    auto q_embedded_overlap = OverlapEmbeddingLayer("q", q_overlap_input, args);
    if (!weights_path.empty())
      args["q_overlap_embedding"] = loadtxt(weights_path + "q3x5", context_, Shape(3, 5));
    
    auto q_conv = ConvLayer("q", q_input, q_embedded_overlap,
      ActivationType, PoolingPoolType::max, true, args);
    if (!weights_path.empty())
    {
      args["q_weight"] =
        loadtxt(weights_path + "q100x1x5x55", context_, Shape(100, 1, 5, 55));
      args["q_bias"] =
        loadtxt(weights_path + "q100", context_, Shape(100));
    }

    auto a_input = Symbol::Variable("a_input");
    auto a_overlap_input = Symbol::Variable("a_overlap_input");
    auto a_embedded_overlap = OverlapEmbeddingLayer("a", a_overlap_input, args);
    if (!weights_path.empty())
      args["a_overlap_embedding"] = loadtxt(weights_path + "a3x5", context_, Shape(3, 5));
    
    auto a_conv = ConvLayer("a", a_input, a_embedded_overlap,
      ActivationType, PoolingPoolType::max, true, args);
    if (!weights_path.empty())
    {
      args["a_weight"] =
        loadtxt(weights_path + "a100x1x5x55", context_, Shape(100, 1, 5, 55));
      args["a_bias"] =
        loadtxt(weights_path + "a100", context_, Shape(100));
    }

    auto labels = Symbol::Variable("labels");
    auto output = ClassificationLayer(q_conv, a_conv, labels, ActivationType, args);
    if (!weights_path.empty())
    {
      args["hidden_weight"] =
        loadtxt(weights_path + "200x200", context_, Shape(200, 200));
      args["hidden_bias"] =
        loadtxt(weights_path + "200", context_, Shape(200));
      args["lr_weight"] =
        loadtxt(weights_path + "200x2", context_, Shape(200, 2));
      args["lr_bias"] =
        loadtxt(weights_path + "2", context_, Shape(2));
    }

    map<string, OpReqType> reqtypes;
    reqtypes[q_input.name()] = OpReqType::kNullOp;
    reqtypes[a_input.name()] = OpReqType::kNullOp;
    reqtypes[q_overlap_input.name()] = OpReqType::kNullOp;
    reqtypes[a_overlap_input.name()] = OpReqType::kNullOp;
    reqtypes[labels.name()] = OpReqType::kNullOp;

    vector<int> grad_indices;
    auto argument_list = output.ListArguments();
    for (const auto &arg_name : argument_list)
      if (arg_name.substr(arg_name.length() - 3) == "pad")
        reqtypes[arg_name] = OpReqType::kNullOp;

    for (size_t i = 0; i < argument_list.size(); ++i)
      if (reqtypes.count(argument_list[i]) == 0)
        grad_indices.push_back(i);
    for (auto x : grad_indices)
      cerr << x << argument_list[x] << endl;

    size_t time_used = 0;
    size_t processed = 0;
    auto t1 = chrono::high_resolution_clock::now();

    for (size_t epoch = 0; epoch < NUM_EPOCH; ++epoch)
    {
      auto time_start = chrono::high_resolution_clock::now();
      size_t batch_processed = 0;
      if (train_mode)
      {
        reader_->Reset();
        size_t batch_count = 0;
        while (true)
        {
          if (batch_count % 10 == 0)
          {
            auto now = chrono::high_resolution_clock::now();
            auto batch_time = chrono::duration_cast<chrono::milliseconds>(now - time_start).count();
            cerr << "Worker " << kv_.GetRank() << " Batch " << batch_count <<
                " Samples/s: " << batch_processed * 1000.0 / batch_time << endl;
          }
          //cerr << '.';
          ++batch_count;
          auto batch = reader_->ReadBatch();
          //LG << "Destruct in" << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - t1).count() / 1000.0 << "s";
          t1 = chrono::high_resolution_clock::now();
          //LG << "Read data batch in " << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - t1).count() / 1000.0 << "s";
          if (batch.size() != BATCH_SIZE)
            break;
          processed += BATCH_SIZE;
          batch_processed += BATCH_SIZE;
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

          unique_ptr<Executor> exe(output.SimpleBind(context_, args, {}));//, reqtypes));
          if (!init_kv)
          {
            for (const auto& index : grad_indices)
              kv_.Init(index, exe->arg_arrays[index]);
            init_kv = true;
            for (const auto& index : grad_indices)
              kv_.Pull(index, &exe->arg_arrays[index]);
          }
          exe->Forward(true);
          exe->Backward();
          for (const auto& index : grad_indices)
            kv_.Push(index, exe->grad_arrays[index]);
          for (const auto& index : grad_indices)
            kv_.Pull(index, &exe->arg_arrays[index]);
          //LG << "ff bp in" << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - t1).count() / 1000.0 << "s";
          t1 = chrono::high_resolution_clock::now();
        }
        cerr << endl << batch_count << " batches" << endl;
      }
      auto now = chrono::high_resolution_clock::now();
      time_used += chrono::duration_cast<chrono::milliseconds>(now - time_start).count();
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
          vector<float> buffer(BATCH_SIZE * (1 + USE_SOFTMAX));
          exe->outputs[0].WaitToRead();
          exe->outputs[0].SyncCopyToCPU(buffer.data(), buffer.size());
          if (USE_SOFTMAX)
		        for (size_t j = 0; j < BATCH_SIZE*2; j += 2)
			        predictions.push_back(buffer[j + 1]);
          else
            predictions.insert(predictions.end(), buffer.begin(), buffer.begin() + BATCH_SIZE);
        }
        NDArray result(Shape(q_array_v.size() * BATCH_SIZE), context_, false);
        result.SyncCopyFromCPU(predictions);
        //for (int i = 0; i < 10; ++i) cerr << predictions[i] << ' '; cerr << endl;
        //LG << "Validate in" << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - t1).count() / 1000.0 << "s";
        cerr << "Epoch " << epoch << ", Dev Auc: " << auc(result, r_array_v_merge) << endl;
      }
    }
    cerr << "Worker " << kv_.GetRank() << " Training Ends" << endl;
    kv_.Barrier();
    cerr << "Worker " << kv_.GetRank() << " Entered Barrier" << endl;
    cerr << "Worker " << kv_.GetRank()
      << " Total Speed: " << processed * 1000.0 / time_used << endl;
  }

private:
  static const bool USE_SOFTMAX = true;
  mx_uint BATCH_SIZE;
  mx_uint NUM_FILTER;
  mx_uint FILTER_WIDTH;
  mx_uint OVERLAP_LENGTH;
  mx_uint SENTENCE_LENGTH;
  mx_uint NUM_EPOCH;
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
    float bound = sqrt(1.0f / (FILTER_WIDTH * (word2vec_.length() + OVERLAP_LENGTH)));
    NDArray::SampleUniform(-bound, bound, &weight_array);
    args.emplace(weight.name(), weight_array);

    auto bias = Symbol::Variable(prefix + "_bias");
    NDArray bias_array(Shape(NUM_FILTER), context_);
    NDArray::SampleUniform(0, 0, &bias_array);
    //NDArray::SampleGaussian(0, 1, &bias_array);
    args.emplace(bias.name(), bias_array);

    auto conv = Convolution(prefix + "_conv", input_overlap, weight, bias,
      Shape(FILTER_WIDTH, word2vec_.length() + OVERLAP_LENGTH), NUM_FILTER);
      //Shape(1, 1), Shape(1, 1), Shape(0, 0), 1, 512, true);
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
        Multiply("hidden_layer", join_layer, hidden_weight, hidden_bias,
            BATCH_SIZE, join_length, join_length),
        ActivationType);

    auto lr_weight = Symbol::Variable("lr_weight");
    NDArray lr_weight_array(Shape(join_length, 1 + USE_SOFTMAX), context_);
    //NDArray::SampleGaussian(0, 1, &lr_weight_array);
    NDArray::SampleUniform(0, 0, &lr_weight_array);
    args.emplace(lr_weight.name(), lr_weight_array);

    auto lr_bias = Symbol::Variable("lr_bias");
    NDArray lr_bias_array(Shape(1 + USE_SOFTMAX), context_);
    NDArray::SampleUniform(0, 0, &lr_bias_array);
    //NDArray::SampleGaussian(0, 1, &lr_bias_array);
    args.emplace(lr_bias.name(), lr_bias_array);

    auto lr = Multiply("lr", hidden_layer, lr_weight, lr_bias,
        join_length, join_length, 1 + USE_SOFTMAX);
    if (USE_SOFTMAX)
      return SoftmaxOutput("softmax", lr, labels, 1.0f/BATCH_SIZE);// , 1.0f, -1.0f, true);
    return LogisticRegressionOutput("sigmoid", lr, labels);
  }

  NDArray oe_;
  Symbol OverlapEmbeddingLayer(string prefix, Symbol overlap,
      map<string, NDArray>& args)
  {
    Symbol embedding(prefix + "_overlap_embedding");
    /*
    NDArray overlap_emb(Shape(3, OVERLAP_LENGTH), context_, false);
    NDArray::SampleGaussian(0, 1, &overlap_emb);
    args.emplace(embedding.name(), overlap_emb);
    */
    if (oe_.GetShape().empty())
    {
      oe_ = NDArray(Shape(3, OVERLAP_LENGTH), context_, false);
      NDArray::SampleGaussian(0, 0.25, &oe_);
      vector<float> oe_v(3 * OVERLAP_LENGTH);
      oe_.SyncCopyToCPU(oe_v.data(), oe_v.size());
      for (int i = 2 * OVERLAP_LENGTH; i < 3 * OVERLAP_LENGTH; ++i)
        oe_v[i] = 0;
      oe_.SyncCopyFromCPU(oe_v);
    }
    args.emplace(embedding.name(), oe_);

    return Reshape(prefix + "_overlap_emb_reshape",
        Multiply(prefix + "_overlap_emb", overlap, embedding,
            BATCH_SIZE * SENTENCE_LENGTH * 3, 3, OVERLAP_LENGTH),
        Shape(BATCH_SIZE, 1, SENTENCE_LENGTH, OVERLAP_LENGTH));
  }

  mx_float auc(NDArray results, NDArray labels)
  {
    results.WaitToRead();
    const int n = labels.GetShape()[0];
    vector<mx_float> result_data(n), label_data(n);
    results.SyncCopyToCPU(result_data.data(), result_data.size());
    labels.SyncCopyToCPU(label_data.data(), label_data.size());
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
#pragma warning(disable:4996)
#include "mxnet-cpp/MxNetCpp.h"
#include "data.h"
#include "model.hpp"
using namespace std;
using namespace mxnet::cpp;

vector<float> seq2onehot(const vector<float> &seq, mx_uint vocab_size)
{
  vector<float> onehot(seq.size() * vocab_size, 0.0f);
  for (size_t i = 0; i < seq.size(); ++i)
  {
    onehot[vocab_size*i + (int)seq[i]] = 1;
  }
  return move(onehot);
}

const int EMB_DIM = 1000;
const int VOCAB_SIZE = 2000;
const int BATCH_SIZE = 1;
int SEQ_LEN = 5; // Varies for different batch
const Context context(DeviceType::kCPU, 0);
// argv[1]: data path
int main(int argc, char *argv[])
{
  const vector<string> param_names{
    "Wr", "Ur", "Wrd", "Urd", "Crd",
    "Wz", "Uz", "Wzd", "Uzd", "Czd",
    "W", "U", "Wd", "Ud", "Cd",
    "V", "b"
  };
  map<string, Symbol> params;
  transform(param_names.begin(), param_names.end(), inserter(params, params.end()),
    [](const string &name) {
      return make_pair(name, Symbol::Variable(name));
    });

  map<string, NDArray> args;
  map<string, OpReqType> reqs;

  // Fake data
  vector<float> q{0,1,2,3,4};
  vector<float> a{0,4,3,2,1};

  // q, a, are one-hot representations of the input sentences
  args["q"] = NDArray(Shape(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), context, false);
  args["q"].SyncCopyFromCPU(seq2onehot(q, VOCAB_SIZE));
  reqs["q"] = OpReqType::kNullOp;

  args["a"] = NDArray(Shape(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE), context, false);
  args["a"].SyncCopyFromCPU(seq2onehot(a, VOCAB_SIZE));
  reqs["a"] = OpReqType::kNullOp;

  // embedding, ith row is the embedding of ith word in vocabulary
  args["Wemb"] = NDArray(Shape(VOCAB_SIZE, EMB_DIM), context);
  NDArray::SampleGaussian(0, 1, &args["Wemb"]);
  reqs["Wemb"] = OpReqType::kNullOp;

  // Initialize params to be trained.
  for (const auto& name : param_names)
    args[name] = NDArray(Shape(EMB_DIM, EMB_DIM), context);
  args["V"] = NDArray(Shape(EMB_DIM, VOCAB_SIZE), context);
  //args["b"] = NDArray(Shape(VOCAB_SIZE), context);

  for (const auto& name : param_names)
    NDArray::SampleGaussian(0, 1, &args[name]);

  // Build model and train
  SkipThoughtsVector model("q", "a", "Wemb", BATCH_SIZE, SEQ_LEN, EMB_DIM, VOCAB_SIZE, params);

  vector<int> indices;
  {
    auto arguments = model.loss.ListArguments();
    for (int i = 0; i < arguments.size(); ++i)
      if (reqs.count(arguments[i]) == 0)
        indices.push_back(i);
  }

  auto* exe = model.loss.SimpleBind(context, args, {}, reqs);
  KVStore kv(true, "C:\\Data\\1ser.txt");
  kv.RunServer();
  unique_ptr<Optimizer> opt(new Optimizer("ccsgd", 0.5, 0));
  kv.SetOptimizer(move(opt));
  for (int i = 0; i < 5; ++i)
  {
    if (i == 0)
    {
      for (auto id : indices)
        kv.Init(id, exe->arg_arrays[id]);
    }
    else
    {
      for (auto id : indices)
        kv.Pull(id, &exe->arg_arrays[id]);
    }
    exe->Forward(true);
    exe->outputs[0].WaitToRead();
    cerr << exe->outputs[0].GetData()[0] << endl;
    exe->Backward();
    for (auto id : indices)
      kv.Push(id, exe->grad_arrays[id]);
  }
  kv.Barrier();
  delete exe;
}
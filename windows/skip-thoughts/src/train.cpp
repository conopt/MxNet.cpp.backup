#pragma warning(disable:4996)
#include "mxnet-cpp/MxNetCpp.h"
#include "data.h"
#include "model.hpp"
#include <algorithm>
using namespace std;
using namespace mxnet::cpp;

vector<float> seq2onehot(const vector<float> &seq, mx_uint vocab_size, mx_uint pad_len)
{
  vector<float> onehot(pad_len * vocab_size, 0.0f);
  for (size_t i = 0; i < seq.size(); ++i)
  {
    onehot[vocab_size*i + (int)seq[i]] = 1;
  }
  return move(onehot);
}

int EMB_DIM = 200;
int VOCAB_SIZE = 20000;
int BATCH_SIZE = 1;
Context context(DeviceType::kCPU, 0);
bool BIDIRECTIONAL = true;

ifstream qin, qembin, ain, aembin;

pair<mx_uint, mx_uint> parse_input(map<string, NDArray>& args)
{
  char dum;
  vector<vector<float>> q, a;
  vector<float> buf;
  q.reserve(BATCH_SIZE);
  a.reserve(BATCH_SIZE);
  size_t qlen = 0, alen = 0;
  for (int i = 0; i < BATCH_SIZE; ++i)
  {
    if (qin.eof() || ain.eof())
      break;

    buf.clear();
    while (qin.peek() != '\n')
    {
      int id;
      qin >> id;
      buf.push_back(id);
    }
    qlen = max(qlen, buf.size());
    q.push_back(move(buf));
    qin.read(&dum, 1);

    buf.clear();
    while (ain.peek() != '\n')
    {
      int id;
      ain >> id;
      buf.push_back(id);
    }
    alen = max(alen, buf.size());
    a.push_back(move(buf));
    ain.read(&dum, 1);
  }

  // Ignore the last batch
  if (q.size() != BATCH_SIZE)
    return make_pair(0, 0);

  // read q, a, which are one-hot representations of the input sentences
  buf.reserve(BATCH_SIZE * max(qlen, alen) * VOCAB_SIZE);
  buf.clear();
  for (int i = 0; i < BATCH_SIZE; ++i)
  {
    auto onehot = seq2onehot(q[i], VOCAB_SIZE, qlen);
    buf.insert(buf.end(), onehot.begin(), onehot.end());
  }
  args["q"] = NDArray(Shape(BATCH_SIZE, qlen, VOCAB_SIZE), context, false);
  args["q"].SyncCopyFromCPU(buf);

  buf.clear();
  for (int i = 0; i < BATCH_SIZE; ++i)
  {
    auto onehot = seq2onehot(a[i], VOCAB_SIZE, alen);
    buf.insert(buf.end(), onehot.begin(), onehot.end());
  }
  args["a"] = NDArray(Shape(BATCH_SIZE, alen, VOCAB_SIZE), context, false);
  args["a"].SyncCopyFromCPU(buf);

  // read embedded q, a
  buf.reserve(EMB_DIM * max(qlen, alen) * BATCH_SIZE);
  buf.resize(EMB_DIM * qlen * BATCH_SIZE);
  memset(buf.data(), 0, buf.size() * sizeof(float));

  for (int i = 0; i < BATCH_SIZE; ++i)
  {
    size_t row_size = EMB_DIM * qlen * sizeof(float);
    qembin.read((char*)buf.data() + row_size * i, row_size);
  }
  args["qemb"] = NDArray(Shape(BATCH_SIZE, qlen, EMB_DIM), context, false);
  args["qemb"].SyncCopyFromCPU(buf);

  buf.resize(EMB_DIM * alen * BATCH_SIZE);
  memset(buf.data(), 0, buf.size() * sizeof(float));
  for (int i = 0; i < BATCH_SIZE; ++i)
  {
    size_t row_size = EMB_DIM * alen * sizeof(float);
    aembin.read((char*)buf.data() + row_size * i, row_size);
  }
  args["aemb"] = NDArray(Shape(BATCH_SIZE, alen, EMB_DIM), context, false);
  args["aemb"].SyncCopyFromCPU(buf);

  return make_pair(qlen, alen);
}

// argv[1]: query id path
// argv[2]: query emb path
// argv[3]: answer id path
// argv[4]: answer emb path
int main(int argc, char *argv[])
{
  qin = ifstream(argv[1]);
  qembin = ifstream(argv[2], ios::binary);
  ain = ifstream(argv[3]);
  aembin = ifstream(argv[4], ios::binary);

  char newline;
  qembin >> EMB_DIM; qembin.read(&newline, 1);
  aembin >> EMB_DIM; aembin.read(&newline, 1);

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
  reqs["q"] = OpReqType::kNullOp;
  reqs["a"] = OpReqType::kNullOp;
  reqs["qemb"] = OpReqType::kNullOp;
  reqs["aemb"] = OpReqType::kNullOp;

  // Initialize params to be trained.
  for (const auto& name : param_names)
  {
    mx_uint dim1 = EMB_DIM;
    mx_uint dim2 = EMB_DIM;
    if (BIDIRECTIONAL && name.back() == 'd')
    {
      dim2 *= 2;
      if (name.front() != 'W')
        dim1 *= 2;
    }
    args[name] = NDArray(Shape(dim1, dim2), context);
  }

  args["V"] = NDArray(Shape(EMB_DIM, VOCAB_SIZE), context);
  if (BIDIRECTIONAL)
  {
    args["V"] = NDArray(Shape(EMB_DIM*2, VOCAB_SIZE), context);
  }

  for (const auto& name : param_names)
    NDArray::SampleGaussian(0, 1, &args[name]);

  KVStore kv(true, "C:\\Data\\1ser.txt");
  kv.RunServer();
  unique_ptr<Optimizer> opt(new Optimizer("ccsgd", 0.5, 0));
  kv.SetOptimizer(move(opt));
  bool init_kv = true;
  while (true)
  {
    auto lens = parse_input(args);
    if (lens.first == 0 || lens.second == 0)
      break;

    // Build model and train
    SkipThoughtsVector model("q", "a", "qemb", "aemb",
      BATCH_SIZE, lens.first, lens.second, EMB_DIM, VOCAB_SIZE, params, BIDIRECTIONAL);

    vector<int> indices;
    {
      auto arguments = model.loss.ListArguments();
      for (int i = 0; i < arguments.size(); ++i)
        if (reqs.count(arguments[i]) == 0)
          indices.push_back(i);
    }

    auto* exe = model.loss.SimpleBind(context, args, {}, reqs);
    if (init_kv)
    {
      for (auto id : indices)
        kv.Init(id, exe->arg_arrays[id]);
      init_kv = false;
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
    delete exe;
  }
  kv.Barrier();
}
#include "deepqa.hpp"
#include <windows.h>
using namespace std;
using namespace mxnet::cpp;

void data_split(string data_dir, string name, size_t output_count)
{
  string output_dir = data_dir;
  unique_ptr<DataReader> reader(DataReader::Create(data_dir + name, 500));
  size_t total = 0;
  while (true)
  {
    auto data = reader->ReadBatch();
    total += data.size();
    cerr << data.size() << endl;
    if (data.size() == 0)
      break;
  }
  reader->Reset();
  cerr << "total pairs = " << total << endl;
  size_t piece = total / output_count;

  const string header = "Q\tA\tU\tR1\tR2\n";
  ofstream output;
  size_t need = 0;
  int current = -1;
  while (true)
  {
    auto data = reader->ReadBatch();
    if (data.size() == 0)
      break;
    auto it = data.begin();
    size_t remain = data.size();
    while (it != data.end())
    {
      if (need == 0)
      {
        ++current;
        need = piece;
        if (current == output_count - 1)
          need = total - piece * current;
        output = ofstream(output_dir + name + "." + to_string(current) + "-" + to_string(output_count) + ".tsv");
        output << header;
      }
      size_t n = min(need, remain);
      need -= n;
      remain -= n;
      while (n > 0)
      {
        const auto &q = get<0>(*it);
        for (auto& w : q)
          output << w << ' ';
        output << '\t';
        const auto &a = get<1>(*it);
        for (auto& w : a)
          output << w << ' ';
        output << "url\tr1\t" << (get<4>(*it) > 0.5 ? "perfect" : "bad");
        ++it;
        --n;
      }
    }
  }
}

void testconv()
{
  vector<float> data_v{ 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
  10,10,10,10,20,20,20,20,30,30,30,30,40,40,40,40,50,50,50,50};
  vector<float> filter_v{ 1, 2, 3, 4, 1, 1, 1, 1 };
  Context context(DeviceType::kCPU, 0);

  NDArray data_a(Shape(2, 1, 5, 4), context, false);
  data_a.SyncCopyFromCPU(data_v);

  NDArray filter_a(Shape(1, 1, 2, 4), context, false);
  filter_a.SyncCopyFromCPU(filter_v);

  NDArray bias_a(Shape(1), context);
  NDArray::SampleUniform(0, 0, &bias_a);

  Symbol data = Symbol::Variable("data");
  Symbol weight = Symbol::Variable("weight");
  Symbol bias = Symbol::Variable("bias");
  map<string, NDArray> args;
  args.emplace(data.name(), data_a);
  args.emplace(weight.name(), filter_a);
  args.emplace(bias.name(), bias_a);
  auto conv = Convolution("conv", data, weight, bias, Shape(2, 4), 1);
  auto exe = conv.SimpleBind(context, args);
  exe->Forward(false);
  exe->outputs[0].WaitToRead();
  auto result = exe->outputs[0].GetData();
  for (int i = 0; i < 8; ++i)
    cerr << result[i] << endl;
}

void local_run(int argc, char *argv[])
{
  const char *kv_env = getenv("KV_MODE");
  string kv_mode = kv_env == nullptr ? "local" : kv_env;
  DeepQA deepqa(kv_mode, argv[1], argv[2]);
  //deepqa.run(argv[3] , "E:\\v-lxini\\data\\1st_weights\\");
  deepqa.run(argv[3]);
}

void distributed_run(int argc, char*argv[])
{
  const char *kv_env = getenv("KV_MODE");
#ifdef USE_CHANA
  string kv_mode = kv_env == nullptr ? "dist_async#worker_machine_list#1" : kv_env;
#else
  string kv_mode = kv_env == nullptr ? "dist_async" : kv_env;
  const char *kv_role = getenv("DMLC_ROLE");
  if (kv_role != string("worker"))
  {
    KVStore kv(kv_mode);
    kv.RunServer();
    return;
  }
#endif
  DeepQA deepqa(kv_mode, argv[1], argv[2]);
  deepqa.run(argv[3]);
}

void testpool()
{
  ifstream in("E:\\v-lxini\\data\\weights\\q_bias_act_50x100x74x1.txt");
  vector<float> act_v;
  for (int i = 0; i < 74; ++i)
  {
    float x;
    in >> x;
    act_v.push_back(x);
  }
  Context context(DeviceType::kCPU, 0);
  NDArray act_a(Shape(1, 1, 74, 1), context, false);
  act_a.SyncCopyFromCPU(act_v);
  Symbol act = Symbol::Variable("act");
  map<string, NDArray> args;
  args[act.name()] = act_a;
  Symbol pool = Pooling("pooling", act, Shape(74, 1), PoolingPoolType::max);
  auto exe = pool.SimpleBind(context, args);
  exe->Forward(false);
  exe->outputs[0].WaitToRead();
  auto result = exe->outputs[0].GetData();
  cerr << result[0] << endl;
}

void testbackward()
{
  int b = 50;
  vector<float> x_v(b*2, 0.0f);
  Context context(DeviceType::kCPU, 0);
  NDArray x_a(Shape(b, 2), context, false);
  x_a.SyncCopyFromCPU(x_v);
  vector<float> y_v(b, 1.0f);
  NDArray y_a(Shape(b), context, false);
  y_a.SyncCopyFromCPU(y_v);
  map<string, NDArray> args;
  auto x = Symbol::Variable("x");
  args[x.name()] = x_a;
  auto y = Symbol::Variable("y");
  args[y.name()] = y_a;
  auto sm = SoftmaxOutput("softmax", x, y);
  auto exe = sm.SimpleBind(context, args);
  exe->Forward(true);
  exe->Backward();
  exe->grad_arrays[0].WaitToRead();
  vector<float> p(b*2);
  exe->grad_arrays[0].SyncCopyToCPU(p.data(), p.size());
  for (int i = 0; i < b; ++i)
  {
    cerr << p[i * 2] << ", " << p[i * 2 + 1];
    cerr << endl;
  }
}

// argv[1]: data path, argv[2]: embedding path, argv[3]: optional validation path
int main(int argc, char *argv[])
{
  /*
  data_split("E:\\v-lxini\\data\\TREC\\", "train.xml", 8);
  */
  //testconv();
  if (argc == 4)
    distributed_run(argc, argv);
  else
    local_run(argc, argv);
  //testpool();
  //testbackward();
}
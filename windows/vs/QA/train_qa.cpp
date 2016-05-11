#include "deepqa.hpp"
using namespace std;
using namespace mxnet::cpp;

// argv[1]: data path, argv[2]: embedding path, argv[3]: optional validation path
int main(int argc, char *argv[])
{
  KVStore kv;
  DeepQA deepqa(std::move(kv), argv[1], argv[2]);
  deepqa.run(argv[3]);
  getchar();
}
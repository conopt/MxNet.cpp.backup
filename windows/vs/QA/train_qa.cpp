#include <cstdint>
#include "MxNetCpp.h"
#include "data.h"
#include "dmlc/io.h"

using namespace std;
using namespace mxnet::cpp;

int main(int argc, char *argv[])
{
  testDataReader(argv[1]);
}
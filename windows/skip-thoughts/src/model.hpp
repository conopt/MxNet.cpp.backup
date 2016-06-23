#include "mxnet-cpp/MxNetCpp.h"
#include <vector>
using namespace mxnet::cpp;

struct SkipThoughtsVector
{
  struct GRU
  {
    Symbol h;
    GRU(Symbol x, Symbol last_h, std::map<std::string, Symbol>& params)
    {
      Symbol Wr = params["Wr"];
      Symbol Ur = params["Ur"];
      Symbol Wz = params["Wz"];
      Symbol Uz = params["Uz"];
      Symbol W = params["W"];
      Symbol U = params["U"];
      auto r = Activation(dot(x, Wr) + dot(last_h, Ur),
        ActivationActType::sigmoid);
      auto z = Activation(dot(x, Wz) + dot(last_h, Uz),
        ActivationActType::sigmoid);
      auto candidate = Activation(dot(x, W) + dot(last_h*r, U),
        ActivationActType::tanh);
      h = (1.0f - z)*last_h + z*candidate;
    }

    GRU(Symbol x, Symbol hi, Symbol last_h, std::map<std::string, Symbol>& params)
    {
      Symbol Wr = params["Wrd"];
      Symbol Ur = params["Urd"];
      Symbol Cr = params["Crd"];
      Symbol Wz = params["Wzd"];
      Symbol Uz = params["Uzd"];
      Symbol Cz = params["Czd"];
      Symbol W = params["Wd"];
      Symbol U = params["Ud"];
      Symbol C = params["Cd"];
      auto r = Activation(dot(x, Wr) + dot(last_h, Ur) + dot(hi, Cr),
        ActivationActType::sigmoid);
      auto z = Activation(dot(x, Wz) + dot(last_h, Uz) + dot(hi, Cz),
        ActivationActType::sigmoid);
      auto candidate = Activation(dot(x, W) + dot(last_h*r, U) + dot(hi, C),
        ActivationActType::tanh);
      h = (1.0f - z)*last_h + z*candidate;
    }
  };

  struct UniSkip
  {
    std::vector<Symbol> states;

    // data: shape(batch, len, dim)
    UniSkip(Symbol data, mx_uint len, std::map<std::string, Symbol>& params)
    {
      // (batch, len, dim) -> vector<shape(batch, dim)>
      auto words = SliceChannel(data, len, 1, true);
      // Initial state
      states.push_back(words[0]);
      for (mx_uint i = 1; i < len; ++i)
      {
        GRU gru(words[i], states.back(), params);
        states.push_back(gru.h);
      }
    }
  };

  struct Decoder
  {
    std::vector<Symbol> states;

    // data: shape(batch, len, dim)
    Decoder(Symbol data, Symbol hi, mx_uint len, std::map<std::string, Symbol>& params)
    {
      // (batch, len, dim) -> vector<shape(batch, dim)>
      auto words = SliceChannel(data, len, 1, true);
      // Initial state
      states.push_back(words[0]);
      for (mx_uint i = 1; i < len; ++i)
      {
        GRU gru(words[i], hi, states.back(), params);
        states.push_back(gru.h);
      }
    }
  };
  Symbol loss;

  // x,l,r should be one-hot
  // can also change Wemb to pre-embedded inputs (xemb, lemb, remb)
  SkipThoughtsVector(Symbol x, Symbol l, Symbol r, Symbol Wemb,
    mx_uint batch_size, mx_uint len, mx_uint emb_dim, mx_uint vocab_size,
    std::map<std::string, Symbol>& params)
  {
    auto xemb = Reshape(dot(Reshape(x, Shape(batch_size*len, vocab_size)), Wemb),
      Shape(batch_size, len, emb_dim));
    UniSkip encoder(xemb, len, params);
    auto lemb = Reshape(dot(Reshape(l, Shape(batch_size*len, vocab_size)), Wemb),
      Shape(batch_size, len, emb_dim));
    Decoder left(lemb, encoder.states.back(), len, params);
    auto remb = Reshape(dot(Reshape(r, Shape(batch_size*len, vocab_size)), Wemb),
      Shape(batch_size, len, emb_dim));
    Decoder right(remb, encoder.states.back(), len, params);

    Symbol V = params["V"];
    //Symbol b = params["b"];

    auto expand = [batch_size, emb_dim](Symbol in) {
      return Reshape(in, Shape(batch_size, 1, emb_dim));
    };
    std::vector<Symbol> tmp;
    Shape dist_shape(batch_size*len, vocab_size);

    std::transform(left.states.begin(), left.states.end(),
      inserter(tmp, tmp.end()), expand);
    auto left_h = Reshape(Concat(tmp, tmp.size(), 1), Shape(batch_size*len, emb_dim));
    //auto left_dist = SoftmaxActivation(dot(left_h, V) + b);
    auto left_dist = SoftmaxActivation(dot(left_h, V));
    auto left_prob = Reshape(l, dist_shape) * left_dist;
    auto left_cost = sum_axis(log(left_prob + 1e-6), -1);

    tmp.clear();
    std::transform(right.states.begin(), right.states.end(),
      inserter(tmp, tmp.end()), expand);
    auto right_h = Reshape(Concat(tmp, tmp.size(), 1), Shape(batch_size*len, emb_dim));
    //auto right_dist = SoftmaxActivation(dot(right_h, V) + b);
    auto right_dist = SoftmaxActivation(dot(right_h, V));
    auto right_prob = Reshape(r, dist_shape) * right_dist;
    auto right_cost = sum_axis(log(right_prob + 1e-6), -1);

    // Assume that MakeLoss would transform the input symbol into loss function.
    loss = MakeLoss(left_cost + right_cost);
  }
};
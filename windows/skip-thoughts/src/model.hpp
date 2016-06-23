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
    UniSkip(Symbol data, mx_uint len, std::map<std::string, Symbol>& params, bool reverse = false)
    {
      // (batch, len, dim) -> vector<shape(batch, dim)>
      auto words = SliceChannel(data, len, 1, true);
      // Initial state
      if (reverse)
      {
        states.push_back(words[len-1]);
        for (mx_uint i = len-2; i >=0; --i)
        {
          GRU gru(words[i], states.back(), params);
          states.push_back(gru.h);
        }
      }
      else
      {
        states.push_back(words[0]);
        for (mx_uint i = 1; i < len; ++i)
        {
          GRU gru(words[i], states.back(), params);
          states.push_back(gru.h);
        }
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
      states.push_back(hi);
      //states.push_back(words[0]);
      for (mx_uint i = 0; i < len; ++i)
      {
        GRU gru(words[i], hi, states.back(), params);
        states.push_back(gru.h);
      }
      states.erase(states.begin());
    }
  };
  Symbol loss;

  // q,a should be one-hot
  // can also change Wemb to pre-embedded inputs (qemb, aemb)
  SkipThoughtsVector(Symbol q, Symbol a, Symbol Wemb,
    mx_uint batch_size, mx_uint len, mx_uint emb_dim, mx_uint vocab_size,
    std::map<std::string, Symbol>& params)
  {
    auto qemb = Reshape(dot(Reshape(q, Shape(batch_size*len, vocab_size)), Wemb),
      Shape(batch_size, len, emb_dim));
    auto aemb = Reshape(dot(Reshape(a, Shape(batch_size*len, vocab_size)), Wemb),
      Shape(batch_size, len, emb_dim));
    Init(q, a, qemb, aemb, batch_size, len, emb_dim, vocab_size, params);
  }

  // q,a should be one-hot
  SkipThoughtsVector(Symbol q, Symbol a, Symbol qemb, Symbol aemb,
    mx_uint batch_size, mx_uint len, mx_uint emb_dim, mx_uint vocab_size,
    std::map<std::string, Symbol>& params)
  {
    Init(q, a, qemb, aemb, batch_size, len, emb_dim, vocab_size, params);
  }

  void Init(Symbol q, Symbol a, Symbol qemb, Symbol aemb,
    mx_uint batch_size, mx_uint len, mx_uint emb_dim, mx_uint vocab_size,
    std::map<std::string, Symbol>& params)
  {
    UniSkip encoder(qemb, len, params);
    Decoder decode_ans(aemb, encoder.states.back(), len, params);

    Symbol V = params["V"];
    //Symbol b = params["b"];

    auto expand = [batch_size, emb_dim](Symbol in) {
      return Reshape(in, Shape(batch_size, 1, emb_dim));
    };
    std::vector<Symbol> tmp;
    Shape dist_shape(batch_size*len, vocab_size);

    std::transform(decode_ans.states.begin(), decode_ans.states.end(),
      inserter(tmp, tmp.end()), expand);
    auto ans_h = Reshape(Concat(tmp, tmp.size(), 1), Shape(batch_size*len, emb_dim));
    //auto left_dist = SoftmaxActivation(dot(left_h, V) + b);
    auto ans_dist = SoftmaxActivation(dot(ans_h, V));
    auto ans_prob = Reshape(a, dist_shape) * ans_dist;
    auto ans_cost = sum_axis(log(ans_prob + 1e-6), -1);

    // Assume that MakeLoss would transform the input symbol into loss function.
    loss = MakeLoss(ans_cost);
  }
};
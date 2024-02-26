#include <torch/extension.h>
#include <vector>

class BACore
{
public:
  BACore(){}
  ~BACore(){}
public:
  void init(torch::Tensor _poses,
            torch::Tensor _disps,
            torch::Tensor _intrinsics,
            torch::Tensor _disps_sens,
            torch::Tensor _targets,
            torch::Tensor _weights,
            torch::Tensor _eta,
            torch::Tensor _ii,
            torch::Tensor _jj,
            const int t0,
            const int t1,
            const int iterations,
            const float lm,
            const float ep,
            const bool motion_only);
  void hessian(torch::Tensor H, torch::Tensor v);
  void optimize(torch::Tensor H, torch::Tensor v);
  std::vector<torch::Tensor> retract(torch::Tensor _dx);
  
public:
  torch::Tensor poses;
  torch::Tensor disps;
  torch::Tensor intrinsics;
  torch::Tensor disps_sens;
  torch::Tensor targets;
  torch::Tensor weights;
  torch::Tensor eta;
  torch::Tensor ii;
  torch::Tensor jj;
  int t0,t1;
  float lm, ep;

  torch::Tensor ts;
  torch::Tensor ii_exp;
  torch::Tensor jj_exp;

  std::tuple<torch::Tensor, torch::Tensor> kuniq;

  torch::Tensor kx;
  torch::Tensor kk_exp; // 不重复元素的索引
    
  torch::Tensor dx;
  torch::Tensor dz;

  // initialize buffers
  torch::Tensor Hs;  
  torch::Tensor vs;  
  torch::Tensor Eii; 
  torch::Tensor Eij; 
  torch::Tensor Cii; 
  torch::Tensor wi;  

  torch::Tensor m ;
  torch::Tensor C ;
  torch::Tensor w ;
  torch::Tensor Q ;
  torch::Tensor Ei;
  torch::Tensor E ;


};

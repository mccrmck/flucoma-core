/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

/*
This program demonstrates the use of the fluid decomposition toolbox
to apply an algorithm on an input dataset
*/

#include <flucoma/algorithms/public/MLP.hpp>
#include <flucoma/algorithms/public/SGD.hpp>
#include <flucoma/algorithms/util/NNFuncs.hpp>
#include <flucoma/clients/nrt/MLPClassifierClient.hpp>
#include <flucoma/data/FluidDataSetSampler.hpp>
#include <flucoma/data/FluidIndex.hpp>
#include <flucoma/data/FluidJSON.hpp>
#include <flucoma/data/FluidTensor.hpp>
#include <iostream>

// TODO: (either in MLP.hpp or in respective hosts)
// throw error if output size == 1?
// throw error if hiddenAct == \softmax?

int main(int argc, char* argv[])
{

  using namespace fluid;
  using namespace fluid::algorithm;
  using fluid::index;
  using Array = Eigen::ArrayXXd;
  using Vector = Eigen::ArrayXd;

  // Our 'data'
  FluidTensor<double, 2> x = {{0.6, 0.8, 0.7}};
  FluidTensor<double, 2> y = {{0, 0, 0}};

  MLP   mlp = MLP();
  index act = static_cast<index>(NNActivations::Activation::kTanh);
  index outAct = static_cast<index>(NNActivations::Activation::kSoftmax);
  // inSize, outSize, hidSize, act, outAct
  mlp.init(3, 3, {2}, act, outAct);
  FluidTensor<double, 2> layer0Coeffs = {{0.1, 0.2}, {0.3, 0.1}, {0.5, 0}};
  FluidTensor<double, 2> layer1Coeffs = {{0.1, 0.2, 0.1}, {0.2, 0.2, 0.1}};
  FluidTensor<double, 1> layer0Bias = {0.1, 0.1};
  FluidTensor<double, 1> layer1Bias = {1.0, 1.0, 0.5};

  mlp.setParameters(0, layer0Coeffs, layer0Bias, act);
  mlp.setParameters(1, layer1Coeffs, layer1Bias, outAct);

  // train for a single iteration
  SGD sgd;
  // mlp, in, out, nIter, batchSize, learnRate, momentum, valFrac
  sgd.train(mlp, x, y, 1, 1, 0.1, 0.0, 0.0);

  RealVector in{1, -0.2, 1.3}; // or whatever
  RealVector out(3);
  // in, out, startlayer, endlayer
  mlp.processFrame(in, out, 0, 2);
  std::cout << std::setprecision(12) << out << std::endl;
  std::cout << "sum: " << out[0] + out[1] + out[2] << "\n";
  return 0;
}

#include <cstddef>
#include <iostream>
#include <vector>
#include <cmath>    // For exp()
#include <cstdlib>  // For rand()
#include <ctime>    // For time()
#include <iomanip>  // For std::setw


// This is going to be on binary classification

double sigmoid(double x){
    return 1.0 / (1.0 + std::exp(-x));
}

double random_weights(){
    return ((double)rand() / RAND_MAX) - 0.5;
}


class Neuron{
    public:
        std::vector<double> weights;
        double bias;
        
        /*@param input_size The number of weights to initialize(as in every W for each x(input)) */
        Neuron(int input_size){
            bias = random_weights();
            
            for (int i = 0; i < input_size; ++i){
                weights.push_back(random_weights());
            }
        }
        
        // Dot product here and calculation
        double forward(const std::vector<double>& inputs){
            double activation = bias;
            
            for (size_t i = 0; i < weights.size(); ++i){
                activation += weights[i] * inputs[i];
            }
            
            return sigmoid(activation);
        }
        
};


/*
 * @brief Dense Layer rep.
 */
 
class Layer{
    public:
        std::vector<Neuron> neurons;
        
        Layer(int input_size, int num_nuerons){
            for (int i = 0; i < num_nuerons; ++i){
                neurons.push_back(Neuron(input_size));
            }
        }
        
        
        std::vector<double> forward(const std::vector<double>& inputs){
            std::vector<double> outputs;
            for (Neuron& neuron: neurons){
                outputs.push_back(neuron.forward(inputs));
            }
            return outputs;
        }
};


// Now let's build up our Multi-Layer Perceptron
class MLP{
    public:
    std::vector<Layer> layers;
    
    // Okay Topology used : {2, 2, 1}
    // means 2 inputs, 1 hidden layer with 2 neurons and 1 output layer with 1 neuron
    MLP(const std::vector<int>& topology){
        for (size_t i = 0; i < topology.size() - 1; ++i){
            int num_inputs = topology[i];
            int num_neurons = topology[i+1];
            layers.push_back(Layer(num_inputs, num_neurons));
        }
    }
    
    std::vector<double> predict(std::vector<double> inputs){
        for (Layer& layer: layers){
            inputs = layer.forward(inputs);
        }
        return inputs;
    }
};



int main() {
    // 1. Seed the random number generator
    // We do this once at the start of the program.
    srand(static_cast<unsigned>(time(0)));

    // 2. Generate Dummy Data (XOR Problem)
    // X = features, y = labels
    // We use vector<vector<double>> as a stand-in for a 2D tensor
    std::vector<std::vector<double>> X = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    std::vector<std::vector<double>> y = {
        {0},
        {1},
        {1},
        {0}
    };

    // 3. Define Network Topology
    // {2, 2, 1} means:
    // - Input Layer: 2 neurons (for {0,0}, {0,1}, etc.)
    // - Hidden Layer: 2 neurons
    // - Output Layer: 1 neuron (for {0} or {1})
    std::vector<int> topology = {2, 2, 1};
    MLP network(topology);

    // 4. Run Predictions (Forward Pass)
    // This shows the network's output *before* any training.
    // The results will be random (around 0.5) because the weights are random.
    std::cout << "--- Predictions (Before Training) ---" << "\n";
    std::cout << std::fixed << std::setprecision(4); // Format output to 4 decimal places

    for (size_t i = 0; i < X.size(); ++i) {
        std::vector<double> input = X[i];
        std::vector<double> target = y[i];

        // Get the prediction from the network
        std::vector<double> prediction = network.predict(input);

        std::cout << "Input:  [" << input[0] << ", " << input[1] << "]"
             << "  ->  Prediction: " << std::setw(7) << prediction[0]
             << "  (Target: " << target[0] << ")" << "\n";
    }

    return 0;
}
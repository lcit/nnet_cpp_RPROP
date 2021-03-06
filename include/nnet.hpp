/*	=========================================================================
	Author: Leonardo Citraro
	Company: 
	Filename: nnet.hpp
	Last modifed: 20.10.2015 by Leonardo Citraro
	Description: Neural network class definition

	=========================================================================

	=========================================================================
*/
#ifndef NNET_H
#define NNET_H

#include <cmath>
#include <stdio.h>
#include <memory>
#include <cstdlib>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

typedef std::vector<MatrixXd> VectorMatrixXd;

namespace neural_network{

    class nnet{

        private:
            unsigned int    nn_size;            ///< number of layers
            VectorXi        size_layers;        ///< number of neuron per layer
            double          Lambda;             ///< Regularization term
            unsigned int    epochs;             ///< number of iterations
            string          loss;               ///< cost function ('quadratic', 'logloss')
            bool            verbose;            ///< print extra infos
            int             random_state;       ///< random seed
            int             training_set_size;  ///< number of samples in the training set          
            VectorMatrixXd  W;                  ///< vector of matrices, it contains the NN weights
            VectorMatrixXd  A;                  ///< vector of matrices, it contains the NN input activation
            VectorMatrixXd  Z;                  ///< vector of matrices, it contains the NN activation
            VectorMatrixXd  delta;              ///< vector of matrices, it contains the NN deltas to get the gradients
            VectorMatrixXd  dEdW;               ///< vector of matrices, it contains the NN gradients
            VectorMatrixXd  dEdW_pre;           ///< vector of matrices, it contains the preceding NN gradients
            VectorMatrixXd  Delta;              ///< vector of matrices, it contains the RPROP Deltas 
            VectorMatrixXd  Delta_pre;          ///< vector of matrices, it contains the preceding RPROP Deltas                          
            bool            mode_test;      
            string          eval;               ///< Evaluate and display a specific score ('quadratic loss', 'logloss', 'accuracy', 'auc')
            const double    eta_plus  = 1.2;    ///< n+ RPROP    
            const double    eta_minus = 0.5;    ///< n- RPROP 
            const double    Delta_max = 50.0;   ///< RPROP Delta max bound
            const double    Delta_min = 0.0;    ///< RPROP Delta min bound
            const double    Delta_zero = 0.0125;///< RPROP Delta starting point
            
            /// Forward process
            ///
            /// @param X: train dataset (one batch)
            /// @param y: train labels (one batch)
            /// 
            /// @return prediction of the labels y (Z[-1])
            MatrixXd forward(const MatrixXd &X);
            
            /// Backpropagation process
            ///
            /// @param X: train dataset (one batch)
            /// @param y: train labels (one batch)
            /// 
            /// @return  no return
            void backpropagation(const MatrixXd &X, const MatrixXd &y);
            
            /// Calculates the resulting NN cost or score
            ///
            /// @param X: train dataset (full dataset)
            /// @param y: train labels (full dataset)
            /// 
            /// @return cost value
            double evaluation(const MatrixXd &X, const MatrixXd &y);

            /// Sigmoid function
            ///
            /// @param x: matrix
            /// 
            /// @return matrix
            MatrixXd sigmoid(MatrixXd x) const;
            
            /// Derivative of the sigmoid function
            ///
            /// @param x: matrix
            /// 
            /// @return matrix
            MatrixXd sigmoid_prime(MatrixXd x) const;
            
            /// Support function for NN initialization
            ///
            /// @param no param
            ///     
            /// @return no return    
            void init_internal_matrices();
        
        public:
            /// Constructor
            ///
            /// @param size_layers: number of neurons per layer (i.e.:[2,3,1])
            ///
            /// @return no return
            nnet(VectorXi size_layers);
            virtual ~nnet();
            
            /// Fitting the NN
            ///
            /// @param X: full training set
            /// @param y: full training labels
            /// @param X_val: full validation set
            /// @param y_val: full validation labels
            /// 
            /// @return no return
            void fit(const MatrixXd &X, const MatrixXd &y);
            void fit(const MatrixXd &X, const MatrixXd &y, const MatrixXd &X_val, const MatrixXd &y_val);
            
            /// Predict labels of X, similar with forward() 
            /// but no class members are modified
            ///
            /// @param X: full training set
            ///
            /// @return matrix of predictions
            MatrixXd predict(const MatrixXd &X);
            
            // Setters
            void set_lambda(double Lambda);
            void set_epochs(unsigned int epochs);
            void set_loss(string loss);
            void set_verbose(bool verbose);
            void set_random_state(int random_state);
            void set_W(VectorMatrixXd W);
            void set_mode_test(bool test);
            void set_eval(string eval);
            
            
            // Getters
            VectorMatrixXd get_W();
            VectorMatrixXd get_A();
            VectorMatrixXd get_Z();
            VectorMatrixXd get_delta();
            VectorMatrixXd get_dEdW();           
    };
};

#endif

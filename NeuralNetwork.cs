using System;


namespace NeuralNetwork {
    public class Network {
        public Layer[] layers;


		/*
			INIT NETWORK.
			TODO: REWORK THIS, CONNECTION IS NOT SET PROPERLY, NEEDS OUTPUT NODE.
		*/
        public Network(int[] layerLengths){

        	//First layer, this layer can't have input nodes, so a remedy is just to have a fake one.
            Node[] inputNodes = new Node[layerLengths[0]];
            layers[0] = new Layer(inputNodes);

            //Middle layers & Last layer.
            for(int i = 1; i < layerLengths.Length; i++){
				Node[] layerInputNodes = layers[i-1].nodes;
				Node[] layerNodes = new Node[layerLengths[i]];
				for(int nodeI = 0; nodeI < layerLengths[i]; nodeI++){
					Connection[] connections = new Connection[layerLengths[i-1]];
					for(int conNodeI = 0; i < layerLengths[i-1]; i++){
						connections[conNodeI]=new Connection(layerInputNodes[conNodeI]);
					}
					layerNodes[nodeI] = new Node(connections);
				}
				layers[i]=new Layer(layerNodes);
            }     
        }

		/*
			FORWARD* AND (STOCHASTIC) BACKPROPAGATION
		*/
		public double[] CalculateOutputs(double[] inputValues){
			this.layers[0].setLayerValues(inputValues);
			for(int i = 1; i < this.layers.Length; i++){
				//Different activation function for output layer.
				bool outputLayer = (i==this.layers.Length-1);
				this.layers[i].updateValues(outputLayer);
			}
			return this.layers[layers.Length-1].getLayerValues();
		}

		private void TrainIteration(){

		}

		private void Train(int epochs, double trainingStep, int batch, int iterations){
			for(int epochIndex = 0; epochIndex<epochs; epochIndex++){
				for(int iterationIndex = 0; iterationIndex<iterations; iterationIndex++){
					TrainIteration();
				} 
			}
		}

		/*
			ERROR.
		*/
		private static double errorSq(double calculated, double target){
			double diff = target-calculated;
			return (diff*diff);
		}

		private static double errorSqDerivative(double calculated, double target){
			return -2*(target-calculated); // With respect to calculated.
		}

		private static double[] errorSqArray(double[] calculated, double[] expected){
			double[] errorSqArr=new double[layerNodes.Length];
			for(int i = 0; i < calculated.Length; i++){
				errorSqArr[i]=errorSq(calculated[i], expected[i]);
			}
			return errorSqArr;
		}

		/*
			LOSS. (Loss is over a single iteration of one set expects and calculates).
		*/
		private static double MSE_LOSS(double[] calculated, double[] expected){
			double[] errorSq = errorSqArray(calculated,expected);
			double MSE = 0d;
			for(int i = 0; i < errorSq.Length; i++){
				MSE+=(errorSq[i]);
			}
			MSE/=(errorSq.Length);
			return MSE;
		}

		/*	
			COST. (Cost is over a whole dataset of expects and calculates).
		*/
		private static double MSE_COST(double[,] calculated, double[,]){
			double MSE=0d;
			int totalCounts = 0;
			for(int i = 0; i < calculated.Length; i++){
				double[] errorSq = errorSqArray(calculated,expected);
				for(int j = 0; j < errorSq.Length; j++){
					MSE+=(errorSq[j]);
				}
				totalCounts+=errorSq.Length;
			}
			MSE/=totalCounts;
			return MSE;
		}

		//Layer of nodes.
        private class Layer{
            public Node[] nodes;
            public Layer(Node[] nodes){
                this.nodes = nodes;
            }

			/*
				UPDATE LAYER DATA.
			*/

			public void updateValues(bool outputLayer){
				foreach(Node node in nodes){
					node.updateNodeValue(outputLayer);
				}
			}

			//Sets nodes values manually. Used by first layer, which doesn't have "real" input Nodes.
			public void setLayerValues(double[] values){
				for(int i = 0; i < values.Length; i++){
					nodes[i].value = values[i];
				}
			}

			/*
				GET INFO ABOUT LAYER.
			*/
			public double[] getLayerValues(){
				double[] vals = new double[nodes.Length];
				for(int i = 0; i < nodes.Length; i++){
					vals[i]=node.val;
				}
				return vals;
			}

        }

		//One node (neuron).
        private class Node {
            public Connection[] inputConnections;
			public double value;
			public double bias;
			private double biasDerivative = 0d;
			private double errorTotal= 0d;
            public Node(Connection[] inputConnections, double value = 0d, double bias = 0d, double errorTotal=0d){
                this.inputConnections = inputConnections;
				this.value = value;
				this.bias = bias;
				this.errorTotal=errorTotal;
            }

			/*
				UPDATE NODE DATA.
			*/

			public void updateBias(double trainingStep, double dBias){
				this.bias+=trainingStep*dBias;
			}

			public void updateBiasDerivative(bool outputLayer){
				double errorOutputNode = 
				double valueInputNode = 
				double valueOutputNode = 
				double activationDerivative = 
				this.biasDerivative = 
			}

			public void updateNodeValue(bool outputLayer){
				double tmpVal = this.bias; 
				foreach (Connection connection in this.inputConnections)
				{
					tmpVal+=connection.nodeIn.value*connection.weight;
				}
				this.value = Activation(tmpVal,outputLayer);
			}

			/* 
				NODE ACTIVATION FUNCTION.
			*/

			private static double Activation(double x, bool outputLayer){
				//SIGMOID
				if(outputLayer){
					return (1)/(1+Math.Exp(-x));
				}
				//RELU
				if(x>0d){
					return x;
				}
				return 0d;
			}

			private static double Activation_Derivative(double x, bool outputLayer){
				if(outputLayer){
					//SIGMOID DERIVATIVE
					double val = Activation(x,true);
					return val*(1-val);				
				}
				//RELU DERIVATIVE
				if(x>0d){
					return 1;
				}
				return 0d;
				
			}

			/*
				ERROR CALCULATION.
			*/
			//Hidden Error.
			public void setHiddenError(Connection[] connections){
				double sumAllWeights = Connection.sumAllWeights(connections);
				double hiddenError = 0d;
				for(int i = 0; i < connections.Length; i++){
					double error = connections[i].nodeOut.errorTotal*((connections[i].weight)/(sumAllWeights));
					hiddenError+=error;
				}
				this.errorTotal=hiddenError;
			}

			public void setOutputError(double target){
				this.errorTotal=errorSq(this.value,target); 
			}
        }

		//Connection from one node to another.
		private class Connection {
			public Node nodeIn;
			public Node nodeOut;
			public double weight;
			public double weightDerivative = 0d;


			//NOTE: weight is, unless specified otherwise, init randomly between [-/sqrt(InputLength),1/sqrt(InputLength)].
			public Connection(Node nodeIn,Node nodeOut, double weight = ((new Random()).NextDouble()*(2/Math.Sqrt(layers[0].Length))-(1/Math.Sqrt(layers[0].Length)))){
				this.nodeIn = nodeIn;
				this.nodeOut = nodeOut;
				this.weight = weight;
			}

			/*
				UPDATE CONNECTION DATA.
			*/
			public void updateWeight(double trainingStep, double dWeight){
				this.weight += trainingStep*dWeight;
			}

			public static double sumAllWeights(Connections[] connections){
				double totalW = 0d;
				for(int i = 0; i<connections.Length; i++){
					totalW+=connections[i].weight;
				}
				return totalW;
			}

			public void updateWeightDerivative(bool outputLayer){
				double errorOutputNode = this.nodeOut.errorTotal;
				double valueInputNode = this.nodeIn.value;
				double valueOutputNode = this.nodeOut.value;
				//TODO: CHECK IF SUPPOSED TO BE VALUEOUTPUTNODE, REMEMBER THAT IS AFTER IT HAS BEEN GONE THROUGH AN ACTIVATION FUNCTION.
				double activationDerivative = Node.Activation_Derivative(valueOutputNode,outputLayer)*valueInputNode;
				this.weightDerivative = errorOutputNode*activationDerivative;
			}
		}
    }
}
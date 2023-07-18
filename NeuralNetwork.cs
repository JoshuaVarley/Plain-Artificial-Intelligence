using System;


namespace NeuralNetwork {
    public class Network {
        public Layer[] layers;


		/*
			INIT NETWORK.
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

		private void TrainIteration(int epochs, double trainingStep, int batch){
			
		}

		/*
			ERROR.
		*/
		private static double[] errorArray(double[] calculated, double[] expected){
			double[] temp = new double[calculated.Length];
			for(int i = 0; i < calculated.Length; i++){
				temp[i]=(expected[i]-calculated[i]);
			}
			return temp;
		}


		private static double[] errorSqArray(double[] calculated, double[] expected){
			double[] errorArr = errorArray(calculated,expected);
			for(int i = 0; i < errorSqArr.Length; i++){
				errorArr[i]*=errorArr[i];
			}
			return errorArr;
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
            private Connection[] inputConnections;
			public double value;
			private double bias;
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


			public void updateNodeValue(bool outputLayer){
				double tmpVal = this.bias; 
				foreach (Connection connection in this.inputConnections)
				{
					tmpVal+=connection.node.value*connection.weight;
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
        }

		//Connection from one node to another.
		private class Connection {
			public Node node;
			public double weight;
			public double weightDerivative = 0d;


			//NOTE: weight is, unless specified otherwise, init randomly between [-1,1].
			public Connection(Node node, double weight = ((new Random()).NextDouble()*2-1)){
				this.node = node;
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
		}
    }
}
namespace NeuralNetwork {
    public class Network {
        public Layer[] layers;

        /*
            INIT NETWORK.
        */
        public Network(int[] layerLengths){
            this.layers = new Layer[layerLengths.Length];

            //INIT NODES.
            for(int i = 0; i < layerLengths.Length; i++){
                Node[] layerNodes = new Node[layerLengths[i]];
                for( int j = 0; j < layerLengths[i]; j++)
                {
                    layerNodes[j] = new Node();
                }
                layers[i] = new Layer(layerNodes, (i==layerLengths.Length-1), (i==0));
            }

            //INIT CONNECTIONS.
            for(int i = layerLengths.Length-1; i >= 1 ; i--)
            {
                Layer layer = this.layers[i];
                Layer prevLayer = this.layers[i - 1];
                foreach (Node node in layer.nodes)
                {
                    foreach (Node prevNode in prevLayer.nodes)
                    {
                        Connection con = new(prevNode, node, Connection.WeightInit(prevLayer.nodes.Length));
                        node.inputConnections.Add(con);
                        prevNode.outputConnections.Add(con);
                    }
                }
            }
        }
        


        /*
            FORWARDPROPAGATION
        */
        public double[] CalculateOutputs(double[] inputValues){
            layers[0].SetLayerValues(inputValues);
            for(int i = 1; i < layers.Length; i++){
                Layer layer = layers[i];
                layer.UpdateValues();
            }
            return layers[^1].GetLayerValues();
        }

        /*
            BACKPROPAGATION
        */
        private void UpdateGradients(double[] expected){
            for(int i = layers.Length-1; i >= 1; i--){
                Layer layer = layers[i];
                for(int j = 0; j < layer.nodes.Length; j++){
                    Node node = layer.nodes[j];
                    node.SetError(layer.outputLayer, (layer.outputLayer) ? expected[j] : 0d);
                    node.UpdateGradient(layer);
                    node.AddBiasDerivative();
                    foreach (Connection connection in node.inputConnections)
                    {
                        connection.AddWeightDerivative();
                    }
                }
            }
        }

        /*
            GRADIENT DESCENT ITERATION
        */
        private void ApplyGradients(double trainingStep, double momentum)
        {
            for (int i = layers.Length - 1; i >= 1; i--)
            {
                foreach(Node node in layers[i].nodes)
                {
                    node.UpdateBias(trainingStep, momentum);
                    foreach (Connection connection in node.inputConnections)
                    {
                        connection.UpdateWeight(trainingStep, momentum);
                    }
                }
            }
        }


        //Stochastic gradient descent iteration with momentum.
        private void MBGD(double trainingStep, double momentum, double[][] batchInput, double[][] batchOutput)
        {
            for(int i = 0; i < batchInput.Length; i++) {
                double[] sampleInput = batchInput[i];
                double[] sampleOutput = batchOutput[i];
                double[] calculates = CalculateOutputs(sampleInput);
                UpdateGradients(sampleOutput);
            }
            ApplyGradients(trainingStep/batchInput.Length, momentum);
        }

        public void Train(int epochs, int batch, double trainingStep, double momentum, double[][] trainingDataInput, double[][] trainingDataOutput, double[][] testDataInput, double[][] testDataOutput)
        {
            for (int epochI = 0; epochI < epochs; epochI++)
            {
                Console.WriteLine($"Starting epoch {epochI+1}");
                for(int iterationI = 0; iterationI+batch < trainingDataInput.Length; iterationI+=batch)
                {
                    double[][] batchInput = new double[batch][];
                    double[][] batchOutput = new double[batch][];
                    for (int batchI = 0; batchI < batch; batchI++)
                    {
                        batchInput[batchI] = trainingDataInput[batchI + iterationI];
                        batchOutput[batchI] = trainingDataOutput[batchI + iterationI];
                    }
                    MBGD(trainingStep, momentum, batchInput, batchOutput);
                }
                if (epochI != epochs - 1)
                {
                    Test(testDataInput, testDataOutput);
                }
            }
        }
        public void Test(double[][] testDataInput, double[][] testDataOutput)
        {
            double[][] calculated = new double[testDataInput.Length][];
            for(int i = 0; i < calculated.Length; i++)
            {
                calculated[i] = CalculateOutputs(testDataInput[i]);
            }

            Console.WriteLine($@"
----------------------
MSE_AVG   : {Cost.MSE.CostMultipleFunction(testDataOutput, calculated)}
----------------------");
        }
    }
}
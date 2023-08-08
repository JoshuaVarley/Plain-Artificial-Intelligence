namespace NeuralNetwork
{
    [Serializable]
    public class Node
    {
        public List<Connection> inputConnections = new();
        public List<Connection> outputConnections = new();
        public double value;
        public double bias;
        public double biasDerivative = 0d;
        public double biasVelocity = 0d;
        public double gradient = 0d;
        public double errorDer = 0d;

        public Node(double value = 0d, double bias = 0d)
        {
            this.value = value;
            this.bias = bias;
        }

        /*
            UPDATE NODE DATA.
        */
        public void UpdateBias(double trainingStep, double momentum)
        {
            double velocity = biasVelocity * momentum - biasDerivative * trainingStep;
            biasVelocity = velocity;
            bias += velocity;
            biasDerivative = 0d;
        }

        public void AddBiasDerivative()
        {
            biasDerivative += gradient;
        }

        
        public void UpdateGradient(Layer curLayer, double target = 0d)
        {
            if (curLayer.outputLayer)
            {
                gradient = Cost.CROSS_ENTROPY.CostFunctionIterationDerivative(target, value) * Activations.SoftMax.Derivative(curLayer,GetNetActivationInput());
            } else
            {
                gradient = outputConnections.Sum(con => con.nodeOut.gradient * con.weight) * Activations.RELU.Derivative(GetNetActivationInput());
            }
        }


        /*
            GET NODE DATA.
        */
        public double GetNetActivationInput()
        {
            double net = bias + inputConnections.Sum(con => con.weight*con.nodeIn.value);
            return net;
        }
    }
}


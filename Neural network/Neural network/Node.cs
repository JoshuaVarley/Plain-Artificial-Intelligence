namespace NeuralNetwork
{
    public class Node
    {
        public List<Connection> inputConnections = new();
        public List<Connection> outputConnections = new();
        public double value;
        public double bias;
        public double biasDerivative = 0d;
        public double biasVelocity = 0d;
        public double errorDerivative = 0d;
        public double gradient = 0d;

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
            double v = biasVelocity * momentum - biasDerivative * trainingStep;
            biasVelocity = v;
            bias += v;
            biasDerivative = 0d;
        }

        public void AddBiasDerivative()
        {
            biasDerivative += gradient;
        }

        public void UpdateGradient(Layer curLayer)
        {
            if (curLayer.outputLayer)
            {
                gradient = errorDerivative * Activations.SoftMax.Derivative(curLayer.GetLayerValues(),value);
            } else
            {
                gradient = errorDerivative * Activations.SILU.Derivative(value);
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

        /*
            SET ERROR.
        */
        public void SetError(bool outputLayer, double target = 0d)
        {
            //Output error.
            if (outputLayer)
            {
                errorDerivative = Cost.MSE.CostFunctionIterationDerivative(target, value);
            } else
            {
                //Hidden layer derivative.
                List<Connection> connections = outputConnections;
                double sumW = connections.Sum(a=>a.weight);
                errorDerivative = connections.Sum(con => (con.nodeOut.errorDerivative * con.weight)/sumW);
            }
        }
    }
}


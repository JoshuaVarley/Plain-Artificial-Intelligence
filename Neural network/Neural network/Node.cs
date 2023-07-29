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
        public double error = 0d;
        public double errorDerivative = 0d;
        public double gradient = 0d;

        public Node(double value = 0d, double bias = 0d, double error = 0d)
        {
            this.value = value;
            this.bias = bias;
            this.error = error;
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

        public void UpdateGradient(bool outputLayer)
        {
            gradient = errorDerivative * Activation_Derivative(value, outputLayer);
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
            NODE ACTIVATION FUNCTION.
        */
        public static double Activation(double x, bool outputLayer)
        {
            //SIGMOID
            if (outputLayer)
            {
                return (1.0d) / (1.0d + Math.Exp(-x));
            }
            //LEAKY_RELU
            if (x >= 0d)
            {
                return x;
            }
            return 0.01d * x;
        }

        public static double Activation_Derivative(double x, bool outputLayer)
        {
            //SIGMOID DERIVATIVE
            if (outputLayer)
            {
                double val = Activation(x, outputLayer);
                return val * (1.0d - val);
            }
            //LEAKY_RELU DERIVATIVE
            if (x >= 0d)
            {
                return 1d;
            }
            return 0.01d;
        }


        /*
            SET ERROR.
        */
        public void SetError(bool outputLayer, double target = 0d)
        {
            if (outputLayer)
            {
                error = Cost.CostFunctionIteration(target, value);
                errorDerivative = Cost.CostFunctionIterationDerivative(target, value);
            } else
            {
                List<Connection> connections = outputConnections;
                double sumW = connections.Sum(a=>a.weight);
                errorDerivative = connections.Sum(con => (con.nodeOut.errorDerivative * con.weight)/sumW);
            }
        }
    }
}


namespace NeuralNetwork
{
    //Connection from one node to another.
    public class Connection
    {
        public Node nodeIn;
        public Node nodeOut;
        public double weight;
        public double weightDerivative = 0d;
        public double weightVelocity = 0d;
        public Connection(Node nodeIn, Node nodeOut, double weight = 1d)
        {
            this.nodeIn = nodeIn;
            this.nodeOut = nodeOut;
            this.weight = weight;
        }

        /*
            UPDATE CONNECTION DATA.
        */
        public void AddWeightDerivative()
        {
            weightDerivative += nodeOut.gradient * nodeIn.value;
        }

        public void UpdateWeight(double trainingStep, double momentum)
        {
            double v = weightVelocity * momentum - weightDerivative * trainingStep;
            weightVelocity = v;
            weight += v;
        }

        /*
            CONNECTION WEIGHT INIT
        */
        public static double WeightInit(int prevLayerLength)
        {
            Random rand = new();
            double weight = (rand.NextDouble() * 2 - 1) / Math.Sqrt(prevLayerLength);
            return weight;
        }
    }
}

